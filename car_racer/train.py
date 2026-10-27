"""
Modular Training Pipeline for Mini World Models:
1. Train V (ConvVAE) on raw frames.
2. Train M (MDN-RNN) on sampled latent codes + actions.
3. Train C (Controller) via Evolution Strategies entirely inside M's dream.
Supports live visual rendering during learning so you can watch the car improve!
"""
import os
import sys

# Ensure repository root and package dir are in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
for p in [SCRIPT_DIR, PARENT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from car_racer.config import Config
from car_racer.dataset import get_or_create_dataset
from car_racer.models.vae import ConvVAE
from car_racer.models.mdn_rnn import MDNRNN, mdn_loss
from car_racer.models.controller import (
    Controller,
    EvolutionOptimizer,
    evaluate_dream,
    evaluate_real
)
from car_racer.visualize import PygameCarRacerViewer, save_racing_gif

def train_v(dataset,
            epochs=Config.VAE_EPOCHS,
            batch_size=Config.VAE_BATCH_SIZE,
            lr=Config.VAE_LR,
            save_path=None,
            device=Config.DEVICE):
    """
    Trains ConvVAE on frames collected from random play.
    """
    if save_path is None:
        save_path = os.path.join(Config.CHECKPOINT_DIR, "vae.pt")

    print("\n" + "=" * 50)
    print(" [Step 1/3] Training Vision Model (V - ConvVAE)")
    print("=" * 50)

    frames = dataset["frames"]
    flat_frames = torch.tensor(frames.reshape(-1, Config.ENV_SIZE, Config.ENV_SIZE, 3), dtype=torch.float32)
    track_t = torch.tensor(dataset["track"], dtype=torch.float32, device=device)

    dataset_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(flat_frames),
        batch_size=batch_size,
        shuffle=True
    )

    vae = ConvVAE().to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

    t0 = time.time()
    for epoch in range(epochs):
        total_loss = 0.0
        for (x,) in dataset_loader:
            x = x.to(device)
            # Encode and reparameterize with extra fuzz for M's near-miss codes
            mu, logvar = vae.encode(x)
            z = vae.reparameterize(mu, logvar, add_noise_std=Config.VAE_LATENT_NOISE)
            xhat = vae.decode(z)

            loss, recon, kl = ConvVAE.vae_loss(x, xhat, mu, logvar, track_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(x)

        avg_loss = total_loss / len(flat_frames)
        if (epoch + 1) % 3 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:2d}/{epochs:2d} | VAE Loss: {avg_loss:8.2f}")

    print(f"V trained in {time.time()-t0:.1f}s. Saving to {save_path}...")
    Config.ensure_dirs()
    vae.save(save_path)
    return vae


def train_m(vae,
            dataset,
            epochs=Config.MDN_EPOCHS,
            batch_size=Config.MDN_BATCH_SIZE,
            lr=Config.MDN_LR,
            save_path=None,
            device=Config.DEVICE):
    """
    Trains MDN-RNN on sequences of sampled latent codes and actions.
    """
    if save_path is None:
        save_path = os.path.join(Config.CHECKPOINT_DIR, "mdn.pt")

    print("\n" + "=" * 50)
    print(" [Step 2/3] Training Memory Model (M - MDN-RNN + Reward Head)")
    print("=" * 50)

    frames = dataset["frames"]
    actions = dataset["actions"]
    rewards = dataset["rewards"]
    episodes, horizon = frames.shape[:2]

    # Pre-encode all frames into sampled latent codes
    print("Pre-encoding frames with VAE into latent space...")
    vae.eval()
    flat_frames = torch.tensor(frames.reshape(-1, Config.ENV_SIZE, Config.ENV_SIZE, 3), dtype=torch.float32)
    zs_parts = []
    with torch.no_grad():
        for i in range(0, len(flat_frames), 4096):
            batch = flat_frames[i:i+4096].to(device)
            mu, logvar = vae.encode(batch)
            z_sample = vae.reparameterize(mu, logvar)
            zs_parts.append(z_sample.cpu())

    zs = torch.cat(zs_parts, dim=0).view(episodes, horizon, Config.LATENT_DIM)
    a1h = F.one_hot(torch.tensor(actions), Config.ENV_ACTIONS).float()
    rew = torch.tensor(rewards, dtype=torch.float32)

    mdn = MDNRNN().to(device)
    optimizer = torch.optim.Adam(mdn.parameters(), lr=lr)

    t0 = time.time()
    for epoch in range(epochs):
        perm = torch.randperm(episodes)
        total_loss = 0.0
        for i in range(0, episodes, batch_size):
            idx = perm[i:i+batch_size]
            z_batch = zs[idx].to(device)
            a_batch = a1h[idx].to(device)
            r_batch = rew[idx].to(device)

            # Predict next latent code and reward
            pi, mu, logsig, rhat, _ = mdn(z_batch[:, :-1], a_batch[:, :-1])
            loss_mdn = mdn_loss(pi, mu, logsig, z_batch[:, 1:])
            loss_rew = Config.MDN_REWARD_WEIGHT * F.mse_loss(rhat, r_batch[:, :-1])
            loss = loss_mdn + loss_rew

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(idx)

        avg_loss = total_loss / episodes
        if (epoch + 1) % 3 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:2d}/{epochs:2d} | MDN Loss: {avg_loss:8.3f}")

    print(f"M trained in {time.time()-t0:.1f}s. Saving to {save_path}...")
    Config.ensure_dirs()
    mdn.save(save_path)
    return mdn, zs, a1h


def train_c(vae,
            mdn,
            zs,
            a1h,
            generations=Config.ES_GENERATIONS,
            num_pairs=Config.ES_NUM_PAIRS,
            sigma=Config.ES_SIGMA,
            lr=Config.ES_LR,
            save_path=None,
            render_during_learning=False,
            render_interval=5,
            device=Config.DEVICE):
    """
    Trains Controller (C) via Evolution Strategies entirely inside M's dream.
    If render_during_learning is True, visually plays an evaluation race on screen
    at specified intervals so the user can watch the car learn in real time!
    """
    if save_path is None:
        save_path = os.path.join(Config.CHECKPOINT_DIR, "controller.npy")

    print("\n" + "=" * 50)
    print(" [Step 3/3] Evolving Controller (C) entirely inside M's Dream")
    print("=" * 50)

    controller = Controller()
    optimizer = EvolutionOptimizer(controller, num_pairs=num_pairs, sigma=sigma, lr=lr)

    rng = np.random.default_rng(Config.SEED)
    episodes = len(zs)

    hist_dream = []
    hist_real = []

    # Initial baseline (untrained, theta = 0)
    base_real, _ = evaluate_real(controller, vae, num_episodes=6, device=device)
    seeds0 = [(int(rng.integers(0, episodes)), int(rng.integers(0, 10**6))) for _ in range(Config.ES_SEEDS_PER_GEN)]
    base_dream = evaluate_dream(controller, mdn, zs, a1h, seeds0, device=device)

    hist_dream.append(base_dream)
    hist_real.append(base_real)
    print(f"Gen  0 (Baseline) | Dream Score: {base_dream:6.2f} | REAL Track Return: {base_real:6.2f}")

    # Optional initial visual demonstration
    if render_during_learning:
        print("[Visualizer] Showing baseline (Gen 0) car driving on track...")
        _render_eval_run(controller, vae, gen=0)

    t0 = time.time()
    for g in range(1, generations + 1):
        # Sample common dream seeds for all candidates in this generation (fair race)
        seeds = [(int(rng.integers(0, episodes)), int(rng.integers(0, 10**6))) for _ in range(Config.ES_SEEDS_PER_GEN)]

        def eval_candidate(candidate_theta):
            cand_c = Controller(params=candidate_theta)
            return evaluate_dream(cand_c, mdn, zs, a1h, seeds, device=device)

        theta_new, _ = optimizer.step(eval_candidate)

        # Measure performance in imagination (dream) and on real track
        dream_score = evaluate_dream(controller, mdn, zs, a1h, seeds, device=device)
        real_score, _ = evaluate_real(controller, vae, num_episodes=6, device=device)

        hist_dream.append(dream_score)
        hist_real.append(real_score)

        print(f"Gen {g:2d}/{generations:2d}        | Dream Score: {dream_score:6.2f} | REAL Track Return: {real_score:6.2f}")

        # Live visual feedback during learning!
        if render_during_learning and (g % render_interval == 0 or g == generations):
            print(f"[Visualizer] Showing Gen {g} car driving on track...")
            _render_eval_run(controller, vae, gen=g)

    print(f"\nC evolved in {time.time()-t0:.1f}s. Saving to {save_path}...")
    Config.ensure_dirs()
    controller.save(save_path)

    # Save training verdict plot
    _save_verdict_plot(hist_dream, hist_real, generations)

    return controller, hist_dream, hist_real


def _render_eval_run(controller, vae, gen=0, max_steps=100):
    """Helper to render an evaluation lap using Pygame."""
    try:
        from car_racer.env import MiniRacer
        env = MiniRacer(seed=1000 + gen)
        viewer = PygameCarRacerViewer(scale=14, fps=30, show_vae=True)
        obs = env.reset()
        cum_reward = 0.0

        for s in range(1, max_steps + 1):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=Config.DEVICE).unsqueeze(0)
            with torch.no_grad():
                z, _ = vae.encode(obs_t)
                action = controller.act(z)
                recon = vae.decode(z).squeeze(0).cpu().numpy()

            obs, r, done, info = env.step(action)
            cum_reward += r
            info["step"] = s
            info["score"] = cum_reward
            info["reward"] = r
            info["action"] = action
            info["mode"] = f"Learning Progress (Gen {gen})"

            viewer.draw_frame(real_obs=obs, recon_obs=recon, info=info)
            if done:
                break
        viewer.close()
    except Exception as e:
        print(f"[Notice] Pygame rendering skipped ({e})")


def _save_verdict_plot(hist_dream, hist_real, generations):
    """Plots and saves the transfer comparison plot."""
    Config.ensure_dirs()
    plt.figure(figsize=(8, 4.5))
    plt.plot(range(0, generations + 1), hist_dream, label="Practice in M's Dream (Imagined)", color="#DD9F3E", lw=2.5, marker="o")
    plt.plot(range(0, generations + 1), hist_real, label="Transfer to REAL Track", color="#2E8F8F", lw=2.5, marker="o")
    plt.title("Mini World Models: Skill Transfer from Imagination to Reality")
    plt.xlabel("Generation")
    plt.ylabel("Return")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    out_path = os.path.join(Config.OUTPUT_DIR, "learning_transfer_verdict.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved learning verdict plot to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Mini World Models (V, M, C)")
    parser.add_argument("--train-all", action="store_true", help="Train V, then M, then C end-to-end")
    parser.add_argument("--train-v", action="store_true", help="Train Vision VAE (V) only")
    parser.add_argument("--train-m", action="store_true", help="Train Memory MDN-RNN (M) only")
    parser.add_argument("--train-c", action="store_true", help="Evolve Controller (C) only")
    parser.add_argument("--render-learning", action="store_true",
                        help="Show live car racing on screen during evolution to watch learning in real time!")
    parser.add_argument("--render-interval", type=int, default=5,
                        help="Render evaluation race every N generations during training")
    parser.add_argument("--epochs-v", type=int, default=Config.VAE_EPOCHS, help="VAE training epochs")
    parser.add_argument("--epochs-m", type=int, default=Config.MDN_EPOCHS, help="MDN training epochs")
    parser.add_argument("--gens-c", type=int, default=Config.ES_GENERATIONS, help="Controller evolution generations")
    parser.add_argument("--force-recollect", action="store_true", help="Force recollecting random exploration dataset")
    return parser.parse_args()


def main():
    args = parse_args()
    Config.ensure_dirs()

    # Default to train all if none explicitly chosen
    if not (args.train_all or args.train_v or args.train_m or args.train_c):
        args.train_all = True

    dataset = get_or_create_dataset(force_recollect=args.force_recollect)

    vae_path = os.path.join(Config.CHECKPOINT_DIR, "vae.pt")
    mdn_path = os.path.join(Config.CHECKPOINT_DIR, "mdn.pt")

    vae = None
    if args.train_all or args.train_v:
        vae = train_v(dataset, epochs=args.epochs_v)
    elif os.path.exists(vae_path):
        vae = ConvVAE().to(Config.DEVICE)
        vae.load(vae_path, device=Config.DEVICE)
        print(f"Loaded existing VAE from {vae_path}")

    mdn, zs, a1h = None, None, None
    if args.train_all or args.train_m:
        if vae is None:
            raise RuntimeError("VAE must be trained or loaded before training MDN-RNN.")
        mdn, zs, a1h = train_m(vae, dataset, epochs=args.epochs_m)
    elif os.path.exists(mdn_path):
        mdn = MDNRNN().to(Config.DEVICE)
        mdn.load(mdn_path, device=Config.DEVICE)
        print(f"Loaded existing MDN from {mdn_path}")

    if args.train_all or args.train_c:
        if vae is None or mdn is None:
            raise RuntimeError("Both VAE and MDN must be trained/loaded to evolve Controller.")

        # If zs was not computed in this session, compute it now
        if zs is None:
            print("Encoding dataset for dream rollout...")
            vae.eval()
            flat_frames = torch.tensor(dataset["frames"].reshape(-1, Config.ENV_SIZE, Config.ENV_SIZE, 3), dtype=torch.float32)
            zs_parts = []
            with torch.no_grad():
                for i in range(0, len(flat_frames), 4096):
                    batch = flat_frames[i:i+4096].to(Config.DEVICE)
                    mu, logvar = vae.encode(batch)
                    z_sample = vae.reparameterize(mu, logvar)
                    zs_parts.append(z_sample.cpu())
            episodes, horizon = dataset["frames"].shape[:2]
            zs = torch.cat(zs_parts, dim=0).view(episodes, horizon, Config.LATENT_DIM)
            a1h = F.one_hot(torch.tensor(dataset["actions"]), Config.ENV_ACTIONS).float()

        train_c(
            vae, mdn, zs, a1h,
            generations=args.gens_c,
            render_during_learning=args.render_learning,
            render_interval=args.render_interval
        )

    print("\nTraining workflow completed successfully!")


if __name__ == "__main__":
    main()
