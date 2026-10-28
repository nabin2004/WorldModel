"""
Interactive and Visual Racing Script for Mini World Models.
Supports:
1. Real-time Pygame GUI with rich HUD, telemetry, and manual/autonomous driving modes.
2. Side-by-side comparison: Real Track vs VAE Reconstruction vs M's Imagined Dream.
3. Visualization of learning progress (evaluating controller across generations).
4. Headless rendering / GIF animation export.
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

from car_racer.config import Config
from car_racer.env import MiniRacer
from car_racer.models.vae import ConvVAE
from car_racer.models.mdn_rnn import MDNRNN, sample_mdn
from car_racer.models.controller import Controller

# Action mappings
ACTION_NAMES = {0: "LEFT ◄", 1: "STRAIGHT ▲", 2: "RIGHT ►"}
ACTION_COLORS = {0: (220, 80, 50), 1: (50, 180, 80), 2: (50, 120, 220)}

def create_hud_overlay(img_rgb, step, reward, cumulative_score, action, d, phi, direction, mode="Autonomous", track_name="circle"):
    """
    Renders high-contrast telemetry text onto an RGB image.
    img_rgb: (H, W, 3) in uint8 [0, 255]
    """
    # If PIL is available, draw text cleanly
    try:
        from PIL import Image, ImageDraw, ImageFont
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)

        # Telemetry panel background
        panel_h = 75
        draw.rectangle([0, 0, img_rgb.shape[1], panel_h], fill=(20, 24, 28))

        # Text information
        dir_str = "CW" if direction > 0 else "CCW"
        act_str = ACTION_NAMES.get(action, "UNKNOWN")

        line1 = f"[{mode}] Track: {track_name.upper()} | Step: {step}/150 | Score: {cumulative_score:.1f}"
        line2 = f"Action: {act_str} | Step Reward: {reward:+.2f}"
        line3 = f"Dev (d): {d:+.2f} | Steering: {phi:+.2f} | Dir: {dir_str}"

        draw.text((10, 6), line1, fill=(240, 240, 240))
        draw.text((10, 28), line2, fill=(255, 215, 0))
        draw.text((10, 50), line3, fill=(180, 220, 240))

        return np.array(pil_img)
    except Exception:
        return img_rgb


class PygameCarRacerViewer:
    """
    Interactive Pygame viewer for MiniRacer.
    Shows real-time car racing at configurable FPS with HUD and side-by-side VAE/Dream views.
    """
    def __init__(self, scale=16, fps=25, show_vae=True):
        import pygame
        pygame.init()
        pygame.font.init()

        self.scale = scale
        self.fps = fps
        self.show_vae = show_vae

        self.track_size = 24 * scale  # e.g. 384x384
        self.hud_height = 90

        if show_vae:
            self.width = self.track_size * 2 + 30
            self.height = self.track_size + self.hud_height + 20
        else:
            self.width = self.track_size
            self.height = self.track_size + self.hud_height

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Mini World Models — Live Car Racing")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 15)
        self.font_large = pygame.font.SysFont("Consolas", 18, bold=True)

    def draw_frame(self, real_obs, recon_obs=None, info=None, dream_obs=None):
        import pygame

        pressed_keys = []
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            elif event.type == pygame.KEYDOWN:
                pressed_keys.append(event.key)

        self.screen.fill((25, 27, 31))

        # 1. Real Track View
        real_upscaled = (np.repeat(np.repeat(real_obs, self.scale, axis=0), self.scale, axis=1) * 255).astype(np.uint8)
        real_surf = pygame.surfarray.make_surface(np.transpose(real_upscaled, (1, 0, 2)))
        self.screen.blit(real_surf, (10, self.hud_height + 10))

        # Title over real track
        title1 = self.font.render("Real Environment Track", True, (200, 230, 200))
        self.screen.blit(title1, (10, self.hud_height - 18))

        # 2. VAE Reconstruction or Dream View (if enabled)
        if self.show_vae and recon_obs is not None:
            recon_upscaled = (np.repeat(np.repeat(recon_obs, self.scale, axis=0), self.scale, axis=1) * 255).clip(0, 255).astype(np.uint8)
            recon_surf = pygame.surfarray.make_surface(np.transpose(recon_upscaled, (1, 0, 2)))
            x_offset = self.track_size + 20
            self.screen.blit(recon_surf, (x_offset, self.hud_height + 10))

            title2_text = "VAE Reconstruction (z=16)" if dream_obs is None else "M's Imagination (Dream)"
            title2 = self.font.render(title2_text, True, (240, 200, 160))
            self.screen.blit(title2, (x_offset, self.hud_height - 18))

        # 3. HUD Display
        if info:
            step = info.get("step", 0)
            score = info.get("score", 0.0)
            reward = info.get("reward", 0.0)
            action = info.get("action", 1)
            d = info.get("d", 0.0)
            phi = info.get("phi", 0.0)
            direction = info.get("direction", 1)
            mode = info.get("mode", "Autonomous")
            track_name = str(info.get("track", "circle")).upper()

            dir_text = "CW" if direction > 0 else "CCW"
            action_text = ACTION_NAMES.get(action, "UNKNOWN")

            # HUD Background card
            pygame.draw.rect(self.screen, (35, 38, 44), (10, 8, self.width - 20, self.hud_height - 15), border_radius=6)

            t_mode = self.font_large.render(f"[{mode}]  Track: {track_name}  Step: {step}/150  |  Score: {score:.1f}", True, (255, 255, 255))
            t_act = self.font.render(f"Steering: {action_text}  (Reward: {reward:+.2f})  |  [T] Cycle Track", True, (255, 215, 0))
            t_tel = self.font.render(f"Centerline (d): {d:+.2f}  |  Angle: {phi:+.2f} rad  |  Dir: {dir_text}", True, (170, 215, 255))

            self.screen.blit(t_mode, (22, 14))
            self.screen.blit(t_act, (22, 38))
            self.screen.blit(t_tel, (22, 58))

        pygame.display.flip()
        self.clock.tick(self.fps)
        return pressed_keys

    def close(self):
        import pygame
        pygame.quit()


def run_inference_episode(controller=None, vae=None, seed=None, fps=25, show_vae=True, max_steps=150, track="circle"):
    """
    Runs a single full episode with real-time Pygame visualization.
    If controller is None, uses an untrained / random policy.
    Supports cycling tracks live at runtime with the 'T' key.
    """
    import pygame
    from car_racer.tracks.registry import list_tracks

    current_track = track
    env = MiniRacer(track=current_track, seed=seed)
    device = Config.DEVICE

    viewer = PygameCarRacerViewer(scale=16, fps=fps, show_vae=(show_vae and vae is not None))

    obs = env.reset()
    cumulative_score = 0.0
    action = 1

    print(f"\n--- Starting Race Episode (Track: {current_track.upper()}, Seed: {seed if seed is not None else 'random'}) ---")
    print("Controls: Press 'T' to cycle tracks. Close window to stop.")

    for step in range(1, max_steps + 1):
        recon_obs = None
        if controller is not None and vae is not None:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                z, _ = vae.encode(obs_t)
                action = controller.act(z)
                recon = vae.decode(z).squeeze(0).cpu().numpy()
                recon_obs = recon
        elif controller is not None:
            action = controller.act(np.zeros(16))
        else:
            action = int(env.rng.integers(0, 3))

        obs, reward, done, info = env.step(action)
        cumulative_score += reward

        info["score"] = cumulative_score
        info["reward"] = reward
        info["action"] = action
        info["mode"] = "Dream-Trained C" if controller is not None else "Untrained/Random"

        pressed_keys = viewer.draw_frame(real_obs=obs, recon_obs=recon_obs, info=info)
        if pygame.K_t in pressed_keys:
            tracks = list_tracks()
            idx = tracks.index(current_track) if current_track in tracks else 0
            current_track = tracks[(idx + 1) % len(tracks)]
            env = MiniRacer(track=current_track, seed=seed)
            obs = env.reset()
            cumulative_score = 0.0
            print(f"[Visualizer] Switched track to: {current_track.upper()} (T key)")

        if done:
            print(f"Episode finished at step {step}! Final Score: {cumulative_score:.2f}")
            time.sleep(1.0)
            break

    viewer.close()
    return cumulative_score


def run_dream_race(mdn, vae, controller, start_seed=42, dream_steps=60, fps=20):
    """
    Visualizes the car driving purely inside M's imagination (closed-loop Dream).
    No game engine is run after the initial 4 warm-up steps!
    """
    import pygame
    device = Config.DEVICE
    env = MiniRacer(seed=start_seed)

    viewer = PygameCarRacerViewer(scale=16, fps=fps, show_vae=True)

    print("\n--- Visualizing M's Dream (Rollout in Imagination) ---")
    # Collect 4 real steps to warm up GRU memory
    warm_obs = []
    warm_actions = []
    obs = env.reset()
    for _ in range(4):
        a = 1  # straight
        warm_obs.append(obs)
        warm_actions.append(a)
        obs, _, _, _ = env.step(a)

    with torch.no_grad():
        warm_obs_t = torch.tensor(np.array(warm_obs), dtype=torch.float32, device=device)
        warm_z, _ = vae.encode(warm_obs_t)
        warm_a1h = F.one_hot(torch.tensor(warm_actions, device=device), 3).float().unsqueeze(0)

        # Warm up M's hidden state
        _, _, _, _, h = mdn(warm_z.unsqueeze(0), warm_a1h, None)

        z = warm_z[-1:].unsqueeze(0)  # (1, 1, 16)
        total_dream_reward = 0.0

        for t in range(dream_steps):
            # C picks action inside dream
            a = controller.act(z[0, 0])
            ah = F.one_hot(torch.tensor([[a]], device=device), 3).float()

            # M predicts next code + reward
            pi, mu, logsig, rhat, h = mdn(z, ah, h)
            dream_r = float(rhat[0, 0].clamp(0, 1).item())
            total_dream_reward += dream_r

            # Sample next latent code
            z = sample_mdn(pi[0, 0:1], mu[0, 0:1], logsig[0, 0:1], temperature=0.7).unsqueeze(1)

            # Decode dreamed latent code back to RGB frame
            dream_frame = vae.decode(z.squeeze(1))[0].cpu().numpy()

            info = {
                "step": t + 1,
                "score": total_dream_reward,
                "reward": dream_r,
                "action": a,
                "d": 0.0,
                "phi": 0.0,
                "direction": 1,
                "mode": "Inside M's Dream"
            }

            viewer.draw_frame(real_obs=dream_frame, recon_obs=dream_frame, info=info, dream_obs=dream_frame)

    print(f"Dream rollout complete! Hallucinated return: {total_dream_reward:.2f}")
    time.sleep(1.0)
    viewer.close()


def save_racing_gif(controller, vae, output_path, num_steps=120, seed=42, track="circle"):
    """
    Renders an episode and exports it to an animated GIF (useful for headless / reporting).
    """
    from PIL import Image
    env = MiniRacer(track=track, seed=seed)
    device = Config.DEVICE
    obs = env.reset()

    frames_pil = []
    tot = 0.0
    for step in range(1, num_steps + 1):
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            z, _ = vae.encode(obs_t)
            a = controller.act(z)
            recon = vae.decode(z).squeeze(0).cpu().numpy()

        obs, r, done, info = env.step(a)
        tot += r

        # Combine real and recon side-by-side
        real_up = (np.repeat(np.repeat(obs, 12, axis=0), 12, axis=1) * 255).astype(np.uint8)
        recon_up = (np.repeat(np.repeat(recon, 12, axis=0), 12, axis=1) * 255).clip(0, 255).astype(np.uint8)
        combined = np.concatenate([real_up, recon_up], axis=1)

        info_hud = create_hud_overlay(
            combined, step=step, reward=r, cumulative_score=tot,
            action=a, d=info["d"], phi=info["phi"], direction=info["direction"],
            track_name=info.get("track", track)
        )
        frames_pil.append(Image.fromarray(info_hud))
        if done:
            break

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    frames_pil[0].save(
        output_path,
        save_all=True,
        append_images=frames_pil[1:],
        duration=40,  # 25 fps
        loop=0
    )
    print(f"Saved racing GIF to {output_path} ({len(frames_pil)} frames, total score: {tot:.1f})")


def save_learning_progression_gif(controller, vae, output_path, num_steps=90, seed=1001, track="circle"):
    """
    Renders a split-screen animation comparing:
    Left: Gen 0 baseline (untrained, theta = 0)
    Right: Gen 20 converged dream-trained controller
    """
    from PIL import Image, ImageDraw
    env_un = MiniRacer(track=track, seed=seed)
    env_tr = MiniRacer(track=track, seed=seed)
    device = Config.DEVICE

    c_untrained = Controller(params=np.zeros(51, dtype=np.float32))

    obs_un = env_un.reset()
    obs_tr = env_tr.reset()

    frames_pil = []
    tot_un = 0.0
    tot_tr = 0.0
    done_un = False
    done_tr = False
    last_un_frame = None

    for step in range(1, num_steps + 1):
        # Untrained agent step
        if not done_un:
            with torch.no_grad():
                obs_t_un = torch.tensor(obs_un, dtype=torch.float32, device=device).unsqueeze(0)
                z_un, _ = vae.encode(obs_t_un)
                a_un = c_untrained.act(z_un)
            obs_un, r_un, done_un, info_un = env_un.step(a_un)
            tot_un += r_un
            un_up = (np.repeat(np.repeat(obs_un, 12, axis=0), 12, axis=1) * 255).astype(np.uint8)
            last_un_frame = un_up
        else:
            un_up = last_un_frame

        # Dream-trained agent step
        if not done_tr:
            with torch.no_grad():
                obs_t_tr = torch.tensor(obs_tr, dtype=torch.float32, device=device).unsqueeze(0)
                z_tr, _ = vae.encode(obs_t_tr)
                a_tr = controller.act(z_tr)
            obs_tr, r_tr, done_tr, info_tr = env_tr.step(a_tr)
            tot_tr += r_tr
            tr_up = (np.repeat(np.repeat(obs_tr, 12, axis=0), 12, axis=1) * 255).astype(np.uint8)
        else:
            tr_up = (np.repeat(np.repeat(obs_tr, 12, axis=0), 12, axis=1) * 255).astype(np.uint8)

        combined = np.concatenate([un_up, tr_up], axis=1)
        pil_img = Image.fromarray(combined)
        draw = ImageDraw.Draw(pil_img)

        # Header HUD
        panel_h = 75
        draw.rectangle([0, 0, combined.shape[1], panel_h], fill=(20, 24, 28))

        draw.text((10, 6), f"Mini World Models — Learning Progression ({track.upper()}) | Step {step}/{num_steps}", fill=(240, 240, 240))
        status_un = f"Score: {tot_un:.1f}" if not done_un else f"CRASHED / OFF TRACK ({tot_un:.1f})"
        draw.text((10, 30), "[Gen 0: Untrained Baseline]", fill=(240, 100, 100))
        draw.text((10, 50), status_un, fill=(200, 200, 200))

        status_tr = f"Score: {tot_tr:.1f} | Dev: {info_tr['d']:+.2f}"
        draw.text((combined.shape[1] // 2 + 10, 30), "[Gen 20: Dream-Trained Policy]", fill=(100, 230, 140))
        draw.text((combined.shape[1] // 2 + 10, 50), status_tr, fill=(255, 215, 0))

        frames_pil.append(pil_img)
        if done_un and done_tr:
            break

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    frames_pil[0].save(
        output_path,
        save_all=True,
        append_images=frames_pil[1:],
        duration=40,
        loop=0
    )
    print(f"Saved learning progression GIF to {output_path} ({len(frames_pil)} frames)")


def save_dream_gif(mdn, vae, controller, output_path, start_seed=42, dream_steps=75):
    """
    Renders an animation of the car navigating inside M's hallucinated recurrent imagination.
    """
    from PIL import Image, ImageDraw
    device = Config.DEVICE
    env = MiniRacer(seed=start_seed)

    # Collect 4 real steps to warm up GRU memory
    warm_obs = []
    warm_actions = []
    obs = env.reset()
    for _ in range(4):
        a = 1
        warm_obs.append(obs)
        warm_actions.append(a)
        obs, _, _, _ = env.step(a)

    frames_pil = []
    with torch.no_grad():
        warm_obs_t = torch.tensor(np.array(warm_obs), dtype=torch.float32, device=device)
        warm_z, _ = vae.encode(warm_obs_t)
        warm_a1h = F.one_hot(torch.tensor(warm_actions, device=device), 3).float().unsqueeze(0)
        _, _, _, _, h = mdn(warm_z.unsqueeze(0), warm_a1h, None)

        z = warm_z[-1:].unsqueeze(0)
        total_dream_r = 0.0

        for t in range(dream_steps):
            a = controller.act(z[0, 0])
            ah = F.one_hot(torch.tensor([[a]], device=device), 3).float()
            pi, mu, logsig, rhat, h = mdn(z, ah, h)
            dream_r = float(rhat[0, 0].clamp(0, 1).item())
            total_dream_r += dream_r

            z = sample_mdn(pi[0, 0:1], mu[0, 0:1], logsig[0, 0:1], temperature=0.7).unsqueeze(1)
            dream_frame = vae.decode(z.squeeze(1))[0].cpu().numpy()

            dream_up = (np.repeat(np.repeat(dream_frame, 16, axis=0), 16, axis=1) * 255).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(dream_up)
            draw = ImageDraw.Draw(pil_img)

            panel_h = 70
            draw.rectangle([0, 0, dream_up.shape[1], panel_h], fill=(20, 24, 28))
            draw.text((10, 6), "M's Hallucinated Dream (Recurrent Memory Rollout)", fill=(240, 200, 160))
            draw.text((10, 28), f"Step: {t+1}/{dream_steps} | Return: {total_dream_r:.1f} | Pred Reward: {dream_r:+.2f}", fill=(255, 215, 0))
            act_str = ACTION_NAMES.get(a, "UNKNOWN")
            draw.text((10, 48), f"Controller Action: {act_str} | Zero physics engine calls", fill=(170, 215, 255))

            frames_pil.append(pil_img)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    frames_pil[0].save(
        output_path,
        save_all=True,
        append_images=frames_pil[1:],
        duration=50,
        loop=0
    )
    print(f"Saved dream GIF to {output_path} ({len(frames_pil)} frames, dream score: {total_dream_r:.1f})")


def parse_args():
    parser = argparse.ArgumentParser(description="Live Car Racing Viewer for Mini World Models")
    parser.add_argument("--mode", type=str, choices=["infer", "dream", "gif", "random"], default="infer",
                        help="Visualization mode: 'infer' (live race), 'dream' (M imagination), 'gif' (export GIF), 'random' (untrained)")
    parser.add_argument("--checkpoint-dir", type=str, default=Config.CHECKPOINT_DIR,
                        help="Path to checkpoints directory")
    parser.add_argument("--seed", type=int, default=1001, help="Random seed for race episode")
    parser.add_argument("--fps", type=int, default=25, help="Playback FPS")
    parser.add_argument("--gif-out", type=str, default="outputs/car_race_demo.gif", help="Output path for GIF mode")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load models
    vae = ConvVAE().to(Config.DEVICE)
    vae_path = os.path.join(args.checkpoint_dir, "vae.pt")
    if os.path.exists(vae_path):
        vae.load(vae_path, device=Config.DEVICE)
        print(f"Loaded VAE from {vae_path}")
    else:
        print(f"[Notice] No VAE checkpoint found at {vae_path}. Using uninitialized VAE.")

    controller = Controller()
    c_path = os.path.join(args.checkpoint_dir, "controller.npy")
    if os.path.exists(c_path):
        controller.load(c_path)
        print(f"Loaded Controller from {c_path}")
    else:
        print(f"[Notice] No Controller checkpoint found at {c_path}. Using untrained controller (theta=0).")

    if args.mode == "infer":
        run_inference_episode(controller=controller, vae=vae, seed=args.seed, fps=args.fps)
    elif args.mode == "random":
        run_inference_episode(controller=None, vae=vae, seed=args.seed, fps=args.fps)
    elif args.mode == "dream":
        mdn = MDNRNN().to(Config.DEVICE)
        mdn_path = os.path.join(args.checkpoint_dir, "mdn.pt")
        if os.path.exists(mdn_path):
            mdn.load(mdn_path, device=Config.DEVICE)
            print(f"Loaded MDN-RNN from {mdn_path}")
        else:
            print(f"[Error] MDN checkpoint not found at {mdn_path}. Please train M first.")
            return
        run_dream_race(mdn=mdn, vae=vae, controller=controller, start_seed=args.seed, fps=args.fps)
    elif args.mode == "gif":
        save_racing_gif(controller=controller, vae=vae, output_path=args.gif_out, seed=args.seed)


if __name__ == "__main__":
    main()
