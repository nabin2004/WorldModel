"""
C (Controller): 51-parameter linear policy trained inside M's dream.
Uses Evolution Strategies (mirrored sampling, rank-based update).
"""
import math
import numpy as np
import torch
import torch.nn.functional as F
from car_racer.config import Config
from car_racer.env import MiniRacer

class Controller:
    """
    Linear policy: z (16) -> 3 action logits.
    Params: 16 x 3 (weights) + 3 (biases) = 51 parameters.
    """
    def __init__(self, latent_dim=Config.LATENT_DIM, action_dim=Config.ENV_ACTIONS, params=None):
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.num_params = latent_dim * action_dim + action_dim  # 51

        if params is None:
            params = np.zeros(self.num_params, dtype=np.float32)
        self.set_params(params)

    def set_params(self, params):
        self.params = np.array(params, dtype=np.float32).copy()
        w_end = self.latent_dim * self.action_dim
        self.W = torch.tensor(self.params[:w_end], dtype=torch.float32).view(self.action_dim, self.latent_dim)
        self.b = torch.tensor(self.params[w_end:], dtype=torch.float32)

    def get_params(self):
        return self.params.copy()

    def act(self, z, device=None):
        """
        z: latent tensor (latent_dim,) or (1, latent_dim)
        returns: int action in {0, 1, 2}
        """
        if not isinstance(z, torch.Tensor):
            z = torch.tensor(z, dtype=torch.float32)
        if z.dim() > 1:
            z = z.squeeze()

        W = self.W.to(z.device)
        b = self.b.to(z.device)
        logits = W @ z + b
        return int(torch.argmax(logits).item())

    def save(self, path):
        np.save(path, self.params)

    def load(self, path):
        params = np.load(path)
        self.set_params(params)


def evaluate_dream(controller, mdn, dataset_zs, dataset_actions, seeds,
                   horizon=Config.ES_DREAM_HORIZON,
                   temperature=1.0,
                   device=Config.DEVICE):
    """
    Simulates driving C inside M's dream rollout.
    Score is accumulated solely from M's learned reward head!
    """
    params = controller.get_params()
    w_end = controller.latent_dim * controller.action_dim
    W = torch.tensor(params[:w_end], dtype=torch.float32, device=device).view(controller.action_dim, controller.latent_dim)
    b = torch.tensor(params[w_end:], dtype=torch.float32, device=device)

    total_reward = 0.0
    with torch.no_grad():
        for (e, ts) in seeds:
            g = torch.Generator(device="cpu").manual_seed(ts)
            # Warm up M's GRU on first 4 steps of real episode
            z_warm = dataset_zs[e:e+1, :4].to(device)
            a_warm = dataset_actions[e:e+1, :4].to(device)
            _, _, _, _, h = mdn(z_warm, a_warm, None)

            z = dataset_zs[e:e+1, 4:5].to(device)
            for _ in range(horizon):
                # Controller picks action from latent code z
                a = int(torch.argmax(W @ z[0, 0] + b))
                ah = F.one_hot(torch.tensor([[a]], device=device), controller.action_dim).float()

                pi, mu, logsig, rhat, h = mdn(z, ah, h)
                total_reward += float(rhat[0, 0].clamp(0, 1).item())

                # Sample next dreamed latent code
                probs = F.softmax(pi[0, 0] / temperature, -1)
                k = int(torch.multinomial(probs.cpu(), 1, generator=g).item())
                sig = logsig[0, 0, k].exp() * math.sqrt(temperature)
                noise = torch.randn(controller.latent_dim, generator=g).to(device)
                z = (mu[0, 0, k] + noise * sig).view(1, 1, controller.latent_dim)

    return total_reward / len(seeds)


def evaluate_real(controller, vae, num_episodes=10, horizon=Config.ES_EVAL_HORIZON,
                  seed_start=1000, track="circle", device=Config.DEVICE):
    """
    Evaluates controller on the real MiniRacer track.
    (Used only for logging/measuring performance, never for training).
    """
    params = controller.get_params()
    w_end = controller.latent_dim * controller.action_dim
    W = torch.tensor(params[:w_end], dtype=torch.float32, device=device).view(controller.action_dim, controller.latent_dim)
    b = torch.tensor(params[w_end:], dtype=torch.float32, device=device)

    # Use first available track if 'all' is passed
    eval_track = "circle" if track == "all" else track

    scores = []
    with torch.no_grad():
        for k in range(num_episodes):
            env = MiniRacer(track=eval_track, seed=seed_start + k)
            obs = env.reset()
            tot = 0.0
            for _ in range(horizon):
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                zt, _ = vae.encode(obs_t)
                a = int(torch.argmax(W @ zt[0] + b))
                obs, r, done, _ = env.step(a)
                tot += r
                if done:
                    break
            scores.append(tot)

    return float(np.mean(scores)), float(np.std(scores))


class EvolutionOptimizer:
    """
    Natural Evolution Strategies (NES) with mirrored perturbation pairs and rank-based updates.
    """
    def __init__(self,
                 controller,
                 num_pairs=Config.ES_NUM_PAIRS,
                 sigma=Config.ES_SIGMA,
                 lr=Config.ES_LR,
                 seed=Config.SEED):
        self.controller = controller
        self.num_pairs = num_pairs
        self.sigma = sigma
        self.lr = lr
        self.rng = np.random.default_rng(seed)

    def step(self, eval_fn):
        """
        Runs one generation of evolution.
        eval_fn(candidate_params) -> scalar score
        """
        theta = self.controller.get_params()
        half = self.rng.standard_normal((self.num_pairs, len(theta))) * self.sigma
        noise = np.concatenate([half, -half], axis=0)  # Mirrored: +epsilon and -epsilon

        scores = np.array([eval_fn(theta + n) for n in noise])

        # Rank-based update
        ranks = (scores.argsort().argsort() - (len(noise) - 1) / 2.0) / len(noise)
        theta_new = theta + self.lr * (ranks @ noise) / self.sigma
        self.controller.set_params(theta_new)

        return theta_new, scores
