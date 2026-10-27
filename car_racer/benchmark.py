"""
Cross-track benchmark suite for evaluating zero-shot policy generalization.
"""
import os
from typing import Dict, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt

from car_racer.config import Config
from car_racer.env import MiniRacer
from car_racer.tracks.registry import list_tracks

def run_cross_track_benchmark(
    controller,
    vae,
    episodes_per_track: int = 5,
    horizon: int = Config.ES_EVAL_HORIZON,
    seed_start: int = 2000,
    plot_path: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates a controller policy across all registered track topologies.
    Measures mean return, survival steps, and mean centerline deviation |d|.
    """
    device = Config.DEVICE
    tracks = list_tracks()
    results = {}

    if plot_path is None:
        Config.ensure_dirs()
        plot_path = os.path.join(Config.OUTPUT_DIR, "zero_shot_generalization.png")

    if verbose:
        print("\n" + "=" * 80)
        print(" ZERO-SHOT GENERALIZATION BENCHMARK ACROSS ROAD TOPOLOGIES")
        print("=" * 80)
        print(f" Episodes per track: {episodes_per_track} | Evaluation Horizon: {horizon} steps")
        print("-" * 80)
        print(f" {'Track Name':<16} | {'Mean Return':<18} | {'Survival Steps':<18} | {'Centerline |d|':<16}")
        print("-" * 80)

    for trk in tracks:
        returns = []
        steps_list = []
        devs = []

        with torch.no_grad():
            for ep in range(episodes_per_track):
                env = MiniRacer(track=trk, seed=seed_start + ep)
                obs = env.reset()
                cum_ret = 0.0
                ep_steps = 0
                ep_devs = []

                for s in range(horizon):
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    if vae is not None:
                        z, _ = vae.encode(obs_t)
                        a = controller.act(z)
                    else:
                        a = controller.act(np.zeros(16))

                    obs, r, done, info = env.step(a)
                    cum_ret += r
                    ep_steps += 1
                    ep_devs.append(abs(info["d"]))
                    if done:
                        break

                returns.append(cum_ret)
                steps_list.append(ep_steps)
                devs.append(float(np.mean(ep_devs)))

        mean_ret, std_ret = float(np.mean(returns)), float(np.std(returns))
        mean_steps = float(np.mean(steps_list))
        mean_dev, std_dev = float(np.mean(devs)), float(np.std(devs))

        results[trk] = {
            "mean_return": mean_ret,
            "std_return": std_ret,
            "mean_steps": mean_steps,
            "mean_dev": mean_dev,
            "std_dev": std_dev
        }

        if verbose:
            trk_label = trk.upper()
            ret_str = f"{mean_ret:6.1f} +/- {std_ret:4.1f}"
            step_str = f"{mean_steps:5.1f} / {horizon}"
            dev_str = f"{mean_dev:4.2f} +/- {std_dev:4.2f}"
            print(f" {trk_label:<16} | {ret_str:<18} | {step_str:<18} | {dev_str:<16}")

    if verbose:
        print("=" * 80)

    # Save summary comparison chart
    if plot_path:
        os.makedirs(os.path.dirname(os.path.abspath(plot_path)), exist_ok=True)
        track_names = [t.upper() for t in tracks]
        mean_returns = [results[t]["mean_return"] for t in tracks]
        std_returns = [results[t]["std_return"] for t in tracks]
        mean_deviations = [results[t]["mean_dev"] for t in tracks]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

        # 1. Return comparison
        bars1 = ax1.bar(track_names, mean_returns, yerr=std_returns, capsize=4, color="#2E8F8F", alpha=0.85, edgecolor="#1B5E5E")
        ax1.set_title("Cross-Track Driving Return", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Cumulative Score", fontsize=10)
        ax1.grid(True, linestyle="--", alpha=0.4, axis="y")

        # 2. Centerline deviation comparison
        bars2 = ax2.bar(track_names, mean_deviations, color="#DD9F3E", alpha=0.85, edgecolor="#A86B1E")
        ax2.set_title("Centerline Displacement |d|", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Mean Deviation (lower is better)", fontsize=10)
        ax2.grid(True, linestyle="--", alpha=0.4, axis="y")

        plt.suptitle("Zero-Shot Generalization Across Road Topologies", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()

        if verbose:
            print(f"[Benchmark] Saved comparison plot to: {plot_path}")

    return results
