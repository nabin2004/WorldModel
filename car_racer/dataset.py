"""
Dataset collection and loading for Mini World Models across customizable tracks.
Experience is gathered using a uniform random exploration policy.
"""
import os
from typing import Optional, Union
import numpy as np
from car_racer.env import MiniRacer
from car_racer.config import Config
from car_racer.tracks.registry import list_tracks

def collect_experience(track: str = "circle",
                       episodes: int = Config.DATASET_EPISODES,
                       horizon: int = Config.DATASET_HORIZON,
                       seed: int = Config.SEED,
                       verbose: bool = True):
    """
    Collects experience using a uniform random exploration policy on the specified track.
    If track == 'all', episodes are distributed evenly across all registered tracks.
    Returns:
        dict with:
            - 'frames': (episodes, horizon, 24, 24, 3) float32 in [0, 1]
            - 'actions': (episodes, horizon) int64 in {0, 1, 2}
            - 'rewards': (episodes, horizon) float32
            - 'track': (24, 24, 3) float32 reference canvas
    """
    rng = np.random.default_rng(seed)
    all_tracks = list_tracks() if track.lower() == "all" else [track]

    frames = np.zeros((episodes, horizon, Config.ENV_SIZE, Config.ENV_SIZE, 3), dtype=np.float32)
    actions = np.zeros((episodes, horizon), dtype=np.int64)
    rewards = np.zeros((episodes, horizon), dtype=np.float32)

    if verbose:
        print(f"Collecting dataset ({track}): {episodes} episodes × {horizon} steps = {episodes * horizon:,} frames...")

    ref_track_canvas = None
    for e in range(episodes):
        trk_name = all_tracks[e % len(all_tracks)]
        env = MiniRacer(track=trk_name, seed=seed + e)
        if ref_track_canvas is None:
            ref_track_canvas = env.track

        obs = env.reset()
        for t in range(horizon):
            a = int(rng.integers(0, env.ACTIONS))
            frames[e, t] = obs
            actions[e, t] = a
            obs, r, done, _ = env.step(a)
            rewards[e, t] = r

    if verbose:
        mb = frames.nbytes / 1e6
        print(f"Dataset collected successfully ({mb:.1f} MB).")

    return {
        "frames": frames,
        "actions": actions,
        "rewards": rewards,
        "track": ref_track_canvas
    }

def get_or_create_dataset(track: str = "circle",
                          cache_path: Optional[str] = None,
                          episodes: int = Config.DATASET_EPISODES,
                          horizon: int = Config.DATASET_HORIZON,
                          seed: int = Config.SEED,
                          force_recollect: bool = False):
    """
    Loads dataset from disk if cached, otherwise collects and saves it.
    Per-track caching: random_dataset_<track>.npz (with fallback to random_dataset.npz for circle).
    """
    Config.ensure_dirs()
    if cache_path is None:
        track_slug = track.lower().replace("-", "_")
        cache_path = os.path.join(Config.DATA_DIR, f"random_dataset_{track_slug}.npz")
        # Check backward compatibility fallback for default circle
        if track_slug == "circle" and not os.path.exists(cache_path):
            legacy_path = os.path.join(Config.DATA_DIR, "random_dataset.npz")
            if os.path.exists(legacy_path):
                cache_path = legacy_path

    if os.path.exists(cache_path) and not force_recollect:
        print(f"Loading cached dataset from {cache_path}...")
        data = np.load(cache_path)
        return {
            "frames": data["frames"],
            "actions": data["actions"],
            "rewards": data["rewards"],
            "track": data["track"]
        }

    dataset = collect_experience(track=track, episodes=episodes, horizon=horizon, seed=seed)
    print(f"Saving dataset to cache: {cache_path}...")
    np.savez_compressed(
        cache_path,
        frames=dataset["frames"],
        actions=dataset["actions"],
        rewards=dataset["rewards"],
        track=dataset["track"]
    )
    return dataset
