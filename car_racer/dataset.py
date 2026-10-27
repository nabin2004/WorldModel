"""
Dataset collection and loading for Mini World Models.
Experience is gathered using a random exploration policy.
"""
import os
import numpy as np
from car_racer.env import MiniRacer
from car_racer.config import Config

def collect_experience(episodes=Config.DATASET_EPISODES,
                       horizon=Config.DATASET_HORIZON,
                       seed=Config.SEED,
                       verbose=True):
    """
    Collects experience using a uniform random exploration policy.
    Returns:
        dict with:
            - 'frames': (episodes, horizon, 24, 24, 3) float32 in [0, 1]
            - 'actions': (episodes, horizon) int64 in {0, 1, 2}
            - 'rewards': (episodes, horizon) float32
    """
    env = MiniRacer(seed=seed)
    rng = np.random.default_rng(seed)

    frames = np.zeros((episodes, horizon, env.SIZE, env.SIZE, 3), dtype=np.float32)
    actions = np.zeros((episodes, horizon), dtype=np.int64)
    rewards = np.zeros((episodes, horizon), dtype=np.float32)

    if verbose:
        print(f"Collecting dataset: {episodes} episodes × {horizon} steps = {episodes * horizon:,} frames...")

    for e in range(episodes):
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
        "track": env.track
    }

def get_or_create_dataset(cache_path=None, force_recollect=False):
    """
    Loads dataset from disk if cached, otherwise collects and saves it.
    """
    if cache_path is None:
        Config.ensure_dirs()
        cache_path = os.path.join(Config.DATA_DIR, "random_dataset.npz")

    if os.path.exists(cache_path) and not force_recollect:
        print(f"Loading cached dataset from {cache_path}...")
        data = np.load(cache_path)
        return {
            "frames": data["frames"],
            "actions": data["actions"],
            "rewards": data["rewards"],
            "track": data["track"]
        }

    dataset = collect_experience()
    Config.ensure_dirs()
    print(f"Saving dataset to cache: {cache_path}...")
    np.savez_compressed(
        cache_path,
        frames=dataset["frames"],
        actions=dataset["actions"],
        rewards=dataset["rewards"],
        track=dataset["track"]
    )
    return dataset
