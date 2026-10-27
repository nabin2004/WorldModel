import os
import numpy as np
import pytest
from car_racer.dataset import collect_experience, get_or_create_dataset
from car_racer.config import Config

def test_collect_experience_specific_track():
    data = collect_experience(track="oval", episodes=2, horizon=10, seed=123, verbose=False)
    assert data["frames"].shape == (2, 10, 24, 24, 3)
    assert data["actions"].shape == (2, 10)
    assert data["rewards"].shape == (2, 10)
    assert data["track"].shape == (24, 24, 3)

def test_collect_experience_all_tracks():
    data = collect_experience(track="all", episodes=4, horizon=10, seed=123, verbose=False)
    assert data["frames"].shape == (4, 10, 24, 24, 3)
    assert data["actions"].shape == (4, 10)
    assert data["rewards"].shape == (4, 10)

def test_dataset_cache_naming(tmp_path):
    cache_path = str(tmp_path / "custom_track.npz")
    data1 = get_or_create_dataset(track="s_curve", cache_path=cache_path, episodes=2, horizon=5, force_recollect=True)
    assert os.path.exists(cache_path)
    data2 = get_or_create_dataset(track="s_curve", cache_path=cache_path, force_recollect=False)
    assert np.array_equal(data1["actions"], data2["actions"])
