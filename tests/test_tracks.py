import numpy as np
import pytest
from car_racer.env import MiniRacer

def test_circular_track_backward_compatibility():
    # Verify that default MiniRacer produces identical dimensions and valid observation
    env = MiniRacer(seed=42)
    obs = env.reset()
    assert obs.shape == (24, 24, 3)
    assert obs.dtype == np.float32
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0)

    # Step for 10 steps and verify reward and info
    for a in [0, 1, 2, 1, 0]:
        obs, r, done, info = env.step(a)
        assert obs.shape == (24, 24, 3)
        assert isinstance(r, float)
        assert -1.5 <= r <= 1.0
        assert "d" in info and "phi" in info and "step" in info
        assert abs(info["d"]) <= env.HALF_W + 0.1
