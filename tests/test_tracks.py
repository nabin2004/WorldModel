import numpy as np
import pytest
from car_racer.env import MiniRacer
from car_racer.tracks.registry import get_track, list_tracks

def test_circular_track_backward_compatibility():
    env = MiniRacer(seed=42)
    obs = env.reset()
    assert obs.shape == (24, 24, 3)
    assert obs.dtype == np.float32
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0)

    for a in [0, 1, 2, 1, 0]:
        obs, r, done, info = env.step(a)
        assert obs.shape == (24, 24, 3)
        assert isinstance(r, float)
        assert -1.5 <= r <= 1.0
        assert "d" in info and "phi" in info and "step" in info
        assert abs(info["d"]) <= env.HALF_W + 0.1

def test_track_registry_and_all_presets():
    available = list_tracks()
    expected = ["circle", "oval", "figure8", "s_curve"]
    for name in expected:
        assert name in available

    for name in expected:
        track = get_track(name)
        assert track.name == name
        assert track.length > 10.0
        canvas = track.get_canvas()
        assert canvas.shape == (24, 24, 3)
        assert np.any(canvas == track.ROAD)

        env = MiniRacer(track=name, seed=123)
        obs = env.reset()
        assert obs.shape == (24, 24, 3)

        for step in range(150):
            a = step % 3
            obs, r, done, info = env.step(a)
            assert obs.shape == (24, 24, 3)
            assert not np.isnan(r)
            assert info["track"] == name
        assert done

