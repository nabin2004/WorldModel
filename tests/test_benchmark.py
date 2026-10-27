import os
from car_racer.models.controller import Controller
from car_racer.models.vae import ConvVAE
from car_racer.benchmark import run_cross_track_benchmark
from car_racer.config import Config

def test_cross_track_benchmark(tmp_path):
    controller = Controller()
    vae = ConvVAE()
    plot_file = str(tmp_path / "test_benchmark.png")
    results = run_cross_track_benchmark(
        controller=controller,
        vae=vae,
        episodes_per_track=2,
        horizon=10,
        plot_path=plot_file,
        verbose=False
    )
    assert "circle" in results
    assert "oval" in results
    assert "figure8" in results
    assert "s_curve" in results
    for name, metrics in results.items():
        assert "mean_return" in metrics
        assert "mean_steps" in metrics
        assert "mean_dev" in metrics
    assert os.path.exists(plot_file)
