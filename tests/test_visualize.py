import os
import numpy as np
from car_racer.visualize import (
    create_hud_overlay,
    save_racing_gif,
    save_learning_progression_gif,
    save_dream_gif
)
from car_racer.models.controller import Controller
from car_racer.models.vae import ConvVAE
from car_racer.models.mdn_rnn import MDNRNN
from car_racer.config import Config

def test_hud_overlay():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    res = create_hud_overlay(img, step=5, reward=0.8, cumulative_score=4.0, action=1, d=0.1, phi=0.05, direction=1, track_name="oval")
    assert res.shape == (100, 100, 3)

def test_save_racing_gif_custom_track(tmp_path):
    controller = Controller()
    vae = ConvVAE()
    out_gif = str(tmp_path / "test_track.gif")
    save_racing_gif(controller, vae, output_path=out_gif, num_steps=5, seed=123, track="s_curve")
    assert os.path.exists(out_gif)
    assert os.path.getsize(out_gif) > 0

def test_save_learning_progression_gif(tmp_path):
    controller = Controller()
    vae = ConvVAE()
    out_gif = str(tmp_path / "test_progression.gif")
    save_learning_progression_gif(controller, vae, output_path=out_gif, num_steps=4, seed=123, track="circle")
    assert os.path.exists(out_gif)
    assert os.path.getsize(out_gif) > 0

def test_save_dream_gif(tmp_path):
    controller = Controller()
    vae = ConvVAE()
    mdn = MDNRNN()
    out_gif = str(tmp_path / "test_dream.gif")
    save_dream_gif(mdn, vae, controller, output_path=out_gif, start_seed=42, dream_steps=5)
    assert os.path.exists(out_gif)
    assert os.path.getsize(out_gif) > 0
