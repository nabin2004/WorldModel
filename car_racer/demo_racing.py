"""
Demo Script: Visualizing Car Racing During Learning and Inferencing.

Usage:
    # 1. Watch trained car racing (inference mode):
    python demo_racing.py --mode inference

    # 2. Watch car racing during learning progression (Gen 0 vs Gen 10 vs Gen 20):
    python demo_racing.py --mode learning

    # 3. Watch car racing inside M's dream (imagination):
    python demo_racing.py --mode dream

    # 4. Play manually with Arrow Keys (Left / Up / Right):
    python demo_racing.py --mode manual

    # 5. Export a high-resolution animated GIF:
    python demo_racing.py --mode gif
"""
import os
import sys
import time
import argparse
import numpy as np
import torch

# Ensure repository root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
for path in [SCRIPT_DIR, PARENT_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from car_racer.config import Config
from car_racer.env import MiniRacer
from car_racer.models.vae import ConvVAE
from car_racer.models.mdn_rnn import MDNRNN
from car_racer.models.controller import Controller
from car_racer.visualize import (
    PygameCarRacerViewer,
    run_inference_episode,
    run_dream_race,
    save_racing_gif,
    ACTION_NAMES
)

from car_racer.tracks.registry import list_tracks

def play_manual_mode(fps=25, track="circle"):
    """
    Allows the human user to drive the car with arrow keys to experience the physics.
    Controls:
        Left Arrow: Steer Left
        Up / Down: Keep Straight
        Right Arrow: Steer Right
        T Key: Cycle to next track layout
    """
    import pygame
    current_track = track
    env = MiniRacer(track=current_track, seed=None)
    viewer = PygameCarRacerViewer(scale=16, fps=fps, show_vae=False)

    print("\n--- Manual Car Racing Mode ---")
    print("Controls: Left/Right Arrow (Steer), Up Arrow (Straight), T (Cycle Track)")
    print("Close window to exit.")

    obs = env.reset()
    cum_score = 0.0
    action = 1

    for step in range(1, 300):
        # Process keyboard input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                viewer.close()
                return
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_t:
                tracks = list_tracks()
                idx = tracks.index(current_track) if current_track in tracks else 0
                current_track = tracks[(idx + 1) % len(tracks)]
                env = MiniRacer(track=current_track, seed=None)
                obs = env.reset()
                cum_score = 0.0
                print(f"[Manual Mode] Switched track to: {current_track.upper()} (T key)")

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action = 0
        elif keys[pygame.K_RIGHT]:
            action = 2
        else:
            action = 1

        obs, r, done, info = env.step(action)
        cum_score += r
        info["step"] = step
        info["score"] = cum_score
        info["reward"] = r
        info["action"] = action
        info["mode"] = "Manual Driver (Human)"

        viewer.draw_frame(real_obs=obs, recon_obs=None, info=info)

        if done:
            print(f"Episode finished! Score: {cum_score:.1f}")
            obs = env.reset()
            cum_score = 0.0

    viewer.close()


def show_learning_progression(controller, vae, fps=25, track="circle"):
    """
    Shows a 3-stage demonstration of how the car races during learning:
    Stage 1: Gen 0 (Untrained - pins to kerb)
    Stage 2: Intermediate (Random perturbations / semi-trained)
    Stage 3: Gen 20 (Dream-trained - smooth racing)
    """
    print("\n" + "=" * 60)
    print(f" SHOWING CAR RACING EVOLUTION ACROSS LEARNING STAGES (Track: {track.upper()})")
    print("=" * 60)

    stages = [
        ("Stage 1: Gen 0 (Untrained Baseline - Pins to Kerb)", np.zeros(51, dtype=np.float32)),
        ("Stage 2: Mid-Learning (Perturbed Driving)", np.random.randn(51).astype(np.float32) * 0.3),
        ("Stage 3: Fully Dream-Trained Controller (Smooth Cornering)", controller.get_params()),
    ]

    for title, params in stages:
        print(f"\n>>> Now racing: {title}")
        test_c = Controller(params=params)
        env = MiniRacer(track=track, seed=1001)
        viewer = PygameCarRacerViewer(scale=16, fps=fps, show_vae=(vae is not None))

        obs = env.reset()
        cum_score = 0.0

        for step in range(1, 90):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=Config.DEVICE).unsqueeze(0)
            recon = None
            if vae is not None:
                with torch.no_grad():
                    z, _ = vae.encode(obs_t)
                    a = test_c.act(z)
                    recon = vae.decode(z).squeeze(0).cpu().numpy()
            else:
                a = test_c.act(np.zeros(16))

            obs, r, done, info = env.step(a)
            cum_score += r
            info["step"] = step
            info["score"] = cum_score
            info["reward"] = r
            info["action"] = a
            info["mode"] = title.split(":")[0]

            viewer.draw_frame(real_obs=obs, recon_obs=recon, info=info)
            if done:
                break

        print(f"Result for {title}: Score = {cum_score:.1f}/90")
        time.sleep(1.2)
        viewer.close()


def main():
    parser = argparse.ArgumentParser(description="Live Visualizer for Car Racing")
    parser.add_argument("--mode", type=str, choices=["inference", "learning", "dream", "manual", "gif"],
                        default="inference",
                        help="Mode: 'inference' (live race), 'learning' (compare learning stages), 'dream' (M's imagination), 'manual' (drive yourself), 'gif' (export animation)")
    parser.add_argument("--track", type=str, default="circle", help="Track name: circle, oval, figure8, s_curve")
    parser.add_argument("--seed", type=int, default=1001, help="Environment seed")
    parser.add_argument("--fps", type=int, default=25, help="Playback speed FPS")
    parser.add_argument("--checkpoint-dir", type=str, default=Config.CHECKPOINT_DIR, help="Directory containing model checkpoints")
    parser.add_argument("--gif-out", type=str, default=os.path.join(Config.OUTPUT_DIR, "racing_showcase.gif"), help="Output path for GIF")
    args = parser.parse_args()

    # Try loading existing models if present
    vae = None
    vae_file = os.path.join(args.checkpoint_dir, "vae.pt")
    if os.path.exists(vae_file):
        vae = ConvVAE().to(Config.DEVICE)
        vae.load(vae_file, device=Config.DEVICE)
        print(f"Loaded VAE from {vae_file}")

    controller = Controller()
    c_file = os.path.join(args.checkpoint_dir, "controller.npy")
    if os.path.exists(c_file):
        controller.load(c_file)
        print(f"Loaded Controller from {c_file}")
    else:
        print("[Notice] No trained controller checkpoint found. Using default parameters.")

    if args.mode == "manual":
        play_manual_mode(fps=args.fps, track=args.track)
    elif args.mode == "learning":
        show_learning_progression(controller, vae, fps=args.fps, track=args.track)
    elif args.mode == "dream":
        mdn = MDNRNN().to(Config.DEVICE)
        mdn_file = os.path.join(args.checkpoint_dir, "mdn.pt")
        if not os.path.exists(mdn_file):
            print(f"[Error] Cannot run dream mode: checkpoint {mdn_file} does not exist. Train M first!")
            return
        if vae is None:
            print("[Error] VAE checkpoint needed for dream mode.")
            return
        mdn.load(mdn_file, device=Config.DEVICE)
        run_dream_race(mdn, vae, controller, start_seed=args.seed, fps=args.fps)
    elif args.mode == "gif":
        if vae is None:
            print("[Warning] No VAE loaded, initializing default VAE for GIF reconstruction.")
            vae = ConvVAE().to(Config.DEVICE)
        save_racing_gif(controller, vae, output_path=args.gif_out, seed=args.seed, track=args.track)
    else:  # inference
        run_inference_episode(controller=controller, vae=vae, seed=args.seed, fps=args.fps, track=args.track)


if __name__ == "__main__":
    main()
