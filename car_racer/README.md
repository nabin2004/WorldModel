# Mini World Models: V · M · C on a Racing Track

A clean, modular implementation of Ha & Schmidhuber's classic **World Models** (2018) architecture on a top-down car racing track, featuring live real-time visualizers for **inferencing** and **learning**.

---

## 🏎️ Architecture Overview

The agent learns to drive **without interacting with the real environment during policy learning**, by hallucinating inside its own learned model:

```
                  ┌────────────────────────────────────────────────────────┐
                  │                   1. Vision (V)                        │
Observation ─────►│              Convolutional VAE                         │─────► Latent Code z (16)
(24×24×3 RGB)     │   Compresses raw pixels into smooth latent space       │
                  └────────────────────────────────────────────────────────┘
                                              │
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │                   2. Memory (M)                        │
Action a (t) ────►│      Recurrent World Model (GRU + MDN + Reward)        │─────► Next Latent Code ẑ
Hidden state h ──►│    Predicts distribution over future codes & rewards   │       + Imagined Reward r̂
                  └────────────────────────────────────────────────────────┘
                                              │
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │                 3. Controller (C)                      │
Latent Code z ───►│       Compact Linear Policy (51 Parameters)            │─────► Action a (t+1)
                  │   Trained purely inside M's dream using Evolution (ES) │
                  └────────────────────────────────────────────────────────┘
```

---

## 📁 Modular Codebase Structure

```
WOM/
├── car_racer/
│   ├── __init__.py           # Package exports
│   ├── config.py             # Hyperparameters, constants & paths
│   ├── env.py                # MiniRacer environment (physics, telemetry, reward)
│   ├── dataset.py            # Random exploration dataset collection & caching
│   ├── models/
│   │   ├── __init__.py
│   │   ├── vae.py            # V: ConvVAE (reparameterization, weighted reconstruction)
│   │   ├── mdn_rnn.py        # M: GRU + Mixture Density Network + Reward Head
│   │   └── controller.py     # C: 51-param linear policy + Evolution Strategies (NES)
│   ├── train.py              # Modular training pipeline with live render callbacks
│   ├── visualize.py          # Pygame GUI viewer, dream viewer & GIF export
│   ├── demo_racing.py        # Interactive CLI demo for learning & inferencing
│   ├── requirements.txt      # Python dependencies
│   ├── checkpoints/          # Saved model weights (vae.pt, mdn.pt, controller.npy)
│   ├── data/                 # Cached experience datasets
│   └── outputs/              # Saved GIF animations and plots
├── run_demo.py               # Root quick launcher for racing demo
├── run_train.py              # Root quick launcher for training
├── pyproject.toml            # Project configuration & dependencies (uv)
├── uv.lock                   # Deterministic lockfile (uv)
└── requirements.txt          # Python dependencies (pip fallback)
```

---

## 🚀 Quick Start

### 1. Environment Setup

This project uses [uv](https://docs.astral.sh/uv/) for ultra-fast dependency resolution and virtual environment management:

```bash
# Sync dependencies and set up .venv
uv sync

# Run demo or training directly with uv
uv run python run_demo.py --mode inference
```

Alternatively, activate the virtual environment or use standard `pip`:

```bash
# Windows
.venv\Scripts\activate

# Unix / macOS
source .venv/bin/activate

# (Fallback) Install via pip
pip install -r requirements.txt
```

---

## 🎮 Running Car Racing During Learning & Inferencing

We provide the dedicated `demo_racing.py` script (or `run_demo.py` from root) with multiple visual modes:

### 1. Real-Time Inferencing Mode (Trained Driver)
Watch the dream-trained agent drive in real-time with an upscaled view, side-by-side VAE reconstruction, and a rich telemetry HUD:

```bash
python run_demo.py --mode inference
```

**HUD Telemetry Displays:**
- **Current Mode**: Autonomous Dream-Trained C
- **Steering Action**: `LEFT ◄`, `STRAIGHT ▲`, or `RIGHT ►`
- **Deviation from Center Line ($d$)**: Distance from the center of the ring road
- **Steering Angle ($\phi$)**: Current car orientation relative to track
- **Direction**: Hidden episode coin flip (`Clockwise` or `Counter-Clockwise`)
- **Cumulative Score & Step Reward**

### 2. Watch Car Racing During Learning Progression
Watch how the car's driving behavior evolves through 3 distinct learning phases:
- **Phase 1 (Gen 0 Baseline)**: Car steers blindly into the kerb and gets stuck.
- **Phase 2 (Mid-Learning)**: Car starts reacting and exploring cornering angles.
- **Phase 3 (Dream-Trained Gen 20)**: Car smoothly corners and maintains center-line driving.

```bash
python run_demo.py --mode learning
```

### 3. Live Visual Racing While Training
Train the Controller while popping up a live racing window every N generations so you can observe the learning curve in real-time:

```bash
python run_train.py --train-c --render-learning --render-interval 5
```

### 4. Inside M's Dream (Pure Imagination Mode)
Unplug reality! Watch the car race **purely inside the memory model's recurrent imagination** without calling the game engine:

```bash
python run_demo.py --mode dream
```

### 5. Manual Play Mode (Drive with Keyboard)
Take control yourself using the Arrow Keys (`Left`, `Straight/Up`, `Right`) to test the physics and see how hard the track is:

```bash
python run_demo.py --mode manual
```

### 6. Export Animated GIF
Generate an animated GIF of the car racing with HUD telemetry:

```bash
python run_demo.py --mode gif --gif-out outputs/car_race_demo.gif
```

---

## 🛠️ Training Workflow

You can train any component individually or all together end-to-end:

```bash
# 1. Train everything end-to-end (V -> M -> C)
python run_train.py --train-all

# 2. Train only Vision (V - ConvVAE)
python run_train.py --train-v --epochs-v 30

# 3. Train only Memory (M - MDN-RNN)
python run_train.py --train-m --epochs-m 30

# 4. Evolve Controller (C) inside M's dream
python run_train.py --train-c --gens-c 20
```

---

## 🔬 Key Engineering Highlights

1. **Weighting What Moves**: The red car is only 4 out of 576 pixels (0.7% of the image). A naive MSE loss ignores the car and reconstructs an empty track. Our VAE assigns $30\times$ weight to non-background pixels.
2. **Smooth Latent Manifold**: Extra Gaussian noise (`+ 0.10`) is added to $z$ during decoder training, ensuring M's near-miss predictions decode to stable visual frames.
3. **Mixture Density Head**: Solves the hidden clockwise/counter-clockwise ambiguity by outputting $K=5$ Gaussian candidates rather than averaging them into a stationary blur.
4. **Zero Real-World Training for Policy**: Controller $C$ (51 parameters) never drives on the real track during evolution—it is evaluated solely on $M$'s learned reward head $\hat{r}$, yet achieves near-optimal real-world transfer.
