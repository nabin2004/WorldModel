# AGENTS.md

Instructions and repository context for AI agents working in this codebase.

## Repository Overview

Mini World Models: an implementation of Ha & Schmidhuber's classic World Models (2018) architecture on a top-down 2D car racing track. The architecture decouples observation encoding, recurrent predictive simulation, and policy evolution:
- **Vision (V)**: ConvVAE (`car_racer/models/vae.py`) encoding `24x24x3` frames into a 16-dimensional latent vector $z$.
- **Memory (M)**: GRU + MDN + Reward Head (`car_racer/models/mdn_rnn.py`) predicting $\hat{z}_{t+1}$ and $\hat{r}_{t+1}$.
- **Controller (C)**: Linear policy (`car_racer/models/controller.py`) with 51 parameters, evolved purely inside $M$'s hallucinated dream using Natural Evolution Strategies (NES).
- **Environment**: Kinematics and telemetry simulation (`car_racer/env.py`).
- **Tracks**: Modular road geometries (`car_racer/tracks/`) with built-in presets (`circle`, `oval`, `figure8`, `s_curve`).

## Tooling and Environment

- **Python**: Pin is 3.11 (see `.python-version`).
- **Package & Environment Manager**: [uv](https://docs.astral.sh/uv/). Always use `uv` for dependencies and command execution.

### Key Commands

```bash
# Environment sync
uv sync

# Run interactive racing demo, benchmark, or visualizer
uv run python run_demo.py --mode inference --track oval
uv run python run_demo.py --mode benchmark
uv run python run_demo.py --mode learning
uv run python run_demo.py --mode dream
uv run python run_demo.py --mode manual
uv run python run_demo.py --mode gif --track figure8 --gif-out outputs/car_race_demo.gif

# Training pipeline on custom track or all tracks
uv run python run_train.py --train-all --track circle
uv run python run_train.py --train-v --track all --epochs-v 30
uv run python run_train.py --train-m --epochs-m 30
uv run python run_train.py --train-c --gens-c 20 --track oval
uv run python run_train.py --train-c --render-learning --render-interval 5 --track figure8
```

---

## Agent skills

### Issue tracker

Issues and specs live as local markdown files under `.scratch/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Canonical triage roles (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context repository layout (`GLOSSARY.md` and `docs/adr/` at root). See `docs/agents/domain.md`.
