# 04: Adaptive Visualizer HUD and Live Interactive Track Switching

**What to build:** An updated Pygame viewer and telemetry HUD that visually renders the active track, displays track metadata in the telemetry bar, reflects `--track` during training evaluation (`--render-learning`), and allows users to cycle through tracks live at runtime using the `T` key.

**Blocked by:** 02: Track Registry and Core Geometry Presets

**Status:** resolved

- [x] Pygame HUD overlay displays active track name and local track curvature.
- [x] `run_demo.py` accepts `--track <name>` for all visualizer modes.
- [x] Pressing `T` in manual or inference demo mode cycles to the next registered track live without crashing.
- [x] `--render-learning` in `run_train.py` renders evaluation laps on the track configured for training.
