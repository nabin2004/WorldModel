# 05: Zero-Shot Cross-Track Generalization Benchmark Suite

**What to build:** A cross-track benchmark suite accessible via `run_demo.py --mode benchmark` that runs evaluation episodes of a trained controller across all registered track topologies and prints a comparative performance table (mean return, survival steps, centerline deviation).

**Blocked by:** 02: Track Registry and Core Geometry Presets, 04: Adaptive Visualizer HUD

**Status:** resolved

- [x] `run_demo.py --mode benchmark` executes evaluation rollouts on every registered track.
- [x] Outputs formatted benchmark metrics table comparing `circle`, `oval`, `figure8`, and `s_curve`.
- [x] Saves summary evaluation plot to `car_racer/outputs/zero_shot_generalization.png`.
