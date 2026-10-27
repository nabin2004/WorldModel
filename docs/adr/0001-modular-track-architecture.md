# ADR 0001: Modular Track Architecture and Generalized Frenet Kinematics

## Status
Accepted

## Context
The initial implementation hardcoded the car environment to a single circular track of radius 8.0 with fixed polar coordinates. This prevented evaluating policy transfer across diverse road topologies, collecting composite datasets, or visualizing generalization performance in real time.

## Decision
1. Introduce an abstract `Track` base class in `car_racer/tracks/` with a registry pattern (`get_track`, `list_tracks`, `register_track`).
2. Implement `ParametricClosedTrack` to support arbitrary 2D closed curves $(x(u), y(u))$ with numerical arc-length discretization and automatic asphalt/kerb canvas rasterization.
3. Retain the arc-length Frenet-Serret coordinate frame $(s, d, \phi, \text{direction})$ across all tracks. This preserves the exact discrete 3-action space (`left`, `straight`, `right`), observation dimensions ($24 \times 24 \times 3$), and normalized reward formulation ($1.0 - |d|/\text{HALF\_W}$).
4. Partition dataset collection and disk caching per track (e.g. `random_dataset_oval.npz`), with support for composite multi-track collection (`--track all`).
5. Support dynamic runtime track switching in the interactive Pygame viewer via the `T` key.

## Consequences
- **Positive**: Direct zero-shot generalization testing is now supported across any road geometry without altering model weights.
- **Positive**: Adding a new track requires only defining a parametric mathematical curve without modifying kinematics or rendering code.
- **Positive**: 100% backward compatibility is retained with existing `circle` checkpoints and tests.
- **Trade-off**: High-curvature tracks can cause excessive kerb scraping if a policy cannot countersteer fast enough, which provides a meaningful test of policy robustness.
