Status: ready-for-agent

# Specification: Modular & Customizable Road Geometry for Car Racing

## Problem Statement

The environment currently couples observation rendering, vehicle kinematics, reward calculation, and dataset collection to a hardcoded circular ring road. As a result, users cannot:
1. Evaluate how an agent trained on a circular road transfers or generalizes to unseen road topologies (such as ovals, figure-8 tracks, or S-curves).
2. Train World Models on diverse or specialized track shapes.
3. Visually inspect agent driving performance across different tracks either during live training feedback or during inference demos.

## Solution

Decouple the track geometry from the simulation environment by introducing a modular Track architecture. Tracks will be defined as closed 2D parametric curves with an arc-length Frenet frame (tracking curve progress $s$, lateral deviation $d$, and heading relative to tangent $\phi$). This preserves the existing discrete action space, reward scale, and observation dimensions ($24 \times 24 \times 3$).

A track registry will provide standard presets (`circle`, `oval`, `figure8`, `s_curve`) and support custom waypoint-based tracks. Dataset collection and caching will be versioned per track. Both the interactive Pygame viewer and the training visualizer will dynamically adapt to any selected track and display track metadata in the HUD.

## User Stories

1. As a researcher, I want to evaluate a controller trained exclusively on a circular track on an oval or figure-8 track without modifying code, so that I can benchmark zero-shot generalization to unseen road curvatures.
2. As a machine learning practitioner, I want to specify `--track <name>` when running inference demos, so that I can visually verify driving behavior on any supported track.
3. As a developer, I want a modular Track interface where I can register new road geometries by defining parametric formulas or waypoints, so that creating a new track does not require modifying environment kinematics or visualization code.
4. As a researcher, I want dataset collection to store rollouts labeled by track name (e.g. `dataset_circle.npz`, `dataset_oval.npz`), so that multiple datasets can coexist without clobbering one another.
5. As a researcher, I want the option to collect a composite dataset across multiple track topologies, so that the Vision (VAE) and Memory (MDN-RNN) models learn representations that generalize across varied track geometries.
6. As a user watching live training (`--render-learning`), I want to see the car driving on the currently selected training track in real time, so that I can visually inspect policy improvement on the exact track being targeted.
7. As a user in manual driving mode, I want to drive on different track layouts using arrow keys, so that I can calibrate the subjective difficulty and feel of each road shape.
8. As an interactive demo user, I want a hotkey in the visualizer to cycle between available tracks live, so that I can compare driving behaviors side-by-side without restarting the script.
9. As a developer, I want the track rendering to automatically draw asphalt and kerb borders onto the grass canvas for any track shape, so that visual observations remain visually consistent with the VAE's perceptual expectations.
10. As an automated test suite, I want to verify that all registered tracks produce valid $24 \times 24 \times 3$ observations, bounded reward signals, and non-crashing simulation steps across 150-step episodes.

## Implementation Decisions

### Modular Track Interface and Registry
- Establish an abstract Track interface responsible for:
  - Total curve length $L$ and closed-loop periodicity.
  - Position mapping: converting arc-length distance $s \in [0, L)$ to 2D center-line coordinates $(x, y)$.
  - Heading mapping: computing the local tangent angle $\psi(s) \in [-\pi, \pi)$ and curvature $\kappa(s)$.
  - Canvas generation: rasterizing the static $24 \times 24 \times 3$ background (grass, asphalt road of configurable width, and bounding kerbs).
  - Spawn state: providing initial position and orientation for episode reset.
- Implement a track registry with built-in presets:
  - `circle`: The baseline circular ring track of radius 8.0.
  - `oval`: A stadium track featuring straightaways connected by semicircular turns.
  - `figure8`: A self-intersecting or pinched figure-eight track testing alternating left and right steering transitions.
  - `s_curve` / `wavy`: A winding serpentine track testing rapid direction reversals.
  - `custom`: Extensible waypoint interpolation (spline or polyline).

### Kinematics and Frenet Coordinate Frame
- Maintain car kinematics in the local Frenet-Serret frame:
  - Longitudinal progress $s$ along the curve (periodically wrapped at total length $L$).
  - Lateral offset $d$ from the centerline, damped when exceeding the road half-width.
  - Heading angle error $\phi$ relative to the curve's local tangent.
- Progress along the curve updates via:
  $$\Delta s = \text{direction} \cdot \frac{\text{SPEED}}{1 - d \cdot \kappa(s)}$$
  where $\kappa(s)$ is the signed local curvature.
- The step reward remains normalized and track-agnostic:
  $$\text{reward} = 1.0 - \frac{|d|}{\text{HALF\_W}}$$

### Environment Parameterization
- Update the environment constructor to accept either a track name string or a Track instance: `MiniRacer(track="circle", seed=None)`.
- Default to `"circle"` to guarantee complete backward compatibility with existing tests, saved checkpoints, and scripts.

### Dataset Pipeline and Versioning
- Parameterize dataset collection and caching with the target track name.
- Store cached dataset files with distinct filenames (e.g. `random_dataset_<track>.npz`).
- Add a multi-track collection mode (`--track all`) that distributes rollout episodes evenly across all registered tracks.

### CLI and Visualization Integration
- Expose `--track` on all entry points (`run_demo.py`, `run_train.py`).
- Update Pygame HUD to display current track name alongside step, score, and steering telemetry.
- Support runtime track cycling in interactive visualizer modes using the `T` key.

## Testing Decisions

- **Testing Seam**: The primary test seam is the environment interface (`MiniRacer(track=...)`). Testing against this single public seam validates geometry rasterization, kinematics stepping, collision damping, and reward bounds without mocking internal details.
- **Contract Tests**:
  - Verify every registered track in the registry can instantiate `MiniRacer(track=name)`.
  - Verify that observations are strictly float32 arrays of shape `(24, 24, 3)` with values in $[0, 1]$.
  - Run multi-step rollout test suites (150 steps) for every registered track under random and fixed actions to verify that lateral displacements remain bounded and rewards stay within $(-\infty, 1.0]$.
  - Verify dataset generation works for individual tracks and composite (`all`) track configurations.
- **Prior Art**: Existing simulation behavior in `env.py` and evaluation runs in `car_racer/models/controller.py:evaluate_real`.

## Out of Scope

- Continuous action spaces or multi-agent vehicle collisions.
- Elevation changes, 3D perspective projection, or dynamic obstacles.
- Variable weather or non-grass background terrain.
- Complete physics simulation of tire friction, drifting, or engine torque dynamics.

## Further Notes

- Retaining the $24 \times 24 \times 3$ observation dimensions and color palette is critical to ensure that pre-trained Vision (ConvVAE) models can perceive new tracks without immediate retraining, facilitating direct zero-shot evaluation experiments.
