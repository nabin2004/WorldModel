# 01: Track Base Interface and Circular Track Implementation

**What to build:** An abstract `Track` base interface and concrete `CircularTrack` implementation that decouples road geometry and rendering from vehicle simulation, while keeping 100% backward compatibility with existing tests, observations, rewards, and checkpoints.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] `car_racer/tracks/` package contains base `Track` interface defining length $L$, centerline $(x, y)(s)$, tangent angle $\psi(s)$, curvature $\kappa(s)$, and $24 \times 24$ rasterization.
- [ ] `CircularTrack` implements the baseline circular ring road ($R = 8.0, \text{HALF\_W} = 2.2$) matching the exact pixel representation and kerbs of existing `MiniRacer`.
- [ ] `MiniRacer` accepts a `track` parameter, defaulting to `CircularTrack`.
- [ ] Existing simulation behavior, step return values, and telemetry keys are preserved without regression.
