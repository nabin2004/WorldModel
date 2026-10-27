# 02: Track Registry and Core Geometry Presets (Oval, Figure-8, S-Curve)

**What to build:** A registry of tracks (`get_track(name)`) providing built-in presets (`oval`, `figure8`, `s_curve`) with generalized arc-length Frenet kinematics allowing cars to race on arbitrary closed curves with collision damping and centerline rewards.

**Blocked by:** 01: Track Base Interface and Circular Track Implementation

**Status:** ready-for-agent

- [ ] `car_racer/tracks/registry.py` provides `register_track`, `get_track`, and `list_tracks`.
- [ ] `OvalTrack` implemented with straightaways and semicircular turns.
- [ ] `Figure8Track` implemented with alternating left and right curvature transitions.
- [ ] `SCurveTrack` implemented with serpentine winding curves.
- [ ] `MiniRacer(track=...)` steps along any registered track using generalized curve kinematics.
- [ ] Bounded rewards ($1 - |d|/W$) and observation shapes ($(24, 24, 3)$) hold across all tracks for 150-step episodes.
