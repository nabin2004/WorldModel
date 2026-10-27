# 02: Track Registry and Core Geometry Presets (Oval, Figure-8, S-Curve)

**What to build:** A registry of tracks (`get_track(name)`) providing built-in presets (`oval`, `figure8`, `s_curve`) with generalized arc-length Frenet kinematics allowing cars to race on arbitrary closed curves with collision damping and centerline rewards.

**Blocked by:** 01: Track Base Interface and Circular Track Implementation

**Status:** resolved

- [x] `car_racer/tracks/registry.py` provides `register_track`, `get_track`, and `list_tracks`.
- [x] `OvalTrack` implemented with straightaways and semicircular turns.
- [x] `Figure8Track` implemented with alternating left and right curvature transitions.
- [x] `SCurveTrack` implemented with serpentine winding curves.
- [x] `MiniRacer(track=...)` steps along any registered track using generalized curve kinematics.
- [x] Bounded rewards ($1 - |d|/W$) and observation shapes ($(24, 24, 3)$) hold across all tracks for 150-step episodes.
