# 03: Multi-Track Experience Collection and Per-Track Cache Versioning

**What to build:** Dataset collection and loading utilities parameterized by track, caching datasets under distinct filenames (`random_dataset_<track>.npz`) and supporting a multi-track composite dataset (`--track all`) so models can be trained on specific or diverse track layouts.

**Blocked by:** 02: Track Registry and Core Geometry Presets

**Status:** ready-for-agent

- [ ] `collect_experience(track="...", ...)` collects rollouts on the specified track.
- [ ] `get_or_create_dataset(track="...", ...)` caches to `random_dataset_<track>.npz`.
- [ ] `track="all"` collects episodes distributed evenly across all registered tracks.
- [ ] `run_train.py` accepts `--track <name>` and passes it to dataset collection and evaluation.
