# Changelog

## 1.0.0

First public release of the benchmark-independent SimplEvo implementation.

- Task-memory annotation from one demonstration, direct end-effector control, working-memory adaptation along a
  chain with one reflection turn per episode, and deployment with a sealed working memory.
- `simplevo baseline`: direct control without working memory, with the task memory (the paper's baseline) or with
  the instruction alone. `examples/summarize_runs.py` averages success rate and score per task, as in the paper.
- YAML configuration (`simplevo/defaults.yaml`) for model settings, retries, memory and reflection budgets,
  demonstration tools, motion segmentation, annotation and toy runs. Every command accepts `--config`, the Python
  API accepts `config=`, and each episode and annotation saves its resolved settings.
- A Codex app-server backend with an ephemeral, read-only session and host-side tool dispatch, and a scripted
  backend for offline runs.
- An offline toy tabletop and an optional headless Panda pick-and-place playground with GIF and demonstration
  export.
- Integrity checks: every waypoint is validated before a motion starts; malformed robot descriptions,
  demonstrations and task-memory frames are rejected; annotation stages must cover the recording; stale views,
  changed archives, incomplete commits and mismatched tasks are detected before adaptation or deployment;
  cancelled episodes stop and environment failures propagate with the partial evidence kept.
- Offline regression tests, CI for editable and wheel installs, a complete source distribution, a citation file,
  and contributor and release guides.
