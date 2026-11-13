# Changelog

## Unreleased

- Add an optional headless Panda pick-and-place playground, offline GIF/demo export, and a standard environment factory.

- Add a packaged YAML configuration for model settings, retries, memory and reflection budgets,
  demonstration tools, motion segmentation, annotation and toy runs.
- Accept `--config` on CLI commands and `config=` in the episode and annotation APIs; save resolved
  settings with each episode and annotation.
- Add paper author affiliations, equal-contribution and corresponding-author markers, and arXiv/PDF/video badges.


## Unreleased

- Add an optional headless Panda pick-and-place playground, offline GIF/demo export, and a standard environment factory.

- Stop cancelled live episodes and propagate environment failures after motion, preserving the partial episode
  evidence for inspection.
- Validate every waypoint before starting a motion batch, including quaternion and motion-duration checks.
- Reject malformed robot descriptions, empty or non-finite demonstrations, and out-of-folder task-memory frames.
- Require ordered annotation stages covering the recording without gaps; stop after the first accepted submission.
- Keep independent copies of episode evidence and memory views. Detect stale views, changed archives, incomplete
  commits and mismatched tasks before adaptation or deployment.
- Preserve an existing chain's carry setting unless an explicit matching value is supplied.
- Release environments when model construction or shutdown fails, require a terminal evaluator outcome, and stop
  reflection at its call budget. Refuse existing deployment directories before opening resources.
- Handle paginated Codex model catalogs, stale turn notifications, server interruptions and subprocess cleanup.
- Add regression tests, CI, complete source-distribution contents, a citation file and contributor/release guides.

## 1.0.0

Initial benchmark-independent implementation of task annotation, direct end-effector control, working-memory
adaptation and deployment with a fixed memory. Includes the scripted toy tabletop and Codex app-server backend.
