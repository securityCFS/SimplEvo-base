# Connecting a benchmark

A benchmark is connected by one class and one factory function. `simplevo/envs/toy.py` is a complete example.

## Environment

Subclass `simplevo.task.Environment`; one instance is one live episode in one scene.

| Member | Contract |
|---|---|
| `embodiment` | an `Embodiment`: arm names, camera names, whether there is a mobile base, limits, and the text that describes the robot to the model |
| `state()` | measured end-effector pose and gripper of each arm (and the base pose), in the frame the description states |
| `render(camera)` | PNG bytes of one RGB camera at the latest executed step |
| `step(actions)` | execute per-step targets, one environment step each; stop when the episode ends; return the number executed |
| `ended`, `outcome` | whether the evaluator ended the episode, and its `Outcome(success, score)` |
| `steps_used` | environment steps executed so far |
| `close()` | release the episode |

State and per-step action share one format:

```python
{"left":  {"xyz_m": [x, y, z], "quaternion_wxyz": [w, x, y, z], "gripper": g},
 "right": {...},
 "base":  {"xy_m": [x, y], "yaw_rad": yaw}}      # only with has_base
```

`gripper` is normalized: 1 open, 0 closed. The adapter converts end-effector targets into the benchmark's native
command (inverse kinematics, gripper scaling) and reports the measured pose back in the same frame.

Rules that keep the information protocol:

- `state()` holds proprioception only: no object poses, no contact flags from the simulator, no progress signal;
- `render` returns the robot's own cameras only, as RGB;
- the step limit is enforced inside `step`; it is not exposed.

The factory takes one scene identifier and returns an `Environment`:

```python
def make_env(scene: str) -> Environment: ...
```

Scene identifiers are opaque to SimplEvo; the caller decides which scenes are for adaptation and which for testing.

## Embodiment description

The description is the only place the model learns the robot's geometry. State the world frame and its axes, where
each arm is mounted, which link the pose refers to and along which axis the fingers extend (with one example
orientation), the gripper convention, and where each camera sits. Say explicitly that depth, object poses and camera
calibration are not available.

## Demonstration

Export one recording per task as described at the top of `simplevo/demonstration.py`, or build a `Demonstration`
with a `frame_reader` that decodes the benchmark's videos on demand. Poses must use the same frame as `state()`,
grippers the same normalization. The segmentation thresholds in `demonstration.segment` are per recorded frame;
adjust them to the recording rate.

## Another model

Implement `simplevo.backends.ModelSession`:

- `start(tools, instructions)`: open a conversation with the tool specifications (name, description, JSON schema);
- `run_turn(inputs, handle)`: send content items as a user message, call `handle(name, arguments)` for each tool
  call and return its content to the model, and stop the turn when a result has `end_turn=True`;
- `usage()`, `is_transient(error)`, `close()`.

Content items are `{"type": "text", "text": ...}` and `{"type": "image", "png": ...}`.
