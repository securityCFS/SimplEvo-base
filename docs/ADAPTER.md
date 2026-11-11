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
The environment's `embodiment` must match the task file, including the description, arms, cameras and limits.
The runner owns the environment and model session created by their factories and closes both on success or error.
When calling `run_episode` directly, the caller owns their cleanup. An episode must end with a non-null `Outcome`;
an adapter that stops advancing without ending the episode raises an error instead of recording a completed run.
Adapter errors propagate to the caller. Cancelling a live model turn stops the episode and releases its resources;
the partial episode record remains available for inspection.

## Task file

The JSON file passed to `--task` has this form (replace the description with your robot's actual geometry):

```json
{
  "name": "block_to_pad",
  "instruction": "Put the red block on the green pad.",
  "guidance": "",
  "embodiment": {
    "description": "Describe the robot, reference frame, end-effector convention and RGB views here.",
    "arms": ["left", "right"],
    "cameras": ["overview", "left_wrist", "right_wrist"],
    "has_base": false,
    "max_waypoints": 6,
    "max_steps": 64
  }
}
```

`max_steps` limits each waypoint, not the episode. Arm and camera names must be distinct within each list and use
letters, digits, `_` or `-`, starting with a letter or digit. Arm names cannot be reserved action/state fields.
Each sealed memory is tied to the task name. Use a new chain for another task or a changed robot/task definition.

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
Recordings must be nonempty, use finite poses and unit quaternions, and keep gripper values in `[0, 1]`. A mobile
base trajectory has shape `(N, 3)`. Export every camera frame for annotation. `save(every=N)` creates a sparse
preview recording; requesting a missing frame returns a clear error and cannot substitute a nearby frame.

Annotation stages use half-open ranges `[start, end)` in chronological order. They must start at frame zero and
cover the whole recording without gaps; overlapping stages are allowed for simultaneous actions. Only the first
accepted `submit_icl` is kept.

## Another model

Implement `simplevo.backends.ModelSession`:

- `start(tools, instructions)`: open a conversation with the tool specifications (name, description, JSON schema);
- `run_turn(inputs, handle)`: send content items as a user message, call `handle(name, arguments)` for each tool
  call and return its content to the model, and stop the turn when a result has `end_turn=True`;
- `usage()`, `is_transient(error)`, `close()`.

Content items are `{"type": "text", "text": ...}` and `{"type": "image", "png": ...}`.

## Codex backend

The backend uses the experimental dynamic-tool interface over newline-delimited JSON on stdin/stdout. The
installed CLI's model catalog determines available model/effort choices. The protocol is described in the
[official app-server documentation](https://learn.chatgpt.com/docs/app-server); generate schemas from your installed
CLI when upgrading it. The local handshake and registration of all three control tools were checked with
`codex-cli 0.167.0`. This check did not issue a model turn.

```bash
codex --version
python examples/check_codex_protocol.py --out runs/codex-check --model gpt-6-astra --effort high
```

Use an installed, configured Codex CLI with access to the selected model. `SIMPLEVO_CODEX_BIN` can override its
executable command. The backend uses the CLI's current login; it does not read credentials into the package.
Its tests use a fake local server and need no login. A successful protocol check establishes the handshake and
tool registration only; a model-backed robot episode still needs validation with the selected model and adapter.
