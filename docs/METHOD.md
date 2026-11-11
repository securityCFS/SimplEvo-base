# Method and code

## Direct control

At decision step `t` of episode `k` the model issues

    a_t = F(l, M_task, M_work[k-1], o_<=t, a_<t)

| Symbol | Meaning | Code |
|---|---|---|
| `l` | instruction with the robot constants | `TaskSpec`, `Embodiment.description`, `task_prompt` |
| `o_t` | RGB views the model asks for, and the measured robot state | `Environment.render`, `Environment.state`, `agent._observe` |
| `a_t` | a chunk of absolute end-effector waypoints | `control.tool_specs`, `Controller` |
| `M_task` | task memory | `TaskMemory`, written by `annotate` |
| `M_work` | working memory | `working_memory.MemoryTools`, stored by `AdaptationChain` |

A waypoint sets a pose and a gripper target for each arm (and a base pose on a mobile robot) and lasts a number of
environment steps. `motion_steps` ramps from the measured pose to the target; the rest of the steps hold it. The
harness neither plans nor approves motions. An episode ends when the evaluator registers success or the step limit
is reached; if the model stops earlier, the robot is held still until the evaluator ends the episode.

## Task memory

`annotate(session, demonstration, embodiment, out)` gives an annotator instance of the model three read-only tools
over one recording:

- `demo_search`: motion bouts per arm (frames, displacement) and gripper events;
- `demo_inspect`: frames of one camera, optionally cropped;
- `demo_actions`: measured motion over a frame range, relative to its first frame.

The annotator submits once with `submit_icl`: stages in time order (label, frame range, arms, what changes, whether
seen or inferred), labelled keyframes, and four text sections (arm roles, observed sequence, observed outcome,
evidence limits). A submission is refused until its stages cover the recording from the first to the last frames.
`render` turns it into the text every episode reads; the last part states the completed state, which gives the
model a stopping criterion before it has any experience. During an episode the model can read the task memory
again with `query_icl` and browse the recording with the same three tools.

## Working memory

The model-facing tools keep the names used in the prompts:

| Tool | Role |
|---|---|
| `read_task_notes`, `write_task_notes` | the working memory: lessons carried across the episodes of one task |
| `read_working`, `write_working` | a scratchpad for the current episode only |
| `history` | earlier attempts with their outcomes; the tool calls of one attempt |
| `inspect` | frames the model received earlier, optionally cropped and enlarged |
| `read_state_history` | measured robot states recorded with each observation |

`write_task_notes` replaces the whole memory inside a byte budget and needs the hash of the latest read, so a write
never builds on stale content and nothing is truncated. Its description asks for reusable lessons with evidence and
states that coordinates measured in one scene are not commands for another.

After an adaptation episode the same conversation gets one reflection turn (`agent._reflect`): the evaluator's
success and score, no robot action, only the memory tools, a bounded number of calls and at most two rewrites.
`show_outcome=False` withholds the result.

What crosses the episode boundary is a setting of the chain (`working_memory.CARRY`):

| Setting | Carried to the next episode |
|---|---|
| `none` | nothing |
| `lessons` | the working memory |
| `lessons+outcomes` (default) | the working memory and the outcome of each earlier attempt |
| `lessons+outcomes+frames` | as above, plus the earlier attempts' calls, frames and states |

## Adaptation and deployment

`adapt(chain, scenes, ...)` runs the training scenes in order. For each one the chain builds a read-only view, the
episode and its reflection run, and the chain commits once: the attempt record, the archive of what the model
received, and the notes if the model rewrote them. `chain.fix(dest)` seals the working memory with file hashes.
`deploy(fixed, scenes, ...)` gives every test episode a private copy of the snapshot, runs no reflection, and
verifies the snapshot after each episode.

## Files

A chain:

    chain.json                 task, carry setting, notes budget
    commits.jsonl              one line per committed episode with the state hash before and after
    committed/working_memory.md
    committed/attempts.jsonl   position, success, score, termination, index of key calls
    committed/episodes/NNN/    independent copy of each episode's evidence

A fixed memory: `working_memory.md`, `attempts.jsonl`, `view.json`, and the seal `fixed_memory.json`.

An episode:

    policy/initial_prompt.txt
    policy/initial_input/      state.json and one PNG per camera
    policy/tools/call_NNNNN/   request.json, response_summary.json, observation/
    policy/reflection_prompt.txt, policy/reflection.json
    policy/result.json
    memory_view/               what this episode could read
    memory_staging/            its notes, their versions, the scratchpad, the log of memory calls

Each chain has one writer. Views record their parent state hash; commits reject a stale parent and validate staged
notes before changing committed state. Resume and sealing check commit/attempt positions, archive directories and
the last committed hash. Multi-file commits are not database transactions: an interrupted commit requires manual
inspection, and the runner refuses to continue or seal it silently. Hashes detect accidental changes; they are not
signatures against someone who can rewrite both a snapshot and its seal.
