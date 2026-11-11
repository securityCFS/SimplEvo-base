"""The agent loop: one episode of direct control, then one reflection turn; and the two phases built on it.

    adapt   online adaptation on training scenes: act, reflect, commit, in a chain (working memory is updated)
    deploy  new scenes with the fixed working memory (no reflection, nothing is written back)

In both phases the model issues every robot command itself:

    a_t = F(l_tau, M^task, M^work_{k-1}, o_<=t, a_<t)

Everything the model received or issued is recorded under ``<episode>/policy`` in the layout the memory tools read
back (``tools/call_NNNNN/{request.json, response_summary.json, observation/}``).
"""
from __future__ import annotations

import json
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from .backends import ModelSession, ToolResult
from .chain import AdaptationChain, fixed_view, validate_fixed
from .content import as_json, image, text
from .control import Controller, InvalidAction, model_state
from .control import tool_specs as control_specs
from .demonstration import DEMO_TOOLS, Demonstration, DemoTools
from .demonstration import tool_specs as demo_specs
from .schema import problems
from .task import BASE_INSTRUCTIONS, Environment, Outcome, TaskSpec, task_prompt
from .task_memory import QUERY_GUIDANCE, TaskMemory
from .working_memory import MEMORY_TOOLS, REFLECTION_CALLS, MemoryTools, guidance, reflection_prompt

CONTROL_TOOLS = ("execute_eef", "set_gripper", "observe")
CONTINUE_AFTER_ERROR = (
    "The last model turn failed on a transient service error. Continue this SAME live robot task from the latest "
    "acknowledged state in this conversation. All earlier tool actions have already executed; do not replay them "
    "or reset the episode.")
CONTINUE_AFTER_SILENCE = (
    "The last model turn ended without an outcome. Continue this SAME live robot task from the latest acknowledged "
    "state in this conversation. All earlier tool actions have already executed; do not replay them or reset the "
    "episode. Continue controlling the robot toward the task, or state the observed outcome explicitly.")


@dataclass
class EpisodeRecord:
    outcome: Outcome | None
    termination: str            # "episode_complete": ended under the model's control; "model_finished": it stopped first
    tool_calls: int
    env_steps: int
    held_steps: int             # steps the robot was held still after the model stopped, until the evaluator's result
    messages: list[str]
    usage: dict
    reflection: dict | None = None

    def attempt(self) -> dict:
        """What later episodes may read about this one: the terminal outcome only."""
        return {"success": bool(self.outcome and self.outcome.success),
                "score": self.outcome.score if self.outcome else None, "termination": self.termination}


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def _observe(env: Environment, cameras, folder: Path) -> list[dict]:
    """o_t for the model, saved as the model received it: the measured state, then the requested RGB views."""
    folder.mkdir(parents=True, exist_ok=True)
    state = model_state(env)
    _write_json(folder / "state.json", state)
    items = [as_json({"measured_state": state})]
    for camera in cameras:
        png = env.render(camera)
        (folder / f"{camera}.png").write_bytes(png)
        items += [text(f"RGB view: {camera}"), image(png)]
    return items


def run_episode(env: Environment, session: ModelSession, task: TaskSpec, out: Path, *,
                task_memory: TaskMemory | None = None, demonstration: Demonstration | None = None,
                memory_view: Path | None = None, fixed: bool = False, show_outcome: bool = True,
                retries: int = 8, retry_wait: Callable[[int], None] | None = None) -> EpisodeRecord:
    """Run one episode in ``env`` and, unless the memory is ``fixed``, its reflection turn.

    ``memory_view`` is the view folder the chain built for this episode (or a private copy of a fixed memory);
    without it the agent acts with the task memory alone.
    """
    if env.embodiment != task.embodiment:
        raise ValueError("environment embodiment does not match the task embodiment")
    out = Path(out)
    policy = out / "policy"
    if policy.exists():
        raise RuntimeError("this episode folder was already used; an episode is never replayed")
    policy.mkdir(parents=True)
    embodiment = env.embodiment
    controller = Controller(env)
    demo_tools = DemoTools(demonstration) if demonstration is not None else None
    memory = MemoryTools(memory_view, policy, out / "memory_staging", embodiment.cameras) if memory_view else None
    control_schemas = {spec["name"]: spec["inputSchema"] for spec in control_specs(embodiment)}

    tools = control_specs(embodiment)
    if task_memory is not None:
        tools.append(task_memory.tool_spec())
    if demo_tools is not None:
        tools += demo_specs(demonstration)
    if memory is not None:
        tools += memory.tool_specs()
    instructions = BASE_INSTRUCTIONS + ("\n" + QUERY_GUIDANCE if task_memory is not None else "")
    prompt = task_prompt(task, (QUERY_GUIDANCE if task_memory is not None else "",
                                guidance(memory.position, memory.carry, fixed) if memory is not None else ""))
    (policy / "initial_prompt.txt").write_text(prompt + "\n", encoding="utf-8", newline="\n")
    calls = 0

    def handle(name: str, arguments: dict) -> ToolResult:
        nonlocal calls
        calls += 1
        folder = policy / "tools" / f"call_{calls:05d}"
        _write_json(folder / "request.json", {"tool": name, "arguments": arguments})
        success = True
        try:
            if name in CONTROL_TOOLS:
                found = problems(arguments, control_schemas[name])
                if found:
                    raise InvalidAction("; ".join(found[:5]))
                summaries = [] if name == "observe" else (
                    controller.execute(arguments) if name == "execute_eef" else controller.set_gripper(arguments))
                content = _observe(env, arguments["cameras"], folder / "observation")
                content.insert(0, as_json({"executed": summaries}))
            elif name == "query_icl" and task_memory is not None:
                content = task_memory.query(arguments)
            elif name in DEMO_TOOLS and demo_tools is not None:
                content = demo_tools.call(name, arguments)
            elif name in MEMORY_TOOLS and memory is not None:
                content = memory.call(name, arguments, calls)
            else:
                raise ValueError(f"unknown tool {name}")
        except (ValueError, KeyError, TypeError) as error:   # MemoryRefusal and DemonstrationError are ValueErrors
            if name in CONTROL_TOOLS and not isinstance(error, InvalidAction):
                raise   # Adapter failures may follow an executed motion; stop the episode.
            success = False
            content = [text(f"No further action executed. Invalid tool parameters: {error}")]
        if memory is not None:
            content.insert(0, text(f"[attempt {memory.position}, call {calls}]"))
        if env.ended:
            content.append(as_json({"episode_ended": True}))
        _write_json(folder / "response_summary.json", {"success": success, "env_steps": env.steps_used,
                                                       "episode_ended": env.ended})
        return ToolResult(content, success, end_turn=env.ended)

    session.start(tools, instructions)
    inputs = [text(prompt), *(task_memory.inputs() if task_memory is not None else []),
              *_observe(env, embodiment.cameras, policy / "initial_input")]
    messages, attempts, silent = [], 0, 0
    while True:
        turn = session.run_turn(inputs, handle)
        messages += turn.messages
        if env.ended:
            break
        if turn.status == "interrupted":
            raise RuntimeError("model turn interrupted before the environment ended")
        if turn.status == "failed":
            if attempts >= retries or not session.is_transient(turn.error):
                raise RuntimeError(f"model turn failed: {turn.error}")
            attempts += 1
            (retry_wait or (lambda n: time.sleep(min(300, 15 * 2 ** (n - 1)))))(attempts)
            inputs = [text(CONTINUE_AFTER_ERROR)]
        elif not turn.messages and silent < retries:
            silent += 1
            inputs = [text(CONTINUE_AFTER_SILENCE)]
        else:
            break
    termination = "episode_complete" if env.ended else "model_finished"
    held = 0 if env.ended else controller.hold_until_end()      # the evaluator, not the model, ends the episode
    if env.outcome is None:
        raise RuntimeError("environment ended without a terminal outcome")
    record = EpisodeRecord(env.outcome, termination, calls, env.steps_used, held, messages, session.usage())
    if memory is not None and not fixed:
        record.reflection = _reflect(session, memory, policy, env.outcome, held > 0, show_outcome)
        record.usage = session.usage()
    _write_json(policy / "result.json", asdict(record))
    return record


def _reflect(session: ModelSession, memory: MemoryTools, policy: Path, outcome, held: bool,
             show_outcome: bool) -> dict:
    """The reflection turn: the same conversation, the memory tools only, a bounded number of calls and writes."""
    memory.start_reflection()
    prompt = reflection_prompt(outcome, held, show_outcome)
    (policy / "reflection_prompt.txt").write_text(prompt + "\n", encoding="utf-8", newline="\n")
    record = {"calls": 0, "refused": 0}

    def handle(name: str, arguments: dict) -> ToolResult:
        record["calls"] += 1
        if name not in MEMORY_TOOLS or record["calls"] > REFLECTION_CALLS:
            record["refused"] += 1
            return ToolResult([text("The episode has ended; only the memory tools work"
                                    + (", and their budget is used up." if name in MEMORY_TOOLS else "."))], False,
                              end_turn=record["calls"] >= REFLECTION_CALLS)
        try:
            return ToolResult(memory.call(name, arguments, f"reflection-{record['calls']}"),
                              end_turn=record["calls"] >= REFLECTION_CALLS)
        except (ValueError, KeyError, TypeError) as error:
            record["refused"] += 1
            return ToolResult([text(f"Refused: {error}")], False, end_turn=record["calls"] >= REFLECTION_CALLS)

    turn = session.run_turn([text(prompt)], handle)
    record.update(status=turn.status, note_writes=memory.reflection_writes, messages=turn.messages)
    _write_json(policy / "reflection.json", record)
    return record


def adapt(chain: AdaptationChain, scenes, make_env: Callable[[object], Environment],
          make_session: Callable[[Path], ModelSession], task: TaskSpec, out: Path, **episode_options) -> list:
    """Online adaptation: run the training ``scenes`` in order along ``chain``, committing after each episode."""
    if chain.task != task.name:
        raise ValueError(f"chain belongs to task {chain.task!r}, not {task.name!r}")
    records = []
    for scene in scenes:
        position = chain.next_position()
        episode = Path(out) / f"{position:03d}"
        if episode.exists():
            raise RuntimeError(f"{episode} exists; an uncommitted episode needs a decision, not a silent rerun")
        view = episode / "memory_view"
        chain.build_view(position, view)
        with ExitStack() as resources:
            env = make_env(scene)
            resources.callback(env.close)
            session = make_session(episode / "session")
            resources.callback(session.close)
            record = run_episode(env, session, task, episode, memory_view=view, **episode_options)
        chain.commit(position, episode=episode, staging=episode / "memory_staging", attempt=record.attempt(),
                     arms=task.embodiment.arms)
        records.append(record)
    return records


def deploy(fixed_memory: Path | None, scenes, make_env: Callable[[object], Environment],
           make_session: Callable[[Path], ModelSession], task: TaskSpec, out: Path, **episode_options) -> list:
    """Deployment: every scene reads the same fixed working memory (or none) and nothing is written back."""
    if fixed_memory is not None and validate_fixed(fixed_memory)["task"] != task.name:
        raise ValueError(f"fixed memory does not belong to task {task.name!r}")
    records = []
    for index, scene in enumerate(scenes, 1):
        episode = Path(out) / f"{index:03d}"
        if episode.exists():
            raise RuntimeError(f"{episode} exists; a deployment episode is never replayed")
        view = None
        if fixed_memory is not None:
            view = episode / "memory_view"
            fixed_view(fixed_memory, view)
        with ExitStack() as resources:
            if fixed_memory is not None:
                resources.callback(validate_fixed, fixed_memory)
            env = make_env(scene)
            resources.callback(env.close)
            session = make_session(episode / "session")
            resources.callback(session.close)
            records.append(run_episode(env, session, task, episode, memory_view=view, fixed=True, **episode_options))
    return records
