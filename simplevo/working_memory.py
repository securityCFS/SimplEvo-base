"""Working memory M^work (paper Section 3.3): lessons the model rewrites after every adaptation episode.

After episode k the same model gets one reflection turn without robot actions and may replace the working memory
with a full rewrite inside a size budget:

    M^work_k = F(M^work_{k-1}, xi_k, r_k),   |M^work_k| <= B.

The model decides what the bounded memory keeps, and can revise or discard outdated lessons. This module holds the
model-facing side: the memory tools, the paragraph that introduces them, and the reflection prompt. The runner side
(what is committed and what the next episode may read) is ``simplevo.chain``.

Tool names and the paper's terms:

    read_task_notes / write_task_notes   the working memory, carried across the episodes of one task
    read_working / write_working         a scratchpad for the current episode only
    history, inspect, read_state_history exact records of the trajectory xi_k: calls, received frames, robot states
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from pathlib import Path

from PIL import Image

from .config import DEFAULTS
from .content import as_json, encode_png, image, text
from .schema import problems, tool

# What crosses the episode boundary. The paper's setting is "lessons+outcomes".
CARRY = {
    "none": {"notes": False, "attempts": False, "archive": False},                # every episode starts empty
    "lessons": {"notes": True, "attempts": False, "archive": False},
    "lessons+outcomes": {"notes": True, "attempts": True, "archive": False},
    "lessons+outcomes+frames": {"notes": True, "attempts": True, "archive": True},  # visual evidence too
}
NOTES_LIMIT = DEFAULTS["memory"]["notes_limit"]
SCRATCH_LIMIT = DEFAULTS["memory"]["scratch_limit"]
MAX_INSPECT_IMAGES = DEFAULTS["memory"]["max_inspect_images"]
MAX_INSPECT_PIXELS = DEFAULTS["memory"]["max_inspect_pixels"]
MAX_INSPECT_CALLS = DEFAULTS["memory"]["max_inspect_calls"]
MAX_INSPECT_SIDE = DEFAULTS["memory"]["max_inspect_side"]
HISTORY_PAGE = DEFAULTS["memory"]["history_page"]
STATE_PAGE = DEFAULTS["memory"]["state_page"]
REFLECTION_CALLS = DEFAULTS["memory"]["reflection_calls"]
REFLECTION_WRITES = DEFAULTS["memory"]["reflection_writes"]
MEMORY_TOOLS = ("read_task_notes", "write_task_notes", "read_working", "write_working", "history", "inspect",
                "read_state_history")
NOTES_FILE = "working_memory.md"


class MemoryRefusal(ValueError):
    """A refused memory request; the session continues and nothing was changed."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check_size(notes: str, limit: int) -> None:
    size = len(notes.encode("utf-8"))
    if size > limit:
        raise MemoryRefusal(f"task notes are {size} bytes; the limit is {limit} (shorten; nothing was saved)")


# ----------------------------------------------------------------------------- the recorded trajectory

def episode_calls(policy: Path) -> list[int]:
    """Tool call numbers of one episode record (``<episode>/policy`` or an archived copy), in order."""
    tools = Path(policy) / "tools"
    if not tools.is_dir():
        return []
    return sorted(int(p.name.removeprefix("call_")) for p in tools.iterdir() if re.fullmatch(r"call_\d{5}", p.name))


def _load(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def call_event(policy: Path, number: int) -> dict:
    folder = Path(policy) / "tools" / f"call_{number:05d}"
    request, summary = _load(folder / "request.json", {}), _load(folder / "response_summary.json", {})
    arguments = request.get("arguments")
    if request.get("tool") in MEMORY_TOOLS and isinstance(arguments, dict) and "content" in arguments:
        arguments = {**arguments, "content": f"<{len(arguments['content'])} chars>"}
    observation = folder / "observation"
    return {"call": number, "tool": request.get("tool"), "arguments": arguments, "success": summary.get("success"),
            "env_steps_after": summary.get("env_steps"), "episode_ended": bool(summary.get("episode_ended")),
            "frames": sorted(p.stem for p in observation.glob("*.png")) if observation.is_dir() else []}


def key_calls(policy: Path, arms) -> dict:
    """Index of an episode's decisive calls for its attempt record: commanded gripper transitions and the end."""
    index = {"gripper_close": [], "gripper_open": [], "last_call": None, "episode_end_call": None}
    closed = dict.fromkeys(arms, False)

    def command(arm, opening, number):
        if (opening < 0.5) != closed[arm]:      # transitions only: a gripper held closed is not a new grasp
            closed[arm] = opening < 0.5
            index["gripper_close" if closed[arm] else "gripper_open"].append({"call": number, "arm": arm})

    for number in episode_calls(policy):
        event = call_event(policy, number)
        arguments = event["arguments"] if isinstance(event["arguments"], dict) else {}
        if event["success"] and event["tool"] == "set_gripper" and arguments.get("side") in closed:
            command(arguments["side"], arguments.get("opening", 1), number)
        elif event["success"] and event["tool"] == "execute_eef":
            for waypoint in arguments.get("waypoints", []):
                for arm in arms:
                    if isinstance(waypoint.get(arm), dict) and "gripper" in waypoint[arm]:
                        command(arm, waypoint[arm]["gripper"], number)
        if event["episode_ended"] and index["episode_end_call"] is None:
            index["episode_end_call"] = number
        index["last_call"] = number
    return index


def _observation(policy: Path, call) -> Path:
    policy = Path(policy)
    return policy / "initial_input" if call == "initial" else policy / "tools" / f"call_{int(call):05d}" / "observation"


# ----------------------------------------------------------------------------- the memory tools

def tool_specs(notes_limit: int, cameras, limits: dict | None = None) -> list[dict]:
    limits = DEFAULTS["memory"] if limits is None else limits
    episode = {"anyOf": [{"type": "string", "enum": ["current"]}, {"type": "integer", "minimum": 1}],
               "description": "'current' or an earlier attempt number of this task (see history view='attempts')."}
    call = {"anyOf": [{"type": "string", "enum": ["initial"]}, {"type": "integer", "minimum": 1}],
            "description": "'initial' (the first observation) or a tool call number (each tool result states it)."}
    view = {"type": "object", "additionalProperties": False, "required": ["episode", "call", "camera", "region"],
            "properties": {"episode": episode, "call": call, "camera": {"type": "string", "enum": list(cameras)},
                           "region": {"anyOf": [{"type": "null"}, {
                               "type": "array", "minItems": 4, "maxItems": 4,
                               "items": {"type": "number", "minimum": 0, "maximum": 1}}]}}}
    return [
        tool("read_task_notes", "Read your task notes for THIS task (never shared across tasks). Returns the text "
             "and its sha256.", {}),
        tool("write_task_notes", (
            f"Replace your task notes (at most {notes_limit} bytes UTF-8; nothing is truncated). Pass the sha256 "
            "returned by your latest read or write. Write reusable, revisable, task-specific knowledge: what "
            "worked, what failed and why, with the evidence (attempt and call numbers). Coordinates measured in one "
            "scene are not commands for another scene."),
            {"content": {"type": "string", "maxLength": notes_limit},
             "expected_sha256": {"type": "string", "maxLength": 64}}),
        tool("read_working", "Read this attempt's scratchpad (current state, plan, open checks).", {}),
        tool("write_working", (
            f"Replace this attempt's scratchpad (at most {limits['scratch_limit']} bytes). Keep it current: your conversation "
            "may be compacted during a long attempt. It is not carried to later attempts."),
            {"content": {"type": "string", "maxLength": limits['scratch_limit']}}),
        tool("history", (
            "Exact records. view='attempts' lists the attempts at this task you may see (earlier ones with their "
            "official outcome and an index of key calls); view='events' lists one attempt's tool calls in order "
            f"(tool, arguments, success, environment steps, saved frames), a page of at most {limits['history_page']} from "
            "'start'."),
            {"view": {"type": "string", "enum": ["attempts", "events"]}, "episode": episode,
             "start": {"type": "integer", "minimum": 0},
             "limit": {"type": "integer", "minimum": 1, "maximum": limits['history_page']}}),
        tool("inspect", (
            "Return ARCHIVED RGB frames you received earlier (this attempt, or earlier attempts where available), "
            f"optionally cropped to a normalized region [x0, y0, x1, y1] and enlarged up to {limits['max_inspect_side']} px. "
            f"At most {limits['max_inspect_images']} images and {limits['max_inspect_pixels']} pixels per call; {limits['max_inspect_calls']} "
            "calls per attempt for this attempt's frames and as many for earlier attempts' frames. Archived frames "
            "show the past, not the current scene; use observe for the current scene."),
            {"question": {"type": "string", "maxLength": 300},
             "views": {"type": "array", "minItems": 1, "maxItems": limits['max_inspect_images'], "items": view}}),
        tool("read_state_history", (
            "Measured robot state recorded with each observation of one attempt, a page of at most "
            f"{limits['state_page']} from 'start' (index 0 is the initial observation). No object poses."),
            {"episode": episode, "start": {"type": "integer", "minimum": 0},
             "limit": {"type": "integer", "minimum": 1, "maximum": limits['state_page']}}),
    ]


def guidance(position: int, carry: str, fixed: bool = False) -> str:
    """The paragraph of the first prompt that introduces the memory. ``fixed``: deployment with a fixed memory."""
    rules = CARRY[carry]
    if not rules["notes"]:
        carried = "Nothing from earlier attempts is available, and your notes are not kept after this attempt."
    else:
        carried = ("read_task_notes holds what earlier attempts at this task recorded (it may be empty), and "
                   "write_task_notes replaces it for later attempts."
                   + (" history view='attempts' lists the earlier attempts' official outcomes and key calls."
                      if rules["attempts"] else "")
                   + (" Their exact events, archived frames and measured states are available through history, "
                      "inspect and read_state_history." if rules["archive"] else ""))
    ending = ("The prior task notes and attempt records are a fixed adaptation snapshot. Changes you make during "
              "this episode are private to this episode and will not be carried to another one. There is no "
              "post-result reflection turn." if fixed else
              "After the episode ends you get one turn, with the official result, to update the task notes.")
    return (f"Working memory: this is attempt {position} of a sequence of attempts at this same task in different "
            f"scenes. {carried} read_working/write_working is this attempt's scratchpad; history, inspect and "
            "read_state_history give exact records and archived views of this attempt; each tool result states its "
            "call number. Before each robot action, briefly predict what you expect to see; after it, describe what "
            f"actually changed. Notes are hypotheses: verify them against the current views. {ending}")


def reflection_prompt(outcome, held: bool, show_outcome: bool = True, writes: int = REFLECTION_WRITES) -> str:
    """The message that opens the reflection turn. ``show_outcome=False`` withholds the evaluator's result."""
    result = (f"Official result from the evaluation server: success={1.0 if outcome and outcome.success else 0.0}, "
              f"score={outcome.score if outcome else None}." if show_outcome else
              "The official result is not shown.")
    ending = (" You ended your control before the server's result; the robot was then held still until the server "
              "ended the episode." if held else "")
    return (f"The episode has ended. {result}{ending} No robot action is possible now; only the memory tools work. "
            f"Review this attempt if useful, then update your task notes with write_task_notes (at most "
            f"{writes} writes) so that later attempts at this task in other scenes do better: what "
            "worked, what failed and the evidence (attempt and call numbers). Keep the notes reusable and "
            "revisable. Reply with one short sentence when done.")


class MemoryTools:
    """The memory tools of one episode.

    ``view``: the read-only folder the chain built for this episode; ``policy``: the live episode record;
    ``staging``: a writable folder of this episode, where its notes and scratchpad are kept until the commit.
    """

    def __init__(self, view: Path, policy: Path, staging: Path, cameras, *, limits: dict | None = None):
        self.limits = dict(DEFAULTS["memory"] if limits is None else limits)
        self.view, self.policy, self.staging = Path(view), Path(policy), Path(staging)
        meta = json.loads((self.view / "view.json").read_text(encoding="utf-8"))
        self.carry, self.position = meta["carry"], meta["position"]
        self.visible, self.notes_limit = meta["visible_positions"], meta["notes_limit"]
        if self.visible and not CARRY[self.carry]["archive"]:
            raise MemoryRefusal("the view exposes earlier episodes to a setting without the archive")
        self.cameras = tuple(cameras)
        self.staging.mkdir(parents=True, exist_ok=True)
        if not (self.staging / NOTES_FILE).exists():
            shutil.copyfile(self.view / NOTES_FILE, self.staging / NOTES_FILE)
        self.specs = {spec["name"]: spec["inputSchema"] for spec in self.tool_specs()}
        self.inspect_calls = {"current": 0, "archived": 0}
        self.reflecting = False
        self.reflection_writes = 0

    def tool_specs(self) -> list[dict]:
        return tool_specs(self.notes_limit, self.cameras, self.limits)

    def start_reflection(self) -> None:
        self.reflecting = True

    def notes(self) -> str:
        return (self.staging / NOTES_FILE).read_text(encoding="utf-8")

    def call(self, name: str, arguments: dict, call_number=None) -> list[dict]:
        if name not in self.specs:
            raise MemoryRefusal(f"unknown memory tool {name}")
        found = problems(arguments, self.specs[name])
        if found:
            raise MemoryRefusal("; ".join(found[:5]))
        content = getattr(self, "_" + name)(arguments)
        record = {"time": time.time(), "tool": name, "position": self.position, "call": call_number,
                  "reflection": self.reflecting, "notes_sha256": sha256(self.notes().encode()),
                  "arguments": {k: (f"<{len(v)} chars>" if k == "content" else v) for k, v in arguments.items()}}
        with (self.staging / "memory_calls.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record) + "\n")
        return content

    def _episode(self, which) -> Path:
        if which == "current":
            return self.policy
        if which not in self.visible:
            raise MemoryRefusal(f"attempt {which!r} is not available here; available earlier attempts: {self.visible}")
        return self.view / "episodes" / f"{which:03d}" / "policy"

    def _read_task_notes(self, _):
        notes = self.notes()
        return [as_json({"sha256": sha256(notes.encode()), "notes": notes})]

    def _write_task_notes(self, arguments):
        if self.reflecting and self.reflection_writes >= self.limits['reflection_writes']:
            raise MemoryRefusal(f"at most {self.limits['reflection_writes']} note writes after the episode; nothing was saved")
        if arguments["expected_sha256"] != sha256(self.notes().encode()):
            raise MemoryRefusal("expected_sha256 does not match the current notes; read them again first")
        check_size(arguments["content"], self.notes_limit)
        self.reflection_writes += self.reflecting
        versions = self.staging / "versions"
        versions.mkdir(exist_ok=True)
        name = f"{len(list(versions.iterdir())) + 1:03d}{'_reflection' if self.reflecting else ''}.md"
        (versions / name).write_text(arguments["content"], encoding="utf-8", newline="\n")
        temporary = self.staging / (NOTES_FILE + ".tmp")
        temporary.write_text(arguments["content"], encoding="utf-8", newline="\n")
        temporary.replace(self.staging / NOTES_FILE)
        return [as_json({"saved": True, "sha256": sha256(arguments["content"].encode())})]

    def _read_working(self, _):
        path = self.staging / "scratchpad.md"
        return [text(path.read_text(encoding="utf-8") if path.exists() else "")]

    def _write_working(self, arguments):
        if self.reflecting:
            raise MemoryRefusal("the scratchpad is closed after the episode; update the task notes instead")
        data = arguments["content"].encode("utf-8")
        if len(data) > self.limits['scratch_limit']:
            raise MemoryRefusal(f"scratchpad is {len(data)} bytes; the limit is {self.limits['scratch_limit']}; nothing was saved")
        (self.staging / "scratchpad.md").write_bytes(data)
        return [as_json({"saved": True, "bytes": len(data)})]

    def _history(self, arguments):
        start, limit = arguments["start"], arguments["limit"]
        if arguments["view"] == "attempts":
            rows = []
            if CARRY[self.carry]["attempts"]:
                rows = [json.loads(line) for line in (self.view / "attempts.jsonl").read_text(encoding="utf-8").splitlines() if line]
            rows.append({"position": self.position, "status": "this attempt" + (" (ended)" if self.reflecting else "")})
            return [as_json({"attempts": rows[start:start + limit], "total": len(rows)})]
        policy = self._episode(arguments["episode"])
        calls = episode_calls(policy)
        return [as_json({"episode": arguments["episode"], "total": len(calls),
                         "events": [call_event(policy, number) for number in calls[start:start + limit]]})]

    def _inspect(self, arguments):
        sources = {"current" if view["episode"] == "current" else "archived" for view in arguments["views"]}
        for source in sources:
            if self.inspect_calls[source] >= self.limits['max_inspect_calls']:
                raise MemoryRefusal(f"the {source}-frame inspect budget of {self.limits['max_inspect_calls']} calls is used up")
        pictures, pixels = [], 0
        for view in arguments["views"]:
            path = _observation(self._episode(view["episode"]), view["call"]) / f"{view['camera']}.png"
            if not path.is_file():
                raise MemoryRefusal(f"no archived {view['camera']} frame for call {view['call']}")
            picture = Image.open(path).convert("RGB")
            box = (0, 0, picture.width, picture.height)
            if view["region"] is not None:
                x0, y0, x1, y1 = view["region"]
                if not (x0 < x1 and y0 < y1):
                    raise MemoryRefusal("region needs x0 < x1 and y0 < y1")
                left, top = int(x0 * picture.width), int(y0 * picture.height)
                box = (left, top, max(left + 1, round(x1 * picture.width)), max(top + 1, round(y1 * picture.height)))
                picture = picture.crop(box)
            scale = min(2.0, self.limits['max_inspect_side'] / max(picture.size))
            if scale > 1:
                picture = picture.resize((round(picture.width * scale), round(picture.height * scale)),
                                         Image.Resampling.LANCZOS)
            pixels += picture.width * picture.height
            pictures.append((view, box, picture))
        if pixels > self.limits['max_inspect_pixels']:
            raise MemoryRefusal(f"{pixels} pixels requested; the limit is {self.limits['max_inspect_pixels']} per call")
        for source in sources:
            self.inspect_calls[source] += 1
        content = []
        for view, box, picture in pictures:
            content += [text(f"ARCHIVED frame (not the current scene): attempt {view['episode']}, call "
                             f"{view['call']}, {view['camera']}, source box {list(box)}, "
                             f"{picture.width}x{picture.height}"), image(encode_png(picture))]
        return content

    def _read_state_history(self, arguments):
        policy = self._episode(arguments["episode"])
        calls = ["initial", *episode_calls(policy)]
        calls = [call for call in calls if (_observation(policy, call) / "state.json").exists()]
        page = calls[arguments["start"]:arguments["start"] + arguments["limit"]]
        return [as_json({"episode": arguments["episode"], "total": len(calls), "states": [
            {"call": call, "state": json.loads((_observation(policy, call) / "state.json").read_text(encoding="utf-8"))}
            for call in page]})]
