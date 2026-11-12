"""Task memory annotation (paper Section 3.2): an annotator instance of the MLLM reviews one demonstration.

The annotator browses the whole recording with the read-only demonstration tools and submits a structured
annotation once: the stages in order, labelled keyframes, the arm roles and the visible end state. It is told only
the task instruction; it sees no score and no success label, and it is asked for operations and order instead of
coordinates, so the result applies to layouts other than the demonstrated one. ``render`` turns the accepted
annotation into the ``TaskMemory`` every later episode reads.
"""
from __future__ import annotations

import json
from pathlib import Path

from .backends import ModelSession, ToolResult
from .config import DEFAULTS, resolve_config, save_config
from .content import encode_png, text
from .demonstration import DEMO_TOOLS, Demonstration, DemonstrationError, DemoTools, tool_specs
from .schema import problems, tool
from .task import Embodiment
from .task_memory import TaskMemory

MAX_KEYFRAMES = DEFAULTS["annotation"]["max_keyframes"]
MAX_STAGES = DEFAULTS["annotation"]["max_stages"]
SECTIONS = ("arm_roles", "observed_sequence", "observed_outcome", "evidence_limits")
PROMPT = """You are annotating one recorded robot demonstration so that a robot agent can learn from it later.
The robot: {robot} Task instruction: "{instruction}". The demonstration has {frames} frames (one frame is one
execution step).

Use demo_search to list its motion bouts and gripper events, demo_inspect to look at frames from the {cameras}
cameras, and demo_actions to read measured arm motion. Review the WHOLE demonstration, including its last frames:
stages are easy to miss when a scene is dark or an arm moves without a gripper event (for example closing a door or
a lid). Then call submit_icl once.

Write for a robot agent that will face a different layout of the same task: describe operations, arm roles, order,
contacts and the visible end state; say what you saw versus what you infer; do not report absolute coordinates or
frame-specific numbers as instructions. There is no success label for this recording; the instruction defines what
must be true at the end. Do not mention datasets, splits or scores."""
REMINDER = "You have not called submit_icl yet. Finish your review if needed and call submit_icl once."


def submit_spec(demo: Demonstration, has_base: bool = False, limits: dict | None = None) -> dict:
    limits = DEFAULTS["annotation"] if limits is None else limits
    parts = [*demo.arms, *(["both"] if len(demo.arms) == 2 else []), *(["base"] if has_base else []), "none"]
    return tool("submit_icl", (
        "Submit the annotation once you have reviewed the whole demonstration. Stages in time order cover the "
        "demonstration from its first to its last frame, including any final operation (closing, releasing, "
        f"withdrawing). Keyframes (at most {limits['max_keyframes']}) each illustrate a stage; pick the camera that shows "
        "that stage best."), {
        "stages": {"type": "array", "minItems": 1, "maxItems": limits['max_stages'], "items": {
            "type": "object", "additionalProperties": False,
            "required": ["label", "start", "end", "arms", "what_changes", "certainty"],
            "properties": {"label": {"type": "string", "maxLength": 120},
                           "start": {"type": "integer", "minimum": 0}, "end": {"type": "integer", "minimum": 1},
                           "arms": {"type": "string", "enum": parts},
                           "what_changes": {"type": "string", "maxLength": 300},
                           "certainty": {"type": "string", "enum": ["seen", "inferred", "uncertain"]}}}},
        "keyframes": {"type": "array", "minItems": 3, "maxItems": limits['max_keyframes'], "items": {
            "type": "object", "additionalProperties": False, "required": ["frame", "camera", "label"],
            "properties": {"frame": {"type": "integer", "minimum": 0},
                           "camera": {"type": "string", "enum": list(demo.cameras)},
                           "label": {"type": "string", "maxLength": 160}}}},
        **{section: {"type": "string", "maxLength": 1200} for section in SECTIONS},
        "reviewed": {"type": "string", "maxLength": 600,
                     "description": "Which frame ranges and cameras you actually looked at."}})


def validate(annotation: dict, demo: Demonstration, spec: dict) -> list[str]:
    """Why an annotation is not accepted: schema, frame ranges, and coverage of the whole recording."""
    found = problems(annotation, spec["inputSchema"])
    if found:
        return found[:5]
    frames, stages = demo.frames, annotation["stages"]
    if any(not 0 <= stage["start"] < stage["end"] <= frames for stage in stages):
        found.append(f"stage ranges must lie in [0, {frames}]")
    if stages[0]["start"] != 0 or max(stage["end"] for stage in stages) != frames:
        found.append("stages must cover the demonstration from its start to its last frames")
    covered, previous_start = 0, -1
    for stage in stages:
        if stage["start"] < previous_start or stage["start"] > covered:
            found.append("stages must be in time order and cover the recording without gaps")
            break
        previous_start, covered = stage["start"], max(covered, stage["end"])
    if any(not 0 <= key["frame"] < frames for key in annotation["keyframes"]):
        found.append(f"keyframes must lie in [0, {frames})")
    if len({(key["frame"], key["camera"]) for key in annotation["keyframes"]}) != len(annotation["keyframes"]):
        found.append("duplicate keyframes")
    return found


def render(annotation: dict, demo: Demonstration) -> TaskMemory:
    """The task memory text and labelled keyframes of an accepted annotation."""
    stages = "\n".join(f"- frames {stage['start']}-{stage['end']} ({stage['arms']}; {stage['certainty']}): "
                       f"{stage['label']}. {stage['what_changes']}" for stage in annotation["stages"])
    body = (f"One recorded demonstration of this task. Reference instruction: {demo.instruction}\n"
            f"Arm roles: {annotation['arm_roles']}\nObserved sequence: {annotation['observed_sequence']}\n"
            f"Stages (frame = one execution step of the recording):\n{stages}\n"
            f"Observed outcome: {annotation['observed_outcome']}\nEvidence limits: {annotation['evidence_limits']}\n"
            "This is a separate recorded episode with no success label; the live instruction defines what must be "
            "true at the end. Transfer operations, order and checks; ground every command in the live views and "
            "measured state.")
    keyframes = [(f"Demonstration frame {key['frame']}; {key['camera']} camera. {key['label']}",
                  encode_png(demo.frame_reader(key["camera"], key["frame"])))
                 for key in sorted(annotation["keyframes"], key=lambda key: key["frame"])]
    return TaskMemory(body, keyframes)


def annotate(session: ModelSession, demo: Demonstration, embodiment: Embodiment, out: Path, *, config: dict | None = None) -> TaskMemory:
    """Run the annotator on ``demo`` and save the annotation and the task memory under ``out``."""
    config = resolve_config(config)
    out = Path(out)
    if (out / "task_memory.json").exists():
        raise DemonstrationError(f"{out} already holds a task memory")
    out.mkdir(parents=True, exist_ok=True)
    save_config(out / "config.yaml", config)
    tools = DemoTools(demo, limits=config["demonstration"], segmentation=config["segmentation"])
    spec = submit_spec(demo, embodiment.has_base, config["annotation"])
    submitted, log = {}, []

    def handle(name: str, arguments: dict) -> ToolResult:
        log.append({"tool": name, "arguments": arguments})
        if "annotation" in submitted:
            return ToolResult([text("An annotation has already been accepted.")], False, end_turn=True)
        if name == "submit_icl":
            found = validate(arguments, demo, spec)
            if found:
                return ToolResult([text("Not accepted: " + "; ".join(found) + ". Fix and submit again.")], False)
            submitted["annotation"] = arguments
            return ToolResult([text("Annotation accepted.")], end_turn=True)
        if name not in DEMO_TOOLS:
            return ToolResult([text("Only the demonstration tools and submit_icl are available.")], False)
        try:
            return ToolResult(tools.call(name, arguments))
        except DemonstrationError as error:
            return ToolResult([text(f"Invalid request: {error}")], False)

    prompt = PROMPT.format(robot=embodiment.description, instruction=demo.instruction, frames=demo.frames,
                           cameras=", ".join(demo.cameras))
    (out / "prompt.txt").write_text(prompt + "\n", encoding="utf-8", newline="\n")
    session.start([*tool_specs(demo, config["demonstration"]), spec])
    session.run_turn([text(prompt)], handle)
    if "annotation" not in submitted:
        session.run_turn([text(REMINDER)], handle)
    (out / "calls.json").write_text(json.dumps(log, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    if "annotation" not in submitted:
        raise DemonstrationError("the annotator did not submit an annotation")
    (out / "annotation.json").write_text(json.dumps(submitted["annotation"], indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    memory = render(submitted["annotation"], demo)
    memory.save(out)
    return memory
