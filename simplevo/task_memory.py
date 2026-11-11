"""Task memory M^task (paper Section 3.2): the annotated demonstration, fixed during execution.

It lists the stages of the task in order and ends with the visual state of the completed task, which gives the
model a stopping criterion before it has any experience of its own. It holds no scene coordinates. The model reads
it at the start of every episode and can query it again with ``query_icl``.

On disk: ``task_memory.json`` ({"version": 1, "text": ..., "keyframes": [{"file": ..., "label": ...}]}) next to
the keyframe PNG files.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .content import image, text
from .schema import problems, tool

QUERY_GUIDANCE = (
    "The saved ICL demonstration remains available through query_icl throughout this episode. Revisit it when the "
    "target region, stage order or arm roles are unclear, and when repeated failures or a change of approach make "
    "the reference useful. Use view=text for the original annotation and keyframe index, view=frames with "
    "frame_indices for selected images, or view=all for the full reference. Indices are zero-based entries in the "
    "keyframe index, not source video frame numbers. Querying does not advance the environment. Ground actions in "
    "the current RGB/state.")
MAX_KEYFRAMES = 20


@dataclass
class TaskMemory:
    text: str
    keyframes: list[tuple[str, bytes]] = field(default_factory=list)   # (label, PNG bytes)

    def __post_init__(self):
        if not self.text.strip():
            raise ValueError("a task memory needs text")
        if len(self.keyframes) > MAX_KEYFRAMES:
            raise ValueError(f"at most {MAX_KEYFRAMES} keyframes")

    def save(self, folder: Path) -> Path:
        folder = Path(folder)
        (folder / "keyframes").mkdir(parents=True, exist_ok=True)
        entries = []
        for index, (label, png) in enumerate(self.keyframes):
            name = f"keyframes/{index:02d}.png"
            (folder / name).write_bytes(png)
            entries.append({"file": name, "label": label})
        path = folder / "task_memory.json"
        path.write_text(json.dumps({"version": 1, "text": self.text, "keyframes": entries}, indent=1,
                                   ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        return path

    @classmethod
    def load(cls, folder: Path) -> "TaskMemory":
        folder = Path(folder)
        data = json.loads((folder / "task_memory.json").read_text(encoding="utf-8"))
        if data.get("version") != 1:
            raise ValueError("unsupported task memory version")
        keyframes = []
        for entry in data["keyframes"]:
            relative = Path(entry["file"])
            path = (folder / relative).resolve()
            if relative.is_absolute() or not path.is_relative_to(folder.resolve()):
                raise ValueError("keyframe must stay inside the task memory folder")
            keyframes.append((entry["label"], path.read_bytes()))
        return cls(data["text"], keyframes)

    def inputs(self) -> list[dict]:
        """The items placed before the first live observation of an episode."""
        items = [text("HISTORICAL DEMONSTRATION: reference from a separate recorded episode, not the current "
                      "observation or the current episode's action history.\n" + self.text)]
        for label, png in self.keyframes:
            items += [text("Historical demonstration: " + label), image(png)]
        items.append(text("END OF HISTORICAL DEMONSTRATION. The following images and measured state are the CURRENT "
                          "episode. Ground all commands in the current observation; adapt the example instead of "
                          "replaying its coordinates or assuming its outcome."))
        return items

    @staticmethod
    def tool_spec() -> dict:
        return tool("query_icl", QUERY_GUIDANCE, {
            "view": {"type": "string", "enum": ["text", "frames", "all"]},
            "frame_indices": {"type": "array", "items": {"type": "integer", "minimum": 0}, "uniqueItems": True,
                              "description": "Keyframe indices; nonempty for frames, empty otherwise."}})

    def query(self, arguments: dict) -> list[dict]:
        found = problems(arguments, self.tool_spec()["inputSchema"])
        if found:
            raise ValueError("; ".join(found[:5]))
        view, indices = arguments["view"], arguments["frame_indices"]
        if any(index >= len(self.keyframes) for index in indices):
            raise ValueError("keyframe index out of range")
        if (view == "frames") != bool(indices):
            raise ValueError("use nonempty frame_indices for view=frames and empty ones otherwise")
        selected = list(range(len(self.keyframes))) if view == "all" else indices
        items = [text("HISTORICAL ICL REFERENCE: not a live observation. No environment step advanced.\n"
                      + (self.text + "\n" if view in ("text", "all") else "") + "Keyframe index:\n"
                      + "\n".join(f"{index}: {label}" for index, (label, _) in enumerate(self.keyframes)))]
        for index in selected:
            label, png = self.keyframes[index]
            items += [text(f"ICL keyframe {index}: {label}"), image(png)]
        items.append(text("END OF ICL REFERENCE. Use current RGB/state to adapt the demonstration."))
        return items
