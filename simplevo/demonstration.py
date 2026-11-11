"""One recorded demonstration D_tau and the read-only tools that browse it.

The annotator reads the recording through ``demo_search``, ``demo_inspect`` and ``demo_actions`` to write the task
memory; the acting model may revisit it with the same tools. They never move the robot or advance the environment,
and they show only the cameras the live agent has. Motion is reported relative to the recording's own pose at the
start of a range: a reference, never a command.

On disk a demonstration is a folder::

    demo.json            {"instruction": ..., "arms": [...], "cameras": [...], "frames": N}
    trajectory.npz       pose_<arm> (N, 7: xyz + quaternion wxyz), gripper_<arm> (N,), optional base (N, 3: x, y, yaw)
    frames/<camera>/<index:06d>.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .content import as_json, encode_png, image, text
from .control import rotation_deg
from .schema import problems, tool
from .task import validate_names

HEADER = "HISTORICAL DEMONSTRATION: a separate recorded episode, not the current scene; no environment step advanced."
MAX_IMAGES = 8
MAX_ACTION_ROWS = 8
MAX_ACTION_SPAN = 600      # frames per demo_actions call
PREVIEW_PX = 384           # longest side of a returned frame
DEMO_TOOLS = ("demo_search", "demo_inspect", "demo_actions")


class DemonstrationError(ValueError):
    """A refused demonstration request; the session continues."""


class Demonstration:
    """A recording: per-frame measured end-effector poses, gripper openings and camera frames.

    ``grippers`` are normalized like the live state (1 open, 0 closed). ``frame_reader(camera, index)`` returns a
    PIL image, so a dataset adapter can decode videos lazily instead of exporting PNG files.
    """

    def __init__(self, instruction: str, poses: dict, grippers: dict, cameras, frame_reader, base=None):
        self.instruction = instruction
        self.arms = tuple(poses)
        self.cameras = tuple(cameras)
        try:
            validate_names(self.arms, "arms")
            validate_names(self.cameras, "cameras")
        except ValueError as error:
            raise DemonstrationError(str(error)) from error
        if set(grippers) != set(self.arms):
            raise DemonstrationError("poses and grippers must describe the same arms")
        self.poses = {arm: np.asarray(poses[arm], dtype=float) for arm in self.arms}
        self.grippers = {arm: np.asarray(grippers[arm], dtype=float) for arm in self.arms}
        self.base = None if base is None else np.asarray(base, dtype=float)
        self.frame_reader = frame_reader
        first = next(iter(self.poses.values()))
        if first.ndim != 2 or first.shape[0] < 1:
            raise DemonstrationError("a demonstration needs at least one frame of poses shaped (N, 7)")
        self.frames = first.shape[0]
        for arm in self.arms:
            if self.poses[arm].shape != (self.frames, 7) or self.grippers[arm].shape != (self.frames,):
                raise DemonstrationError(f"{arm}: poses must be (N, 7) and grippers (N,) with one N for all arms")
            if not np.isfinite(self.poses[arm]).all() or not np.isfinite(self.grippers[arm]).all():
                raise DemonstrationError(f"{arm}: poses and grippers must be finite")
            norms = np.linalg.norm(self.poses[arm][:, 3:], axis=1)
            if not np.all(np.abs(norms - 1) <= 0.02):
                raise DemonstrationError(f"{arm}: quaternions must be unit length")
            if not np.all((self.grippers[arm] >= 0) & (self.grippers[arm] <= 1)):
                raise DemonstrationError(f"{arm}: grippers must lie in [0, 1]")
        if self.base is not None and (self.base.shape != (self.frames, 3) or not np.isfinite(self.base).all()):
            raise DemonstrationError("base must be a finite array shaped (N, 3)")

    @classmethod
    def load(cls, folder: Path) -> "Demonstration":
        folder = Path(folder)
        meta = json.loads((folder / "demo.json").read_text(encoding="utf-8"))
        def read(camera, index):
            path = folder / "frames" / camera / f"{index:06d}.png"
            if not path.is_file():
                raise DemonstrationError(f"recording has no {camera} frame {index}; export all frames for annotation")
            with Image.open(path) as picture:
                return picture.convert("RGB")
        with np.load(folder / "trajectory.npz", allow_pickle=False) as data:
            demo = cls(meta["instruction"], {arm: data[f"pose_{arm}"] for arm in meta["arms"]},
                       {arm: data[f"gripper_{arm}"] for arm in meta["arms"]}, meta["cameras"], read,
                       data["base"] if "base" in data.files else None)
        if demo.frames != meta["frames"]:
            raise DemonstrationError("demo.json frame count does not match trajectory.npz")
        return demo

    def save(self, folder: Path, every: int = 1) -> None:
        """Write the folder layout above, keeping every ``every``-th camera frame and the last one."""
        if type(every) is not int or every < 1:
            raise DemonstrationError("every must be a positive integer")
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        arrays = {f"pose_{arm}": self.poses[arm] for arm in self.arms}
        arrays.update({f"gripper_{arm}": self.grippers[arm] for arm in self.arms})
        if self.base is not None:
            arrays["base"] = self.base
        np.savez_compressed(folder / "trajectory.npz", **arrays)
        (folder / "demo.json").write_text(json.dumps({"instruction": self.instruction, "arms": list(self.arms),
                                                      "cameras": list(self.cameras), "frames": self.frames}, indent=1), encoding="utf-8", newline="\n")
        for camera in self.cameras:
            (folder / "frames" / camera).mkdir(parents=True, exist_ok=True)
            for index in sorted({*range(0, self.frames, every), self.frames - 1}):
                self.frame_reader(camera, index).save(folder / "frames" / camera / f"{index:06d}.png")


# ----------------------------------------------------------------------------- kinematic segmentation

def _median(values: np.ndarray, size: int) -> np.ndarray:
    if size <= 1 or len(values) < size:
        return values
    padded = np.pad(values, size // 2, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, size), axis=1)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of True as (start, end) with ``end`` exclusive."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(int), [0]))))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def _persistent(mask: np.ndarray, persist: int, gap: int) -> list[tuple[int, int]]:
    """Runs of True after closing gaps of at most ``gap`` frames and dropping runs shorter than ``persist``."""
    merged = []
    for start, end in _runs(mask):
        if merged and start - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return [(start, end) for start, end in merged if end - start >= persist]


def segment(demo: Demonstration, *, moving_m: float = 0.0006, moving_rad: float = 0.005, smooth: int = 7,
            persist: int = 6, gap: int = 30) -> dict:
    """Motion bouts per arm (and base) and gripper events.

    A frame counts as moving when the smoothed per-frame translation exceeds ``moving_m`` or the rotation exceeds
    ``moving_rad``; the thresholds are per recorded frame, so scale them with the recording rate. A gripper event
    is a persistent crossing of the half-open point.
    """
    bouts, events = [], []
    for arm in demo.arms:
        pose, gripper = demo.poses[arm], demo.grippers[arm]
        moved = np.linalg.norm(np.diff(pose[:, :3], axis=0), axis=1)
        dots = np.abs(np.einsum("ij,ij->i", pose[1:, 3:], pose[:-1, 3:]))
        turned = 2 * np.arccos(np.clip(dots / (np.linalg.norm(pose[1:, 3:], axis=1) *
                                               np.linalg.norm(pose[:-1, 3:], axis=1)), 0.0, 1.0))
        score = _median(np.maximum(moved / moving_m, turned / moving_rad), smooth)
        for start, end in _persistent(score >= 1, persist, gap):
            end = min(end, demo.frames - 1)
            bouts.append({"part": arm, "start": start, "end": end,
                          "displacement_m": round(float(np.linalg.norm(pose[end, :3] - pose[start, :3])), 4),
                          "path_m": round(float(moved[start:end].sum()), 4),
                          "rotation_deg": round(rotation_deg(pose[start, 3:], pose[end, 3:]), 1),
                          "gripper_start": round(float(gripper[start]), 3),
                          "gripper_end": round(float(gripper[end]), 3)})
        closed = _median(gripper, smooth) < 0.5
        state, run = bool(closed[0]), 0
        for index in range(1, demo.frames):
            # A change must persist for ``persist`` frames to count; it is dated at its first frame.
            run = run + 1 if closed[index] != state else 0
            if run >= persist:
                state, first, run = not state, index - run + 1, 0
                events.append({"arm": arm, "frame": first, "event": "close" if state else "open",
                               "opening_after": round(float(np.median(gripper[first:first + persist])), 3)})
    if demo.base is not None:
        moved = np.linalg.norm(np.diff(demo.base[:, :2], axis=0), axis=1)
        turned = np.abs(np.diff(np.unwrap(demo.base[:, 2])))
        score = _median(np.maximum(moved / (2.5 * moving_m), turned / moving_rad), smooth)
        for start, end in _persistent(score >= 1, persist, gap):
            end = min(end, demo.frames - 1)
            bouts.append({"part": "base", "start": start, "end": end,
                          "displacement_m": round(float(np.linalg.norm(demo.base[end, :2] - demo.base[start, :2])), 4),
                          "yaw_change_rad": round(float(np.unwrap(demo.base[:, 2])[end] -
                                                        np.unwrap(demo.base[:, 2])[start]), 3)})
    return {"frames": demo.frames, "bouts": sorted(bouts, key=lambda bout: (bout["start"], bout["part"])),
            "gripper_events": sorted(events, key=lambda event: (event["frame"], event["arm"]))}


# ----------------------------------------------------------------------------- the demonstration tools

def tool_specs(demo: Demonstration) -> list[dict]:
    frame = {"type": "integer", "minimum": 0, "maximum": demo.frames - 1}
    region = {"anyOf": [{"type": "null"}, {"type": "array", "minItems": 4, "maxItems": 4,
                                           "items": {"type": "number", "minimum": 0, "maximum": 1}}]}
    return [
        tool("demo_search", (
            "List the recorded demonstration's motion bouts (which arm or base moved, over which frames, how far) "
            "and its gripper events (close/open with the measured opening). " + HEADER), {}),
        tool("demo_inspect", (
            f"Return up to {MAX_IMAGES} frames of the recorded demonstration from one camera, optionally cropped "
            "to a normalized region [x0, y0, x1, y1]. " + HEADER),
            {"frames": {"type": "array", "items": frame, "minItems": 1, "maxItems": MAX_IMAGES, "uniqueItems": True},
             "camera": {"type": "string", "enum": list(demo.cameras)}, "region": region}),
        tool("demo_actions", (
            f"Measured motion of the recorded demonstration over a frame range of at most {MAX_ACTION_SPAN} frames, "
            f"sampled at up to {MAX_ACTION_ROWS} rows and reported relative to the first frame of the range "
            "(translation in metres, rotation in degrees, gripper opening). A reference, never a command for the "
            "current scene. " + HEADER),
            {"start": frame, "end": frame, "rows": {"type": "integer", "minimum": 2, "maximum": MAX_ACTION_ROWS}}),
    ]


class DemoTools:
    def __init__(self, demo: Demonstration):
        self.demo = demo
        self.specs = {spec["name"]: spec["inputSchema"] for spec in tool_specs(demo)}
        self._segments = None

    def call(self, name: str, arguments: dict) -> list[dict]:
        if name not in self.specs:
            raise DemonstrationError(f"unknown demonstration tool {name}")
        found = problems(arguments, self.specs[name])
        if found:
            raise DemonstrationError("; ".join(found[:5]))
        return [text(HEADER), *getattr(self, "_" + name)(arguments)]

    def _demo_search(self, _):
        if self._segments is None:
            self._segments = segment(self.demo)
        return [as_json(self._segments)]

    def _demo_inspect(self, arguments):
        content = []
        for index in arguments["frames"]:
            picture = self.demo.frame_reader(arguments["camera"], index).convert("RGB")
            if arguments["region"] is not None:
                x0, y0, x1, y1 = arguments["region"]
                if not (x0 < x1 and y0 < y1):
                    raise DemonstrationError("region needs x0 < x1 and y0 < y1")
                left, top = int(x0 * picture.width), int(y0 * picture.height)
                picture = picture.crop((left, top, max(left + 1, round(x1 * picture.width)),
                                        max(top + 1, round(y1 * picture.height))))
            scale = min(2.0, PREVIEW_PX / max(picture.size))
            if scale != 1:
                picture = picture.resize((max(1, round(picture.width * scale)), max(1, round(picture.height * scale))),
                                         Image.Resampling.LANCZOS)
            content += [text(f"Demonstration frame {index}; {arguments['camera']} camera"
                             + (f"; region {arguments['region']}" if arguments["region"] else "")),
                        image(encode_png(picture))]
        return content

    def _demo_actions(self, arguments):
        start, end = arguments["start"], arguments["end"]
        if not start < end or end - start > MAX_ACTION_SPAN:
            raise DemonstrationError(f"need start < end and a span of at most {MAX_ACTION_SPAN} frames")
        rows = []
        for index in sorted({int(round(value)) for value in np.linspace(start, end, arguments["rows"])}):
            row = {"frame": index}
            for arm in self.demo.arms:
                pose, origin = self.demo.poses[arm][index], self.demo.poses[arm][start]
                row[arm] = {"delta_xyz_m": [round(float(v), 4) for v in pose[:3] - origin[:3]],
                            "rotation_deg": round(rotation_deg(origin[3:], pose[3:]), 1),
                            "gripper": round(float(self.demo.grippers[arm][index]), 3)}
            if self.demo.base is not None:
                delta = self.demo.base[index] - self.demo.base[start]
                row["base"] = {"delta_xy_m": [round(float(v), 4) for v in delta[:2]],
                               "delta_yaw_rad": round(float(delta[2]), 3)}
            rows.append(row)
        return [as_json({"relative_to_frame": start, "rows": rows})]
