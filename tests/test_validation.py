"""Reject malformed inputs before motions, file reads or annotation acceptance."""
import json

import numpy as np
import pytest
from PIL import Image

from simplevo import Demonstration, Embodiment, TaskMemory
from simplevo.annotate import submit_spec, validate
from simplevo.control import Controller, unit
from simplevo.demonstration import DemonstrationError
from simplevo.envs.toy import IDENTITY, ToyTabletop
from simplevo.schema import problems


@pytest.mark.parametrize("invalid", [
    {"steps": 2, "motion_steps": 3},
    {"left": {"xyz_m": [0, 0, 0.2], "quaternion_wxyz": [0, 0, 0, 0], "gripper": 1}},
])
def test_entire_waypoint_batch_is_validated_before_any_step(invalid):
    env = ToyTabletop(0)
    controller = Controller(env)
    waypoint = {"left": None, "right": None, "steps": 2}
    with pytest.raises(ValueError):
        controller.execute({"waypoints": [waypoint, {**waypoint, **invalid}]})
    assert env.steps_used == 0 and controller.waypoints == 0


@pytest.mark.parametrize("quaternion", [[0, 0, 0, 0], [1, 2, 3], [float("nan"), 0, 0, 0]])
def test_invalid_measured_quaternion_is_refused(quaternion):
    with pytest.raises(ValueError, match="quaternion"):
        unit(quaternion)


@pytest.mark.parametrize("options", [
    {"cameras": ("../outside",)}, {"arms": ("left", "left")}, {"arms": ("base",)},
    {"max_steps": 0}, {"max_waypoints": -1}, {"max_steps": True}, {"has_base": "false"},
])
def test_invalid_embodiment_is_refused(options):
    with pytest.raises(ValueError):
        Embodiment("robot", **options)


def demo(frames=10):
    return Demonstration("move", {"left": np.tile([0, 0, 0.2, *IDENTITY], (frames, 1))},
                         {"left": np.ones(frames)}, ["overview"], lambda camera, index: Image.new("RGB", (8, 8)))


@pytest.mark.parametrize("stages", [
    [(0, 2), (8, 10)], [(0, 7), (5, 8), (3, 10)], [(1, 10)], [(0, 9)],
])
def test_annotation_requires_ordered_coverage_without_gaps(stages):
    recording = demo()
    annotation = {
        "stages": [{"start": start, "end": end, "label": "move", "arms": "left", "what_changes": "moves",
                    "certainty": "seen"} for start, end in stages],
        "keyframes": [{"frame": index, "camera": "overview", "label": "frame"} for index in (0, 5, 9)],
        "arm_roles": "left", "observed_sequence": "move", "observed_outcome": "moved",
        "evidence_limits": "RGB", "reviewed": "all",
    }
    assert validate(annotation, recording, submit_spec(recording))


@pytest.mark.parametrize("frames", [0])
def test_empty_demonstration_is_refused(frames):
    with pytest.raises(DemonstrationError):
        demo(frames)


@pytest.mark.parametrize("change", ["nan_pose", "zero_quaternion", "bad_gripper", "bad_base"])
def test_invalid_demonstration_arrays_are_refused(change):
    recording = demo()
    base = None
    if change == "nan_pose":
        recording.poses["left"][0, 0] = np.nan
    elif change == "zero_quaternion":
        recording.poses["left"][0, 3:] = 0
    elif change == "bad_gripper":
        recording.grippers["left"][0] = 2
    else:
        base = np.zeros((2, 3))
    with pytest.raises(DemonstrationError):
        Demonstration(recording.instruction, recording.poses, recording.grippers, recording.cameras,
                      recording.frame_reader, base)


@pytest.mark.parametrize("name", ["../outside.png", "/outside.png"])
def test_task_memory_cannot_read_keyframes_outside_its_folder(tmp_path, name):
    (tmp_path / "memory").mkdir()
    (tmp_path / "outside.png").write_bytes(b"private")
    (tmp_path / "memory/task_memory.json").write_text(json.dumps({
        "version": 1, "text": "memory", "keyframes": [{"file": name, "label": "frame"}],
    }))
    with pytest.raises(ValueError, match="keyframe|folder"):
        TaskMemory.load(tmp_path / "memory")


def test_task_memory_cannot_follow_a_keyframe_symlink_outside(tmp_path):
    memory = tmp_path / "memory"
    memory.mkdir()
    (tmp_path / "outside.png").write_bytes(b"private")
    (memory / "frame.png").symlink_to(tmp_path / "outside.png")
    (memory / "task_memory.json").write_text(json.dumps({
        "version": 1, "text": "memory", "keyframes": [{"file": "frame.png", "label": "frame"}],
    }))
    with pytest.raises(ValueError, match="keyframe"):
        TaskMemory.load(memory)


def test_sparse_demo_reports_unavailable_frame(tmp_path):
    recording = demo()
    recording.save(tmp_path / "demo", every=5)
    loaded = Demonstration.load(tmp_path / "demo")
    with pytest.raises(DemonstrationError, match="no overview frame 1"):
        loaded.frame_reader("overview", 1)


def test_utf8_task_memory_round_trip(tmp_path):
    TaskMemory("夹取物体；确认抓稳。🤖").save(tmp_path)
    assert TaskMemory.load(tmp_path).text == "夹取物体；确认抓稳。🤖"


def test_oversized_numeric_arguments_are_refused_without_overflow():
    assert problems(10 ** 1000, {"type": "number"})
