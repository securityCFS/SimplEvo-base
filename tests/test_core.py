"""Schema checks, the waypoint controller, the demonstration tools and the annotator."""
import json

import numpy as np
from PIL import Image
import pytest

from simplevo.annotate import annotate, render, submit_spec, validate
from simplevo.backends import ScriptedSession
from simplevo.control import Controller, slerp, tool_specs
from simplevo.demonstration import DemoTools, Demonstration, DemonstrationError, segment
from simplevo.envs.toy import EMBODIMENT, IDENTITY, ToyTabletop
from simplevo.schema import problems
from simplevo.task_memory import TaskMemory


def make_demo(frames=200):
    """Left arm: reach (frames 10-60), close at 80, carry (100-160), open at 180. Right arm: still."""
    left = np.tile([0.0, 0.2, 0.2, 1, 0, 0, 0], (frames, 1)).astype(float)
    left[10:60, 0] = np.linspace(0, 0.2, 50)
    left[60:, 0] = 0.2
    left[100:160, 1] = np.linspace(0.2, 0.4, 60)
    left[160:, 1] = 0.4
    gripper = np.ones(frames)
    gripper[80:180] = 0.1
    right = np.tile([0.3, 0.2, 0.2, 1, 0, 0, 0], (frames, 1)).astype(float)

    def frame(camera, index):
        return Image.new("RGB", (64, 48), (index % 256, 80 if camera == "overview" else 160, 0))
    return Demonstration("Move the block.", {"left": left, "right": right},
                         {"left": gripper, "right": np.ones(frames)}, EMBODIMENT.cameras, frame)


def annotation(frames=200):
    return {"stages": [{"label": "reach", "start": 0, "end": 80, "arms": "left", "what_changes": "left arm reaches",
                        "certainty": "seen"},
                       {"label": "carry and release", "start": 80, "end": frames, "arms": "left",
                        "what_changes": "block moves to the goal", "certainty": "seen"}],
            "keyframes": [{"frame": 0, "camera": "overview", "label": "start"},
                          {"frame": 90, "camera": "left_wrist", "label": "grasp"},
                          {"frame": frames - 1, "camera": "overview", "label": "end state"}],
            "arm_roles": "left arm only", "observed_sequence": "reach, close, carry, open",
            "observed_outcome": "block at the goal, gripper open", "evidence_limits": "none", "reviewed": "all"}


def test_schema_reports_each_problem():
    schema = tool_specs(EMBODIMENT)[0]["inputSchema"]
    arm = {"xyz_m": [0, 0, 0.1], "quaternion_wxyz": IDENTITY, "gripper": 1}
    good = {"waypoints": [{"left": arm, "right": None, "steps": 8}], "cameras": ["overview"], "reason": "test"}
    assert problems(good, schema) == []
    assert problems({**good, "extra": 1}, schema) == ["$: unknown field 'extra'"]
    assert problems({**good, "cameras": ["top"]}, schema)
    assert problems({**good, "waypoints": [{"left": arm, "right": None, "steps": 65}]}, schema)
    assert problems({**good, "waypoints": [{"left": {**arm, "xyz_m": [0, 0]}, "right": None, "steps": 8}]}, schema)
    assert problems({**good, "waypoints": [{"left": arm, "steps": 8}]}, schema) == ["$.waypoints[0]: missing 'right'"]


def test_waypoint_ramps_then_holds_and_null_arm_stays():
    env = ToyTabletop(0)
    controller = Controller(env)
    start = env.state()
    waypoint = {"left": {"xyz_m": [-0.25, 0.05, 0.05], "quaternion_wxyz": IDENTITY, "gripper": 0}, "right": None,
                "steps": 10, "motion_steps": 5}
    actions, _ = controller.waypoint_actions(waypoint)
    heights = [action["left"]["xyz_m"][2] for action in actions]
    assert len(actions) == 10 and heights[0] == pytest.approx(0.21) and heights[4:] == [pytest.approx(0.05)] * 6
    assert all(action["left"]["gripper"] == 0 for action in actions)           # gripper applies at once
    assert all(action["right"] == start["right"] for action in actions)        # a null arm holds
    summary = controller.execute({"waypoints": [waypoint]})
    assert summary[0]["steps_executed"] == 10 and summary[0]["position_error_m"]["left"] < 1e-6
    assert controller.grippers == {"left": 0.0, "right": 1.0}
    with pytest.raises(ValueError):
        controller.waypoint_actions({**waypoint, "motion_steps": 11})
    with pytest.raises(ValueError):
        controller.waypoint_actions({**waypoint, "left": {**waypoint["left"], "quaternion_wxyz": [2, 0, 0, 0]}})


def test_slerp_takes_the_short_arc():
    half = slerp([1, 0, 0, 0], [0, 0, 0, -1], 0.5)
    assert np.allclose(np.abs(half), [np.sqrt(0.5), 0, 0, np.sqrt(0.5)])


def test_segmentation_finds_bouts_and_gripper_events():
    found = segment(make_demo())
    bouts = [bout for bout in found["bouts"] if bout["part"] == "left"]
    assert len(bouts) == 2 and all(bout["part"] == "left" for bout in found["bouts"])
    assert abs(bouts[0]["start"] - 10) <= 4 and abs(bouts[0]["end"] - 60) <= 4
    assert bouts[0]["displacement_m"] == pytest.approx(0.2, abs=0.02)
    assert [(e["event"], e["frame"]) for e in found["gripper_events"]] == [("close", 80), ("open", 180)]


def test_demo_tools_return_frames_and_relative_motion():
    tools = DemoTools(make_demo())
    shown = tools.call("demo_inspect", {"frames": [0, 199], "camera": "overview", "region": [0, 0, 0.5, 0.5]})
    assert [item["type"] for item in shown] == ["text", "text", "image", "text", "image"]
    table = json.loads(tools.call("demo_actions", {"start": 10, "end": 60, "rows": 3})[1]["text"])
    assert table["rows"][-1]["left"]["delta_xyz_m"][0] == pytest.approx(0.2, abs=0.01)
    assert table["rows"][-1]["right"]["delta_xyz_m"] == [0, 0, 0]
    with pytest.raises(DemonstrationError):
        tools.call("demo_inspect", {"frames": [500], "camera": "overview", "region": None})
    with pytest.raises(DemonstrationError):
        tools.call("demo_actions", {"start": 60, "end": 10, "rows": 3})


def test_annotation_must_cover_the_whole_recording():
    demo = make_demo()
    spec = submit_spec(demo)
    assert validate(annotation(), demo, spec) == []
    short = annotation()
    short["stages"] = short["stages"][:1]
    assert "cover" in validate(short, demo, spec)[0]
    twice = annotation()
    twice["keyframes"][1] = twice["keyframes"][0]
    assert validate(twice, demo, spec) == ["duplicate keyframes"]


def test_annotator_session_writes_a_task_memory(tmp_path):
    demo = make_demo()

    def annotator(inputs, call):
        assert "annotating one recorded robot demonstration" in inputs[0]["text"]
        assert json.loads(call("demo_search", {}).content[1]["text"])["frames"] == 200
        rejected = call("submit_icl", {**annotation(), "stages": annotation()["stages"][:1]})
        assert not rejected.success
        assert call("submit_icl", annotation()).success
        return "done"

    memory = annotate(ScriptedSession(annotator), demo, EMBODIMENT, tmp_path / "memory")
    assert "Observed outcome: block at the goal" in memory.text and "frames 80-200 (left; seen)" in memory.text
    loaded = TaskMemory.load(tmp_path / "memory")
    assert loaded.text == memory.text and len(loaded.keyframes) == 3
    assert [item["type"] for item in loaded.inputs()] == ["text", "text", "image", "text", "image", "text", "image",
                                                          "text"]
    assert len(loaded.query({"view": "frames", "frame_indices": [2]})) == 4
    with pytest.raises(ValueError):
        loaded.query({"view": "text", "frame_indices": [0]})
    assert render(annotation(), demo).text == memory.text
    with pytest.raises(DemonstrationError):
        annotate(ScriptedSession(annotator), demo, EMBODIMENT, tmp_path / "memory")


def test_demonstration_round_trips_through_its_folder(tmp_path):
    demo = make_demo()
    demo.save(tmp_path / "demo", every=50)
    loaded = Demonstration.load(tmp_path / "demo")
    assert loaded.frames == 200 and loaded.instruction == demo.instruction
    assert np.allclose(loaded.poses["left"], demo.poses["left"])
    assert loaded.frame_reader("overview", 199).size == (64, 48)
