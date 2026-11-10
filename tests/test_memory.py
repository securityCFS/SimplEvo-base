"""Working-memory tools, the adaptation chain, the fixed memory, and the whole loop on the toy tabletop."""
import json

import pytest

from simplevo import AdaptationChain, adapt, deploy, run_episode, validate_fixed
from simplevo.backends import ModelSession, ScriptedSession, Turn
from simplevo.chain import fixed_view
from simplevo.cli import main
from simplevo.content import decode_png
from simplevo.envs.toy import IDENTITY, TASK, ToyModel, ToyTabletop
from simplevo.task_memory import TaskMemory
from simplevo.working_memory import (MEMORY_TOOLS, REFLECTION_WRITES, MemoryRefusal, MemoryTools, guidance,
                                     reflection_prompt, sha256)
from simplevo.task import Outcome


def scripted(_):
    return ScriptedSession(ToyModel())


def tools_for(tmp_path, carry="lessons+outcomes", notes_limit=200):
    chain = AdaptationChain(tmp_path / "chain", task="t", carry=carry, notes_limit=notes_limit)
    chain.build_view(1, tmp_path / "view")
    return chain, MemoryTools(tmp_path / "view", tmp_path / "episode" / "policy", tmp_path / "episode" / "staging",
                              TASK.embodiment.cameras)


def read(tools):
    return json.loads(tools.call("read_task_notes", {})[0]["text"])


def test_notes_need_the_current_hash_and_respect_the_budget(tmp_path):
    _, tools = tools_for(tmp_path)
    assert read(tools) == {"sha256": sha256(b""), "notes": ""}
    with pytest.raises(MemoryRefusal, match="expected_sha256"):
        tools.call("write_task_notes", {"content": "x", "expected_sha256": "stale"})
    with pytest.raises(MemoryRefusal):                         # over the chain's budget: refused, never truncated
        tools.call("write_task_notes", {"content": "x" * 201, "expected_sha256": sha256(b"")})
    saved = json.loads(tools.call("write_task_notes", {"content": "lesson", "expected_sha256": sha256(b"")})[0]["text"])
    assert saved["sha256"] == sha256(b"lesson") and read(tools)["notes"] == "lesson"


def test_reflection_allows_two_rewrites_and_closes_the_scratchpad(tmp_path):
    _, tools = tools_for(tmp_path)
    tools.call("write_working", {"content": "plan"})
    assert tools.call("read_working", {})[0]["text"] == "plan"
    tools.start_reflection()
    with pytest.raises(MemoryRefusal):
        tools.call("write_working", {"content": "late"})
    for index in range(REFLECTION_WRITES):
        tools.call("write_task_notes", {"content": f"v{index}", "expected_sha256": read(tools)["sha256"]})
    with pytest.raises(MemoryRefusal, match="at most"):
        tools.call("write_task_notes", {"content": "v9", "expected_sha256": read(tools)["sha256"]})
    assert read(tools)["notes"] == "v1"


def test_prompts_state_what_is_carried_and_what_the_result_was():
    assert "may be empty" in guidance(1, "lessons+outcomes") and "official outcomes" in guidance(1, "lessons+outcomes")
    assert "Nothing from earlier attempts" in guidance(1, "none")
    assert "archived frames" in guidance(2, "lessons+outcomes+frames")
    assert "fixed adaptation snapshot" in guidance(4, "lessons+outcomes", fixed=True)
    assert "one turn, with the official result" not in guidance(4, "lessons+outcomes", fixed=True)
    assert "success=1.0, score=0.75" in reflection_prompt(Outcome(True, 0.75), held=False)
    hidden = reflection_prompt(Outcome(True, 0.75), held=True, show_outcome=False)
    assert "not shown" in hidden and "success=" not in hidden and "held still" in hidden


def test_adaptation_improves_then_the_fixed_memory_is_read_only(tmp_path):
    chain = AdaptationChain(tmp_path / "chain", task=TASK.name)
    records = adapt(chain, [1, 2, 3], ToyTabletop, scripted, TASK, tmp_path / "train")
    assert [record.outcome.success for record in records] == [False, False, True]
    assert records[0].termination == "model_finished" and records[0].held_steps > 0
    assert records[2].termination == "episode_complete" and records[2].reflection["note_writes"] == 1
    assert chain.notes().count("Lesson") == 3 and [c["notes"] for c in chain.commits()] == ["rewritten"] * 3
    attempts = [json.loads(line) for line in (chain.committed / "attempts.jsonl").read_text().splitlines()]
    assert [row["success"] for row in attempts] == [False, False, True]
    assert attempts[2]["key_calls"]["gripper_close"] == [{"call": 3, "arm": "left"}]
    assert attempts[2]["key_calls"]["episode_end_call"] == attempts[2]["key_calls"]["last_call"]
    # the second episode read the first one's lesson and outcome, and no archived frames
    view = tmp_path / "train" / "002" / "memory_view"
    assert "attempt 1" in (view / "working_memory.md").read_text() and not list((view / "episodes").iterdir())
    assert len((view / "attempts.jsonl").read_text().splitlines()) == 1

    seal = chain.fix(tmp_path / "fixed")
    before = (tmp_path / "fixed" / "working_memory.md").read_text()
    tested = deploy(tmp_path / "fixed", [11, 12], ToyTabletop, scripted, TASK, tmp_path / "test")
    assert all(record.outcome.success and record.reflection is None for record in tested)
    assert (tmp_path / "fixed" / "working_memory.md").read_text() == before
    assert validate_fixed(tmp_path / "fixed")["snapshot_id"] == seal["snapshot_id"] and chain.next_position() == 4
    assert not any(record.outcome.success for record in
                   deploy(None, [11, 12], ToyTabletop, scripted, TASK, tmp_path / "zero"))
    (tmp_path / "fixed" / "working_memory.md").write_text(before + "\ntampered")
    with pytest.raises(ValueError, match="changed"):
        fixed_view(tmp_path / "fixed", tmp_path / "copy")


def test_chain_commits_once_in_order_and_filters_what_is_carried(tmp_path):
    chain = AdaptationChain(tmp_path / "chain", task=TASK.name, carry="lessons+outcomes+frames")
    adapt(chain, [1], ToyTabletop, scripted, TASK, tmp_path / "train")
    with pytest.raises(MemoryRefusal):
        chain.build_view(1, tmp_path / "again")
    with pytest.raises(MemoryRefusal):
        chain.commit(1, episode=tmp_path / "train" / "001", staging=tmp_path / "none", attempt={},
                     arms=TASK.embodiment.arms)
    chain.build_view(2, tmp_path / "view")
    tools = MemoryTools(tmp_path / "view", tmp_path / "live" / "policy", tmp_path / "live" / "staging",
                        TASK.embodiment.cameras)
    events = json.loads(tools.call("history", {"view": "events", "episode": 1, "start": 0, "limit": 8})[0]["text"])
    assert [event["tool"] for event in events["events"]] == ["read_task_notes", "execute_eef", "set_gripper",
                                                             "execute_eef"]
    frames = tools.call("inspect", {"question": "where is the block", "views": [
        {"episode": 1, "call": "initial", "camera": "overview", "region": [0, 0, 0.5, 0.5]}]})
    assert "ARCHIVED" in frames[0]["text"] and decode_png(frames[1]["png"]).size == (320, 192)
    states = json.loads(tools.call("read_state_history", {"episode": 1, "start": 0, "limit": 8})[0]["text"])
    assert states["states"][0]["call"] == "initial" and states["states"][0]["state"]["env_steps_used"] == 0
    with pytest.raises(MemoryRefusal):
        tools.call("history", {"view": "events", "episode": 5, "start": 0, "limit": 8})
    assert AdaptationChain(tmp_path / "chain").carry == "lessons+outcomes+frames"
    with pytest.raises(MemoryRefusal):
        AdaptationChain(tmp_path / "chain", task="another task")


def test_episode_records_what_the_model_received_and_refuses_bad_calls(tmp_path):
    memory = TaskMemory("Stages: reach, close, carry, release. End state: block on the pad.", [])
    seen = {}

    def model(inputs, call):
        seen["prompt"] = inputs[0]["text"]
        seen["kinds"] = [item["type"] for item in inputs]
        bad = call("execute_eef", {"waypoints": [], "cameras": ["overview"], "reason": "empty"})
        seen["bad"] = (bad.success, bad.content[0]["text"])
        seen["icl"] = call("query_icl", {"view": "text", "frame_indices": []}).content[0]["text"]
        seen["memory"] = call.tools.isdisjoint(MEMORY_TOOLS)
        good = call("execute_eef", {"cameras": ["overview"], "reason": "wait", "waypoints": [
            {"left": None, "right": {"xyz_m": [0.2, 0.1, 0.2], "quaternion_wxyz": IDENTITY, "gripper": 1},
             "steps": 4}]})
        seen["state"] = json.loads(good.content[1]["text"])["measured_state"]
        return "Nothing more to do."

    record = run_episode(ToyTabletop(3), ScriptedSession(model), TASK, tmp_path / "episode", task_memory=memory)
    assert seen["prompt"].startswith("Complete this single task yourself using the tools: Put the red block")
    assert "query_icl" in seen["prompt"] and "Working memory" not in seen["prompt"] and seen["memory"]
    assert seen["kinds"] == ["text", "text", "text", "text", "text", "image", "text", "image", "text", "image"]
    assert seen["bad"] == (False, "No further action executed. Invalid tool parameters: $.waypoints: needs at least "
                                  "1 items")
    assert "HISTORICAL ICL REFERENCE" in seen["icl"]
    assert seen["state"]["env_steps_used"] == 4 and "block" not in json.dumps(seen["state"])
    assert record.tool_calls == 3 and record.termination == "model_finished" and not record.outcome.success
    assert record.reflection is None and record.env_steps == 400
    call = tmp_path / "episode" / "policy" / "tools" / "call_00003"
    assert json.loads((call / "response_summary.json").read_text()) == {"success": True, "env_steps": 4,
                                                                         "episode_ended": False}
    assert sorted(p.name for p in (call / "observation").iterdir()) == ["overview.png", "state.json"]
    with pytest.raises(RuntimeError, match="already used"):
        run_episode(ToyTabletop(3), ScriptedSession(model), TASK, tmp_path / "episode")


class Flaky(ModelSession):
    """Fails once with a transient error and once with an empty turn before acting."""

    def __init__(self):
        self.turns, self.inputs = 0, []

    def start(self, tools, instructions=None):
        pass

    def is_transient(self, error):
        return "overloaded" in error

    def run_turn(self, inputs, handle):
        self.turns += 1
        self.inputs.append(inputs[0]["text"])
        if self.turns == 1:
            return Turn("failed", [], "server overloaded")
        if self.turns == 2:
            return Turn("completed", [])
        handle("observe", {"cameras": ["overview"]})
        return Turn("completed", ["done"])


def test_a_failed_or_silent_turn_continues_the_same_episode(tmp_path):
    session, waits = Flaky(), []
    record = run_episode(ToyTabletop(1), session, TASK, tmp_path / "episode", retry_wait=waits.append)
    assert waits == [1] and record.tool_calls == 1 and record.messages == ["done"]
    assert "transient service error" in session.inputs[1] and "without an outcome" in session.inputs[2]

    class Broken(Flaky):
        def run_turn(self, inputs, handle):
            return Turn("failed", [], "invalid request")
    with pytest.raises(RuntimeError, match="invalid request"):
        run_episode(ToyTabletop(1), Broken(), TASK, tmp_path / "broken")


def test_toy_command_runs_the_whole_method(tmp_path, capsys):
    assert main(["toy", "--out", str(tmp_path / "run"), "--train", "3", "--test", "2"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["test_without_working_memory"]["successes"] == 0
    assert report["test_with_fixed_working_memory"]["successes"] == 2
    assert report["working_memory"].count("Lesson") == 3
    assert main(["fix", "--chain", str(tmp_path / "run" / "chain"), "--out", str(tmp_path / "again")]) == 0
