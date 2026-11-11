"""Regression checks for task boundaries and immutable episode evidence."""
import json
from dataclasses import replace

import pytest

from simplevo import AdaptationChain, adapt, deploy, validate_fixed
from simplevo.backends import ScriptedSession
from simplevo.envs.toy import TASK, ToyModel, ToyTabletop
from simplevo.working_memory import MemoryRefusal, MemoryTools


def scripted(_):
    return ScriptedSession(ToyModel())


def test_archives_and_views_do_not_share_writable_files(tmp_path):
    chain = AdaptationChain(tmp_path / "chain", task=TASK.name, carry="lessons+outcomes+frames")
    adapt(chain, [0], ToyTabletop, scripted, TASK, tmp_path / "train")
    chain.build_view(2, tmp_path / "view")
    relative = "policy/initial_input/state.json"
    archive = chain.committed / "episodes/001" / relative
    original = archive.read_bytes()
    (tmp_path / "train/001" / relative).write_text("changed original")
    assert archive.read_bytes() == original
    (tmp_path / "view/episodes/001" / relative).write_text("changed view")
    assert archive.read_bytes() == original


def test_stale_view_cannot_commit_over_changed_chain(tmp_path):
    chain = AdaptationChain(tmp_path / "chain", task=TASK.name)
    episode = tmp_path / "episode"
    chain.build_view(1, episode / "memory_view")
    (episode / "policy").mkdir()
    (chain.committed / "working_memory.md").write_text("changed after view was built")
    before = chain.state_hash()
    with pytest.raises(MemoryRefusal, match="parent|changed|stale"):
        chain.commit(1, episode=episode, staging=episode / "memory_staging",
                     attempt={"success": False, "score": 0, "termination": "model_finished"}, arms=TASK.embodiment.arms)
    assert chain.state_hash() == before and chain.commits() == []


def test_oversized_staged_notes_do_not_partially_commit(tmp_path):
    chain = AdaptationChain(tmp_path / "chain", task=TASK.name, notes_limit=10)
    episode = tmp_path / "episode"
    chain.build_view(1, episode / "memory_view")
    (episode / "policy").mkdir()
    staging = episode / "memory_staging"
    (staging / "versions").mkdir(parents=True)
    (staging / "versions/001.md").write_text("too long")
    (staging / "working_memory.md").write_text("x" * 11)
    before = chain.state_hash()
    with pytest.raises(MemoryRefusal, match="limit"):
        chain.commit(1, episode=episode, staging=staging, attempt={}, arms=TASK.embodiment.arms)
    assert chain.state_hash() == before
    assert not (chain.committed / "episodes/001").exists()
    assert chain.commits() == []


@pytest.mark.parametrize("phase", ["adapt", "deploy"])
def test_wrong_task_memory_is_refused_before_opening_environment(tmp_path, phase):
    chain = AdaptationChain(tmp_path / "chain", task="another task")
    chain.fix(tmp_path / "fixed")
    opened = []

    def factory(scene):
        opened.append(scene)
        return ToyTabletop(scene)

    with pytest.raises(ValueError, match="task"):
        if phase == "adapt":
            adapt(chain, [0], factory, scripted, TASK, tmp_path / "episodes")
        else:
            deploy(tmp_path / "fixed", [0], factory, scripted, TASK, tmp_path / "episodes")
    assert opened == []


def test_changed_carry_setting_is_not_silently_ignored(tmp_path):
    AdaptationChain(tmp_path / "chain", task=TASK.name, carry="lessons")
    assert AdaptationChain(tmp_path / "chain").carry == "lessons"
    with pytest.raises(MemoryRefusal, match="carry"):
        AdaptationChain(tmp_path / "chain", task=TASK.name, carry="none")


def test_fixed_seal_cannot_claim_another_task(tmp_path):
    chain = AdaptationChain(tmp_path / "chain", task=TASK.name)
    chain.fix(tmp_path / "fixed")
    path = tmp_path / "fixed/fixed_memory.json"
    seal = json.loads(path.read_text())
    seal["task"] = "another task"
    path.write_text(json.dumps(seal))
    with pytest.raises(ValueError, match="task"):
        validate_fixed(tmp_path / "fixed")


@pytest.mark.parametrize("operation", ["resume", "seal", "view"])
def test_incomplete_commit_is_not_used(tmp_path, operation):
    chain = AdaptationChain(tmp_path / "chain", task=TASK.name)
    (chain.committed / "episodes/001").mkdir()
    with pytest.raises(MemoryRefusal, match="incomplete"):
        if operation == "resume":
            AdaptationChain(chain.root)
        elif operation == "seal":
            chain.fix(tmp_path / "fixed")
        else:
            chain.build_view(1, tmp_path / "view")


def test_modified_archive_is_detected_on_resume(tmp_path):
    chain = AdaptationChain(tmp_path / "chain", task=TASK.name)
    adapt(chain, [0], ToyTabletop, scripted, TASK, tmp_path / "train")
    (chain.committed / "episodes/001/policy/initial_input/state.json").write_text("changed")
    with pytest.raises(MemoryRefusal, match="evidence changed"):
        AdaptationChain(chain.root)


def test_rejected_actions_do_not_break_commit_or_record_false_gripper_events(tmp_path):
    def model(inputs, call):
        if not inputs[0]["text"].startswith("The episode has ended."):
            assert not call("set_gripper", {
                "side": "left", "opening": "closed", "steps": 1, "cameras": ["overview"],
            }).success
            assert not call("execute_eef", {
                "waypoints": ["bad waypoint"], "cameras": ["overview"], "reason": "invalid",
            }).success
        return "done"

    chain = AdaptationChain(tmp_path / "chain", task=TASK.name)
    adapt(chain, [0], ToyTabletop, lambda _: ScriptedSession(model), TASK, tmp_path / "train")
    attempt = json.loads((chain.committed / "attempts.jsonl").read_text())
    assert attempt["key_calls"]["gripper_close"] == []
    assert attempt["key_calls"]["gripper_open"] == []


def test_environment_embodiment_must_match_task_before_model_start(tmp_path):
    task = replace(TASK, embodiment=replace(TASK.embodiment, cameras=("different",)))
    started = []

    class Session(ScriptedSession):
        def start(self, tools, instructions=None):
            started.append(True)
            return super().start(tools, instructions)

    with pytest.raises(ValueError, match="embodiment"):
        deploy(None, [0], ToyTabletop, lambda _: Session(ToyModel()), task, tmp_path / "run")
    assert started == []


@pytest.mark.parametrize("carry", ["none", "lessons", "lessons+outcomes", "lessons+outcomes+frames"])
def test_sealing_preserves_carry_visibility(tmp_path, carry):
    chain = AdaptationChain(tmp_path / "chain", task=TASK.name, carry=carry)
    adapt(chain, [0], ToyTabletop, scripted, TASK, tmp_path / "train")
    chain.fix(tmp_path / "fixed")
    validate_fixed(tmp_path / "fixed")
    tools = MemoryTools(tmp_path / "fixed", tmp_path / "policy", tmp_path / "staging", TASK.embodiment.cameras)
    rows = json.loads(tools.call("history", {
        "view": "attempts", "episode": "current", "start": 0, "limit": 32,
    })[0]["text"])["attempts"]
    assert len(rows) == (1 if carry in ("none", "lessons") else 2)
    assert tools.visible == []
