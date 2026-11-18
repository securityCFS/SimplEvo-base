"""Exercise all public CLI phases through the fake subprocess backend."""
import json
import shlex
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from simplevo import Demonstration, TaskMemory, validate_fixed
from simplevo.cli import load_factory, load_task, main
from simplevo.envs.toy import TASK


def test_annotation_adaptation_sealing_and_deployment_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SIMPLEVO_CODEX_BIN", shlex.join([sys.executable, str(Path(__file__).with_name("fake_codex.py"))]))
    monkeypatch.setenv("FAKE_CODEX_LOG", str(tmp_path / "rpc.jsonl"))
    task = tmp_path / "task.json"
    task.write_text(json.dumps(asdict(TASK)))
    demo = Demonstration(TASK.instruction,
                         {arm: np.tile([0, 0, 0.2, 1, 0, 0, 0], (3, 1)) for arm in TASK.embodiment.arms},
                         {arm: np.ones(3) for arm in TASK.embodiment.arms}, TASK.embodiment.cameras,
                         lambda camera, index: Image.new("RGB", (16, 16)))
    demo.save(tmp_path / "demo")
    submission = {
        "stages": [{"start": 0, "end": 3, "label": "hold", "arms": "both", "what_changes": "nothing",
                    "certainty": "seen"}],
        "keyframes": [{"frame": index, "camera": "overview", "label": "holding"} for index in range(3)],
        "arm_roles": "hold", "observed_sequence": "hold", "observed_outcome": "unchanged",
        "evidence_limits": "a tiny fixture", "reviewed": "all three frames",
    }
    monkeypatch.setenv("FAKE_CODEX_SCRIPT", json.dumps([{
        "calls": [{"tool": "submit_icl", "arguments": submission}], "message": "done",
    }]))
    assert main(["annotate", "--task", str(task), "--demo", str(tmp_path / "demo"),
                 "--out", str(tmp_path / "task-memory")]) == 0
    assert len(TaskMemory.load(tmp_path / "task-memory").keyframes) == 3
    monkeypatch.setenv("FAKE_CODEX_SCRIPT", "[]")
    common = ["--task", str(task), "--env", "simplevo.envs.toy:ToyTabletop", "--scenes", "0",
              "--task-memory", str(tmp_path / "task-memory"), "--demo", str(tmp_path / "demo")]
    assert main(["adapt", *common, "--out", str(tmp_path / "train"), "--chain", str(tmp_path / "chain")]) == 0
    assert main(["fix", "--chain", str(tmp_path / "chain"), "--out", str(tmp_path / "fixed")]) == 0
    seal = validate_fixed(tmp_path / "fixed")
    assert main(["deploy", *common, "--out", str(tmp_path / "test"), "--memory", str(tmp_path / "fixed")]) == 0
    assert validate_fixed(tmp_path / "fixed") == seal
    result = json.loads((tmp_path / "test/001/policy/result.json").read_text())
    assert result["outcome"] is not None and result["reflection"] is None
    assert not (tmp_path / "test/001/policy/reflection.json").exists()
    capsys.readouterr()



def _registered_tools(log: Path) -> list[set]:
    starts = [json.loads(line) for line in log.read_text().splitlines() if '"thread/start"' in line]
    return [{spec["name"] for spec in start["params"]["dynamicTools"]} for start in starts]


@pytest.mark.parametrize("with_task_memory", [True, False])
def test_direct_control_baseline_cli(tmp_path, monkeypatch, capsys, with_task_memory):
    """`baseline`: direct control without working memory, with the task memory (the paper's baseline) or without."""
    log = tmp_path / "rpc.jsonl"
    monkeypatch.setenv("SIMPLEVO_CODEX_BIN", shlex.join([sys.executable, str(Path(__file__).with_name("fake_codex.py"))]))
    monkeypatch.setenv("FAKE_CODEX_LOG", str(log))
    monkeypatch.setenv("FAKE_CODEX_SCRIPT", "[]")
    task = tmp_path / "task.json"
    task.write_text(json.dumps(asdict(TASK)))
    extra = []
    if with_task_memory:
        demo = Demonstration(TASK.instruction,
                             {arm: np.tile([0, 0, 0.2, 1, 0, 0, 0], (3, 1)) for arm in TASK.embodiment.arms},
                             {arm: np.ones(3) for arm in TASK.embodiment.arms}, TASK.embodiment.cameras,
                             lambda camera, index: Image.new("RGB", (16, 16)))
        demo.save(tmp_path / "demo")
        TaskMemory(text="Stages: hold the block.", keyframes=[]).save(tmp_path / "task-memory")
        extra = ["--task-memory", str(tmp_path / "task-memory"), "--demo", str(tmp_path / "demo")]
    assert main(["baseline", "--task", str(task), "--env", "simplevo.envs.toy:ToyTabletop", "--scenes", "0", "1",
                 "--out", str(tmp_path / "baseline"), *extra]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["episodes"] == 2
    for index in ("001", "002"):
        episode = tmp_path / "baseline" / index
        result = json.loads((episode / "policy/result.json").read_text())
        assert result["outcome"] is not None and result["reflection"] is None
        assert not (episode / "memory_view").exists() and not (episode / "memory_staging").exists()
        prompt = (episode / "policy/initial_prompt.txt").read_text()
        assert "read_task_notes" not in prompt and ("query_icl" in prompt) == with_task_memory
    for tools in _registered_tools(log):
        assert {"execute_eef", "set_gripper", "observe"} <= tools
        assert not tools & {"read_task_notes", "write_task_notes", "history", "inspect"}
        assert ("query_icl" in tools) == with_task_memory

@pytest.mark.parametrize("count", ["0", "-1"])
@pytest.mark.parametrize("flag", ["--train", "--test"])
def test_invalid_scene_counts_do_not_create_output(tmp_path, count, flag):
    with pytest.raises(SystemExit) as error:
        main(["toy", "--out", str(tmp_path / "run"), flag, count])
    assert error.value.code == 2 and not (tmp_path / "run").exists()


@pytest.mark.parametrize("name", ["simplevo.envs.toy", ":ToyTabletop", "simplevo.envs.toy:TASK"])
def test_invalid_factory_has_actionable_error(name):
    with pytest.raises(ValueError, match="factory"):
        load_factory(name)


def test_task_file_does_not_coerce_string_booleans(tmp_path):
    path = tmp_path / "task.json"
    task = asdict(TASK)
    task["embodiment"]["has_base"] = "false"
    path.write_text(json.dumps(task))
    with pytest.raises(ValueError, match="boolean"):
        load_task(path)
