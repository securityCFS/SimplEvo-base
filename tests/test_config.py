"""Configuration validation, runtime budgets, CLI precedence and packaged defaults."""
import json

import pytest
from test_core import make_demo

from simplevo import AdaptationChain, adapt
from simplevo.backends import ScriptedSession
from simplevo.cli import main
from simplevo.config import load_config, resolve_config
from simplevo.content import decode_png
from simplevo.demonstration import DemonstrationError, DemoTools
from simplevo.envs.toy import TASK, ToyModel, ToyTabletop
from simplevo.working_memory import MemoryRefusal, MemoryTools, sha256


@pytest.mark.parametrize("overrides", [
    {"memroy": {}}, {"memory": {"notes_limt": 100}}, {"memory": {"notes_limit": True}},
    {"memory": {"reflection_calls": 0}}, {"episode": {"retries": -1}},
    {"model": {"rpc_timeout_s": float("nan")}}, {"model": {"turn_timeout_s": 0}},
    {"memory": {"carry": "all"}}, {"annotation": {"max_keyframes": 21}},
    {"demonstration": {"max_action_rows": 1}}, {"episode": {"show_outcome": "false"}},
    {"toy": []}, ["toy"],
])
def test_invalid_settings(overrides):
    with pytest.raises(ValueError):
        resolve_config(overrides)


def test_defaults_are_packaged_and_overrides_are_independent(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text("memory:\n  reflection_calls: 4\nmodel:\n  turn_timeout_s: 12.5\n")
    config = load_config(path)
    assert config["memory"]["reflection_calls"] == 4
    assert config["model"]["turn_timeout_s"] == 12.5
    assert config["memory"]["notes_limit"] == 32768
    config["toy"]["train"] = 999
    assert load_config()["toy"]["train"] == 3
    path.write_text("!!python/object/apply:os.system ['false']")
    with pytest.raises(ValueError, match="invalid YAML"):
        load_config(path)


def test_memory_budgets_change_schema_and_enforcement(tmp_path):
    config = resolve_config({"memory": {"scratch_limit": 4, "reflection_writes": 1, "history_page": 2}})
    chain = AdaptationChain(tmp_path / "chain", task=TASK.name)
    chain.build_view(1, tmp_path / "view")
    memory = MemoryTools(tmp_path / "view", tmp_path / "policy", tmp_path / "staging",
                         TASK.embodiment.cameras, limits=config["memory"])
    assert memory.specs["history"]["properties"]["limit"]["maximum"] == 2
    # Four characters, eight bytes: runtime enforces the UTF-8 budget too.
    with pytest.raises(MemoryRefusal, match="8 bytes"):
        memory.call("write_working", {"content": "éééé"})
    memory.start_reflection()
    memory.call("write_task_notes", {"content": "one", "expected_sha256": sha256(b"")})
    with pytest.raises(MemoryRefusal, match="at most 1"):
        memory.call("write_task_notes", {"content": "two", "expected_sha256": sha256(b"one")})


def test_demo_limits_and_segmentation_are_instance_local():
    demo = make_demo()
    config = resolve_config({"demonstration": {"max_images": 1, "preview_px": 16},
                             "segmentation": {"moving_m": 1.0, "moving_rad": 1.0}})
    tools = DemoTools(demo, limits=config["demonstration"], segmentation=config["segmentation"])
    with pytest.raises(DemonstrationError):
        tools.call("demo_inspect", {"camera": "overview", "frames": [0, 1], "region": None})
    result = tools.call("demo_inspect", {"camera": "overview", "frames": [0], "region": None})
    png = next(item["png"] for item in result if item["type"] == "image")
    assert decode_png(png).size == (16, 12)
    assert tools._demo_search({}) != DemoTools(demo)._demo_search({})
    assert DemoTools(demo).specs["demo_inspect"]["properties"]["frames"]["maxItems"] == 8


def test_reflection_call_budget_and_outcome_reach_runner(tmp_path):
    config = resolve_config({"memory": {"reflection_calls": 1}, "episode": {"show_outcome": False}})
    chain = AdaptationChain(tmp_path / "chain", task=TASK.name)
    records = adapt(chain, [0], ToyTabletop, lambda _: ScriptedSession(ToyModel()), TASK,
                    tmp_path / "train", config=config)
    assert records[0].reflection["calls"] == 1
    policy = tmp_path / "train" / "001" / "policy"
    assert "not shown" in (policy / "reflection_prompt.txt").read_text()
    assert load_config(policy / "config.yaml")["memory"]["reflection_calls"] == 1


def test_cli_yaml_and_flag_precedence(tmp_path, capsys):
    config = tmp_path / "run.yaml"
    config.write_text("toy:\n  train: 2\n  test: 1\nmemory:\n  notes_limit: 4096\n")
    out = tmp_path / "run"
    assert main(["toy", "--config", str(config), "--train", "3", "--out", str(out)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["adaptation"]["episodes"] == 3
    assert result["test_with_fixed_working_memory"]["successes"] == 1
    saved = load_config(out / "train" / "001" / "policy" / "config.yaml")
    assert saved["toy"] == {"train": 3, "test": 1}
    assert saved["memory"]["notes_limit"] == 4096
    assert json.loads((out / "chain" / "chain.json").read_text())["notes_limit"] == 4096


def test_cli_model_overrides_and_timeout_settings(tmp_path, monkeypatch, capsys):
    from dataclasses import asdict

    import simplevo.cli as cli

    settings = tmp_path / "settings.yaml"
    settings.write_text("model:\n  name: yaml-model\n  effort: low\n  rpc_timeout_s: 7\n  turn_timeout_s: 9\n")
    task = tmp_path / "task.json"
    task.write_text(json.dumps(asdict(TASK)))
    observed = {}

    class Session:
        def __init__(self, folder, **kwargs):
            observed.update(kwargs)

        def close(self):
            observed["closed"] = True

    monkeypatch.setattr(cli, "CodexSession", Session)
    monkeypatch.setattr(cli.Demonstration, "load", lambda _: make_demo())
    monkeypatch.setattr(cli, "annotate", lambda *args, **kwargs: observed.update(config=kwargs["config"]))
    main(["annotate", "--config", str(settings), "--task", str(task), "--demo", str(tmp_path),
          "--out", str(tmp_path / "out"), "--model", "cli-model", "--effort", "high"])
    capsys.readouterr()
    assert observed["model"] == "cli-model"
    assert observed["effort"] == "high"
    assert observed["rpc_timeout_s"] == 7 and observed["turn_timeout_s"] == 9
    assert observed["config"]["model"]["name"] == "cli-model"
    assert observed["closed"]
