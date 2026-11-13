"""Optional real-physics checks. No Codex process, login or network is used."""
import json

import pytest

pytest.importorskip("pybullet")

from simplevo import run_episode  # noqa: E402
from simplevo.backends import ScriptedSession  # noqa: E402
from simplevo.content import decode_png  # noqa: E402
from simplevo.envs.playground import SETTINGS, TASK, PandaTabletop, reference_policy  # noqa: E402


@pytest.mark.parametrize("scene", [0, 1, 100])
def test_physical_pick_and_place_through_public_tools(tmp_path, scene):
    env = PandaTabletop(scene)
    try:
        start = env.state()
        assert set(start) == {"arm"}
        assert set(start["arm"]) == {"xyz_m", "quaternion_wxyz", "gripper"}
        assert decode_png(env.render("overview")).size == (SETTINGS["width"], SETTINGS["height"])
        result = run_episode(env, ScriptedSession(reference_policy(scene)), TASK, tmp_path / str(scene))
        assert result.outcome.success
        assert result.env_steps <= SETTINGS["step_limit"]
        assert env.step([start]) == 0
        logged = json.loads((tmp_path / str(scene) / "policy" / "initial_input" / "state.json").read_text())
        assert "cube" not in json.dumps(logged)
    finally:
        env.close()
        env.close()


def test_clients_are_independent_and_timeout_is_terminal():
    a, b = PandaTabletop(0), PandaTabletop(1)
    try:
        action = a.state()
        a.step([action] * SETTINGS["step_limit"])
        assert a.ended and not a.outcome.success
        assert not b.ended and b.steps_used == 0
        with pytest.raises(ValueError, match="camera"):
            b.render("missing")
        a.close()
        assert b.state()["arm"]["gripper"] > 0.9
    finally:
        a.close()
        b.close()


def test_cli_adapter_with_fake_model_transport(tmp_path, monkeypatch, capsys):
    """Exercise the actual CLI/backend wiring using a local fake server, never the installed Codex CLI."""
    import shlex
    import sys
    from dataclasses import asdict
    from pathlib import Path

    from simplevo.cli import main

    fake = Path(__file__).with_name("fake_codex.py")
    monkeypatch.setenv("SIMPLEVO_CODEX_BIN", shlex.join([sys.executable, str(fake)]))
    monkeypatch.setenv("FAKE_CODEX_LOG", str(tmp_path / "rpc.jsonl"))
    monkeypatch.setenv("FAKE_CODEX_SCRIPT", "[]")
    task = tmp_path / "task.json"
    task.write_text(json.dumps(asdict(TASK)))
    common = ["--env", "simplevo.envs.playground:make_env", "--task", str(task)]
    main(["adapt", *common, "--scenes", "0", "--chain", str(tmp_path / "chain"), "--out", str(tmp_path / "train")])
    main(["fix", "--chain", str(tmp_path / "chain"), "--out", str(tmp_path / "fixed")])
    main(["deploy", *common, "--scenes", "100", "--memory", str(tmp_path / "fixed"),
          "--out", str(tmp_path / "test")])
    result = json.loads((tmp_path / "test" / "001" / "policy" / "result.json").read_text())
    assert result["outcome"] == {"success": False, "score": 0.0}  # fake server does not perform the task
    assert result["reflection"] is None
    assert (tmp_path / "rpc.jsonl").is_file()
    capsys.readouterr()
