"""The Codex app-server backend against a local stand-in for the server: no model, no network, no login."""
import json
import shlex
import sys
from pathlib import Path

import pytest

from simplevo import run_episode
from simplevo.backends import CodexSession, ToolResult
from simplevo.backends.codex import IMAGE_OUTPUT_GUIDANCE
from simplevo.content import image, text
from simplevo.envs.toy import IDENTITY, TASK, ToyTabletop

FAKE = shlex.join([sys.executable, str(Path(__file__).with_name('fake_codex.py'))])


def start(tmp_path, monkeypatch, turns):
    monkeypatch.setenv("FAKE_CODEX_SCRIPT", json.dumps(turns))
    monkeypatch.setenv("FAKE_CODEX_LOG", str(tmp_path / "server.jsonl"))
    monkeypatch.setenv("SIMPLEVO_CODEX_BIN", FAKE)
    return CodexSession(tmp_path / "session")


def received(tmp_path):
    return [json.loads(line) for line in (tmp_path / "server.jsonl").read_text().splitlines()]


def test_tools_instructions_and_results_use_the_server_format(tmp_path, monkeypatch):
    session = start(tmp_path, monkeypatch, [{"calls": [{"tool": "look", "arguments": json.dumps({"camera": "a"})}],
                                             "message": "seen"}])
    spec = {"type": "function", "name": "look", "description": "Look.", "inputSchema": {"type": "object"}}
    calls = []

    def handle(name, arguments):
        calls.append((name, arguments))
        return ToolResult([text("frame"), image(ToyTabletop(0).render("overview"))])
    try:
        session.start([spec], "Be careful.")
        turn = session.run_turn([text("go"), image(ToyTabletop(0).render("overview"))], handle)
    finally:
        session.close()
    assert (turn.status, turn.messages, calls) == ("completed", ["seen"], [("look", {"camera": "a"})])
    assert session.usage()["totalTokens"] == 110
    log = {message.get("method", "reply"): message for message in received(tmp_path)}
    thread = log["thread/start"]["params"]
    assert thread["baseInstructions"] == "Be careful.\n\n" + IMAGE_OUTPUT_GUIDANCE and thread["dynamicTools"] == [spec]
    assert thread["ephemeral"] and thread["config"]["features.shell_tool"] is False
    assert [item["type"] for item in log["turn/start"]["params"]["input"]] == ["text", "image"]
    result = log["reply"]["result"]
    assert result["success"] and [item["type"] for item in result["contentItems"]] == ["inputText", "inputImage"]
    assert result["contentItems"][1]["imageUrl"].startswith("data:image/png;base64,")


def test_without_instructions_the_image_guidance_goes_into_the_tools(tmp_path, monkeypatch):
    session = start(tmp_path, monkeypatch, [])
    try:
        session.start([{"type": "function", "name": "look", "description": "Look.", "inputSchema": {}}])
    finally:
        session.close()
    thread = next(m for m in received(tmp_path) if m.get("method") == "thread/start")["params"]
    assert "baseInstructions" not in thread and thread["dynamicTools"][0]["description"].endswith(IMAGE_OUTPUT_GUIDANCE)


def test_a_failed_turn_reports_its_error_and_whether_to_retry(tmp_path, monkeypatch):
    session = start(tmp_path, monkeypatch, [{"calls": [], "status": "failed", "error": {"message": "Overloaded"}}])
    try:
        session.start([])
        turn = session.run_turn([text("go")], lambda name, arguments: ToolResult([]))
    finally:
        session.close()
    assert turn.status == "failed" and session.is_transient(turn.error) and not session.is_transient("bad request")


def test_an_unavailable_model_is_refused_at_start(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_CODEX_LOG", str(tmp_path / "server.jsonl"))
    monkeypatch.setenv("SIMPLEVO_CODEX_BIN", FAKE)
    with pytest.raises(RuntimeError, match="not available"):
        CodexSession(tmp_path / "session", model="another-model")


def test_episode_end_interrupts_the_turn_and_later_calls_do_not_run(tmp_path, monkeypatch):
    env = ToyTabletop(1)
    pad = [*env.pad, 0.2]
    carry = {"left": {"xyz_m": pad, "quaternion_wxyz": IDENTITY, "gripper": 1}, "right": None, "steps": 32,
             "motion_steps": 24}
    env.block = list(env.pad)                      # the block already lies on the pad: the first step ends the episode
    session = start(tmp_path, monkeypatch, [{"calls": [
        {"tool": "execute_eef", "arguments": {"waypoints": [carry], "cameras": ["overview"], "reason": "finish"}},
        {"tool": "observe", "arguments": {"cameras": ["overview"]}}], "message": "never sent"}])
    try:
        record = run_episode(env, session, TASK, tmp_path / "episode")
    finally:
        session.close()
    assert record.outcome.success and record.termination == "episode_complete" and record.tool_calls == 1
    assert record.env_steps == 1 and record.messages == []
    replies = [m["result"] for m in received(tmp_path) if "result" in m]
    assert replies[-1] == {"success": False, "contentItems": [
        {"type": "inputText", "text": "The episode has ended; this action was NOT executed."}]}
    assert any(m.get("method") == "turn/interrupt" for m in received(tmp_path))


def test_paginated_model_catalog(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_CODEX_PAGINATE", "1")
    session = start(tmp_path, monkeypatch, [])
    session.close()
    requests = [m["params"] for m in received(tmp_path) if m.get("method") == "model/list"]
    assert [p["cursor"] for p in requests] == [None, "page-2"]


def test_stale_completion_does_not_finish_the_current_turn(tmp_path, monkeypatch):
    session = start(tmp_path, monkeypatch, [{"calls": [], "stale_completion": True, "message": "current"}])
    try:
        session.start([])
        turn = session.run_turn([text("go")], lambda name, arguments: ToolResult([]))
    finally:
        session.close()
    assert turn.messages == ["current"]


def test_server_interrupt_is_reported_as_an_interrupt(tmp_path, monkeypatch):
    session = start(tmp_path, monkeypatch, [{"calls": [], "status": "interrupted"}])
    try:
        session.start([])
        turn = session.run_turn([text("go")], lambda name, arguments: ToolResult([]))
    finally:
        session.close()
    assert turn.status == "interrupted"


def test_close_reaps_server_and_closes_streams(tmp_path, monkeypatch):
    session = start(tmp_path, monkeypatch, [])
    session.close()
    session.close()
    assert session.process.poll() is not None and not session.reader.is_alive()
    assert session.process.stdin.closed and session.process.stdout.closed and session.stderr.closed


def test_startup_failure_closes_log(tmp_path, monkeypatch):
    handles = []
    original = Path.open

    def track(path, *args, **kwargs):
        handle = original(path, *args, **kwargs)
        if path.name == "app_server_stderr.log":
            handles.append(handle)
        return handle

    monkeypatch.setattr(Path, "open", track)
    with pytest.raises(FileNotFoundError):
        CodexSession(tmp_path / "session", executable=str(tmp_path / "does-not-exist"))
    assert len(handles) == 1 and handles[0].closed
