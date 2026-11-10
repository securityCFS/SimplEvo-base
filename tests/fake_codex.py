"""A stand-in for `codex app-server --stdio` that speaks the few JSON-RPC messages the backend uses.

FAKE_CODEX_SCRIPT is a JSON list of turns: {"calls": [{"tool": ..., "arguments": ...}], "message": ...,
"status": "completed" | "failed", "error": ...}. Everything received is appended to FAKE_CODEX_LOG.
"""
import json
import os
import sys

turns = iter(json.loads(os.environ.get("FAKE_CODEX_SCRIPT", "[]")))
log = open(os.environ["FAKE_CODEX_LOG"], "a")


def send(message):
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def read():
    line = sys.stdin.readline()
    if not line:
        raise SystemExit(0)
    message = json.loads(line)
    log.write(json.dumps(message) + "\n")
    log.flush()
    return message


number = 0
while True:
    message = read()
    method = message.get("method")
    if method == "initialize":
        send({"id": message["id"], "result": {}})
    elif method == "model/list":
        send({"id": message["id"], "result": {"data": [
            {"model": "gpt-6-astra", "supportedReasoningEfforts": [{"reasoningEffort": "high"}]}]}})
    elif method == "thread/start":
        params = message["params"]
        send({"id": message["id"], "result": {"thread": {"id": "thread-1"}, "model": params["model"],
                                              "reasoningEffort": params["config"]["model_reasoning_effort"]}})
    elif method == "turn/start":
        number += 1
        turn = next(turns, {"calls": [], "message": "done"})
        send({"id": message["id"], "result": {"turn": {"id": f"turn-{number}"}}})
        status = turn.get("status", "completed")
        for index, call in enumerate(turn["calls"]):
            send({"id": 9000 + 10 * number + index, "method": "item/tool/call", "params": {
                "threadId": "thread-1", "callId": f"call-{number}-{index}", **call}})
            reply = read()
            if any('"episode_ended": true' in item.get("text", "") for item in reply["result"]["contentItems"]):
                interrupt = read()                    # the harness ends the turn
                send({"id": interrupt["id"], "result": {}})
                send({"id": 9999, "method": "item/tool/call", "params": {
                    "threadId": "thread-1", "callId": f"late-{number}", "tool": "observe", "arguments": {}}})
                read()                                # the refusal of the late call
                status = "interrupted"
                break
        send({"method": "thread/tokenUsage/updated", "params": {"threadId": "thread-1", "tokenUsage": {
            "total": {"inputTokens": 100 * number, "outputTokens": 10 * number, "totalTokens": 110 * number}}}})
        if status == "completed" and turn.get("message"):
            send({"method": "item/completed", "params": {"threadId": "thread-1", "item": {
                "type": "agentMessage", "text": turn["message"]}}})
        send({"method": "turn/completed", "params": {"threadId": "thread-1", "turn": {
            "id": f"turn-{number}", "status": status, "error": turn.get("error")}}})
