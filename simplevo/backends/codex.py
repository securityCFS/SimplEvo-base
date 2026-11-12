"""Model backend over the Codex app-server (JSON-RPC on stdio), with the harness's tools as dynamic tools.

The conversation is one ephemeral thread without shell, web search, project documents or sub-agents, so the model
can act only through the declared tools. Authentication is whatever the ``codex`` CLI is logged in with (its
``CODEX_HOME``); this module never reads or stores credentials.
"""
from __future__ import annotations

import json
import os
import queue
import shlex
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

from ..config import DEFAULTS
from ..content import data_url
from . import ModelSession, Turn

# Models that issue tool calls from a code cell receive dynamic tool output as newline-delimited strings; this
# tells them how to turn the image lines back into images.
IMAGE_OUTPUT_GUIDANCE = r"""Dynamic tool output in Codex code mode:
The tools declared for this thread return newline-delimited strings, not MCP
objects with a .content field. A result can contain text lines followed by full
data:image/... image URL lines. In the SAME functions.exec call that awaits a
tool, forward each image URL with image(url), and emit the other lines with
text(line), preserving their order. Printing the whole result as text does not
display its images and can truncate their encoded bytes. For example, after
`const result = await tools.TOOL_NAME(arguments);`, use:
for (const line of result.split(/\r?\n/)) {
  if (line.startsWith("data:image/")) image(line);
  else if (line.trim()) text(line);
}
Use the actual declared tool name and arguments. Inspect the emitted images
before deciding the next action. Do not access result.content or print base64.
"""
TRANSIENT = ("capacity", "overloaded", "server_error", "internal error", "502", "503", "504", "timed out",
             "timeout", "stream disconnected", "connection", "temporarily", "429", "too many requests")
TOKEN_KEYS = ("inputTokens", "outputTokens", "cachedInputTokens", "reasoningOutputTokens", "totalTokens")


def _wire(item: dict, result: bool) -> dict:
    """A content item in the app-server's format (tool results and user input use different field names)."""
    if item["type"] == "text":
        return {"type": "inputText" if result else "text", "text": item["text"]}
    url = data_url(item["png"])
    return {"type": "inputImage", "imageUrl": url} if result else {"type": "image", "url": url}


class CodexSession(ModelSession):
    def __init__(self, workdir: Path, *, model: str = DEFAULTS["model"]["name"], effort: str = DEFAULTS["model"]["effort"],
                 executable: str | None = None, rpc_timeout_s: float = DEFAULTS["model"]["rpc_timeout_s"],
                 turn_timeout_s: float | None = DEFAULTS["model"]["turn_timeout_s"]):
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.model, self.effort = model, effort
        self.rpc_timeout_s, self.turn_timeout_s = rpc_timeout_s, turn_timeout_s
        self.messages: queue.Queue = queue.Queue()
        self.deferred: deque = deque()
        self.sequence = 0
        self.thread_id = None
        self.tokens = dict.fromkeys(TOKEN_KEYS, 0)
        self.handled_calls: set[str] = set()
        command = shlex.split(executable or os.environ.get("SIMPLEVO_CODEX_BIN", "codex"))
        self.stderr = (self.workdir / "app_server_stderr.log").open("a", encoding="utf-8", newline="\n")
        self.process = None
        self.reader = None
        try:
            self.process = subprocess.Popen([*command, "app-server"], stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE, stderr=self.stderr, text=True,
                                            encoding="utf-8", bufsize=1)
            self.reader = threading.Thread(target=self._read, daemon=True)
            self.reader.start()
            self.rpc("initialize", {"clientInfo": {"name": "simplevo", "version": "1"},
                                    "capabilities": {"experimentalApi": True}})
            self._send({"method": "initialized", "params": {}})
            cursor, seen, selected = None, set(), None
            while True:
                page = self.rpc("model/list", {"includeHidden": True, "cursor": cursor})
                selected = next((entry for entry in page["data"] if entry["model"] == model), None)
                cursor = page.get("nextCursor")
                if selected is not None or not cursor:
                    break
                if cursor in seen:
                    raise RuntimeError("model/list repeated a pagination cursor")
                seen.add(cursor)
            if selected is None or effort not in [e["reasoningEffort"] for e in selected["supportedReasoningEfforts"]]:
                raise RuntimeError(f"{model}/{effort} is not available to this login")
        except Exception:
            self.close()
            raise

    # ------------------------------------------------------------------ transport

    def _read(self):
        for line in self.process.stdout:
            try:
                self.messages.put(json.loads(line))
            except json.JSONDecodeError:
                self.messages.put({"transport_error": "non-JSON server output"})
        self.messages.put({"transport_error": "app-server stdout closed"})

    def _send(self, message: dict):
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()

    def _receive(self, timeout: float = 1):
        if self.deferred:
            return self.deferred.popleft()
        try:
            message = self.messages.get(timeout=timeout)
        except queue.Empty:
            return None
        if "transport_error" in message:
            raise RuntimeError(message["transport_error"])
        return message

    def rpc(self, method: str, params: dict):
        self.sequence += 1
        request_id = self.sequence
        self._send({"id": request_id, "method": method, "params": params})
        deadline, pending = time.monotonic() + self.rpc_timeout_s, []
        try:
            while time.monotonic() < deadline:
                response = self._receive(1)
                if response is None:
                    continue
                if response.get("id") != request_id or "method" in response:
                    pending.append(response)        # notifications and tool requests stay for the turn loop
                    continue
                if "error" in response:
                    raise RuntimeError(f"{method}: {response['error']}")
                return response["result"]
            raise TimeoutError(method)
        finally:
            self.deferred.extendleft(reversed(pending))

    # ------------------------------------------------------------------ ModelSession

    def start(self, tools, instructions=None):
        params = {"model": self.model, "ephemeral": True, "sandbox": "read-only", "approvalPolicy": "never",
                  "cwd": str(self.workdir.resolve()),
                  "config": {"model_reasoning_effort": self.effort, "project_doc_max_bytes": 0,
                             "features.shell_tool": False, "features.multi_agent": False, "web_search": "disabled"}}
        if instructions is not None:
            params["baseInstructions"] = instructions + "\n\n" + IMAGE_OUTPUT_GUIDANCE
            params["dynamicTools"] = tools
        else:
            params["dynamicTools"] = [dict(spec, description=spec.get("description", "") + "\n\n" +
                                           IMAGE_OUTPUT_GUIDANCE) for spec in tools]
        thread = self.rpc("thread/start", params)
        actual = (thread.get("model"), thread.get("reasoningEffort"))
        if any(value is not None for value in actual) and actual != (self.model, self.effort):
            raise RuntimeError(f"the thread runs {actual}, not {(self.model, self.effort)}")
        self.thread_id = thread["thread"]["id"]

    def run_turn(self, inputs, handle):
        start = self.rpc("turn/start", {"threadId": self.thread_id, "model": self.model, "effort": self.effort,
                                        "input": [_wire(item, result=False) for item in inputs]})
        turn_id, messages = start["turn"]["id"], []
        deadline = None if self.turn_timeout_s is None else time.monotonic() + self.turn_timeout_s
        while deadline is None or time.monotonic() < deadline:
            event = self._receive(1)
            if event is None:
                continue
            method, data = event.get("method"), event.get("params", {})
            if data.get("threadId") not in (None, self.thread_id):
                continue
            event_turn = data.get("turnId", data.get("turn", {}).get("id"))
            if event_turn not in (None, turn_id):
                if "id" in event and method == "item/tool/call":
                    self._send({"id": event["id"], "result": {"success": False, "contentItems": [
                        {"type": "inputText", "text": "This call is from an inactive turn; no action executed."}]}})
                continue
            if method == "thread/tokenUsage/updated":
                self._take_usage(data)
            elif method == "item/tool/call" and "id" in event:
                if data["callId"] in self.handled_calls:
                    raise RuntimeError("duplicate tool call id; no action is replayed")
                self.handled_calls.add(data["callId"])
                arguments = data["arguments"]
                result = handle(data["tool"].split(".")[-1],
                                json.loads(arguments) if isinstance(arguments, str) else arguments)
                self._send({"id": event["id"], "result": {
                    "success": result.success, "contentItems": [_wire(item, result=True) for item in result.content]}})
                if result.end_turn:
                    self._interrupt(turn_id)
                    return Turn("interrupted", messages)
            elif "id" in event and method:
                self._send({"id": event["id"], "error": {"code": -32601,
                                                         "message": "only the declared tools are supported"}})
            elif method == "item/completed" and data.get("item", {}).get("type") == "agentMessage":
                messages.append(data["item"]["text"])
            elif method == "turn/completed":
                turn = data["turn"]
                if turn["status"] == "completed":
                    return Turn("completed", messages)
                if turn["status"] == "interrupted":
                    return Turn("interrupted", messages)
                return Turn("failed", messages, json.dumps(turn.get("error"), ensure_ascii=False))
        self._interrupt(turn_id)
        return Turn("failed", messages, "turn timed out")

    def _take_usage(self, data):
        updated = {key: int(data["tokenUsage"]["total"].get(key, 0)) for key in TOKEN_KEYS}
        if updated["totalTokens"] < self.tokens["totalTokens"]:
            raise RuntimeError("the thread's token counter decreased")
        self.tokens = updated

    def _interrupt(self, turn_id):
        """Stop the running turn; refuse the tool calls it still issues, so nothing runs after the episode."""
        self.sequence += 1
        self._send({"id": self.sequence, "method": "turn/interrupt",
                    "params": {"threadId": self.thread_id, "turnId": turn_id}})
        deadline = time.monotonic() + self.rpc_timeout_s
        while time.monotonic() < deadline:
            event = self._receive(1)
            if event is None:
                continue
            method, data = event.get("method"), event.get("params", {})
            if method == "thread/tokenUsage/updated" and data.get("threadId") == self.thread_id:
                self._take_usage(data)
            elif method == "item/tool/call" and "id" in event:
                self._send({"id": event["id"], "result": {"success": False, "contentItems": [
                    {"type": "inputText", "text": "The episode has ended; this action was NOT executed."}]}})
            elif (method == "turn/completed" and data.get("threadId") == self.thread_id
                  and data.get("turn", {}).get("id") == turn_id):
                return
        raise RuntimeError("the model turn did not close")

    def usage(self):
        return dict(self.tokens)

    def is_transient(self, error):
        return error is not None and any(marker in error.lower() for marker in TRANSIENT)

    def close(self):
        try:
            if self.process is not None:
                if self.process.poll() is None:
                    try:
                        self.process.stdin.close()
                        self.process.wait(timeout=5)
                    except (subprocess.TimeoutExpired, OSError):
                        self.process.terminate()
                        try:
                            self.process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            self.process.kill()
                            self.process.wait(timeout=5)
                if self.reader is not None:
                    self.reader.join(timeout=5)
                self.process.stdin.close()
                self.process.stdout.close()
        finally:
            self.stderr.close()
