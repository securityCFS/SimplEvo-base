"""A scripted stand-in for the model: a Python callable drives the same tools a model would.

``script(inputs, call)`` receives the content items of the user message and a ``call(name, arguments)`` function
that returns a ``ToolResult``; ``call.tools`` is the set of declared tool names. It returns the final message (or
None). Used by the tests and by the offline toy example; it involves no model and proves nothing about one.
"""
from __future__ import annotations

from . import ModelSession, ToolResult, Turn


class _TurnEnded(Exception):
    pass


class ScriptedSession(ModelSession):
    def __init__(self, script):
        self.script = script
        self.tools: set[str] = set()
        self.instructions = None
        self.calls = 0

    def start(self, tools, instructions=None):
        self.tools = {spec["name"] for spec in tools}
        self.instructions = instructions

    def run_turn(self, inputs, handle):
        ended = False

        def call(name: str, arguments: dict) -> ToolResult:
            nonlocal ended
            if ended:
                raise _TurnEnded
            if name not in self.tools:
                raise KeyError(f"tool {name} was not declared")
            self.calls += 1
            result = handle(name, arguments)
            ended = result.end_turn
            return result

        call.tools = self.tools
        try:
            message = self.script(inputs, call)
        except _TurnEnded:
            message = None
        return Turn("interrupted" if ended else "completed", [message] if message else [])

    def usage(self):
        return {"toolCalls": self.calls}
