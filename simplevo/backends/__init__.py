"""Model backends: one conversation with an MLLM that can call the harness's tools.

The method needs only this from a model: a persistent conversation, image input, and tool calls whose results may
contain images. ``ModelSession`` is that contract. ``CodexSession`` drives a model through the Codex app-server;
``ScriptedSession`` runs a Python callable in place of a model, for tests and the offline example.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ToolResult:
    content: list[dict]                 # content items (simplevo.content)
    success: bool = True
    end_turn: bool = False              # the harness ends the model's turn after this result (episode over)


@dataclass
class Turn:
    status: str                         # "completed" | "interrupted" (ended by the harness) | "failed"
    messages: list[str] = field(default_factory=list)
    error: str | None = None


ToolHandler = Callable[[str, dict], ToolResult]


class ModelSession(ABC):
    @abstractmethod
    def start(self, tools: list[dict], instructions: str | None = None) -> None:
        """Open the conversation with its tool specifications and system instructions."""

    @abstractmethod
    def run_turn(self, inputs: list[dict], handle: ToolHandler) -> Turn:
        """Send content items as the next user message and serve tool calls with ``handle`` until the turn ends."""

    def usage(self) -> dict:
        """Cumulative token usage of the conversation, if the backend reports it."""
        return {}

    def is_transient(self, error: str | None) -> bool:
        """Whether a failed turn may be retried in the same conversation."""
        return False

    def close(self) -> None:
        pass


from .codex import CodexSession        # noqa: E402
from .scripted import ScriptedSession  # noqa: E402

__all__ = ["CodexSession", "ModelSession", "ScriptedSession", "ToolHandler", "ToolResult", "Turn"]
