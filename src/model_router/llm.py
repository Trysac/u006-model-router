"""Canonical LLM port and deterministic test doubles."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal, Protocol, TypedDict

from .errors import RouterError

MAX_RECORDED_CALLS = 1000

# The port is specified as a functional TypedDict; keep exactly that form.
Message = TypedDict("Message", {"role": Literal["system", "user", "assistant"], "content": str})  # noqa: UP013


class LLMPort(Protocol):
    """Minimal synchronous chat-completion port that adapters implement."""

    def complete(self, messages: Sequence[Message], *, max_tokens: int) -> str:
        """Return the completion text for ``messages``."""


class RecordingExhaustedError(RouterError):
    """Raised by ``RecordedLLM`` when more requests arrive than recorded replies."""

    def __init__(self) -> None:
        super().__init__("invalid_argument", "recorded replies are exhausted")


def _record(calls: list[list[Message]], messages: Sequence[Message]) -> None:
    calls.append([{"role": m["role"], "content": m["content"]} for m in messages])
    if len(calls) > MAX_RECORDED_CALLS:
        del calls[0]


class RecordedLLM:
    """Test double that returns pre-recorded replies in order.

    Every request is appended to ``calls`` (capped at ``MAX_RECORDED_CALLS`` entries).
    """

    def __init__(self, replies: Sequence[str]) -> None:
        if isinstance(replies, str) or not all(isinstance(r, str) for r in replies):
            raise RouterError("invalid_argument", "replies must be a sequence of strings")
        self._replies: tuple[str, ...] = tuple(replies)
        self._next = 0
        self.calls: list[list[Message]] = []

    def complete(self, messages: Sequence[Message], *, max_tokens: int) -> str:
        """Record the request and return the next reply.

        Raises:
            RecordingExhaustedError: if every recorded reply was already returned.
        """
        _record(self.calls, messages)
        if self._next >= len(self._replies):
            raise RecordingExhaustedError()
        reply = self._replies[self._next]
        self._next += 1
        return reply


class FakeLLM:
    """Test double that answers with ``fn(messages)``.

    Every request is appended to ``calls`` (capped at ``MAX_RECORDED_CALLS`` entries).
    """

    def __init__(self, fn: Callable[[Sequence[Message]], str]) -> None:
        self._fn = fn
        self.calls: list[list[Message]] = []

    def complete(self, messages: Sequence[Message], *, max_tokens: int) -> str:
        """Record the request and return ``fn(messages)``."""
        _record(self.calls, messages)
        return self._fn(messages)
