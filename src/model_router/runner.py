"""Ordered fallback execution of a routing decision."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from .errors import AllModelsFailedError, RetryableError, RouterError
from .llm import LLMPort, Message
from .models import MAX_PROFILES, MAX_TOKENS, RouteDecision, check_int, invalid

logger = logging.getLogger(__name__)

MAX_MESSAGES = 1000
MAX_MESSAGE_CHARS = 1_000_000
MAX_TOTAL_CHARS = 4_000_000

_ROLES = frozenset({"system", "user", "assistant"})

EventCallback = Callable[[dict[str, object]], None]


@dataclass(frozen=True)
class RunResult:
    """Outcome of a successful ``run``.

    Attributes:
        model_id: Id of the model that answered.
        output: The completion text.
        attempts: Number of models called, including the successful one.
    """

    model_id: str
    output: str
    attempts: int


def _valid_retry_on(value: object) -> bool:
    if not isinstance(value, tuple):
        return False
    return all(isinstance(item, type) and issubclass(item, BaseException) for item in value)


def _validate_messages(messages: Sequence[Message]) -> None:
    if isinstance(messages, str | bytes) or not isinstance(messages, Sequence):
        raise invalid("messages must be a sequence of message objects")
    if not 1 <= len(messages) <= MAX_MESSAGES:
        raise invalid(f"messages must contain between 1 and {MAX_MESSAGES} items")
    total = 0
    for raw in cast("Sequence[object]", messages):
        if not isinstance(raw, Mapping):
            raise invalid("each message must be a mapping")
        role = raw.get("role")
        if not isinstance(role, str) or role not in _ROLES:
            raise invalid("each message needs a role of system, user or assistant")
        content = raw.get("content")
        if not isinstance(content, str) or len(content) > MAX_MESSAGE_CHARS:
            raise invalid(f"message content must be a string of at most {MAX_MESSAGE_CHARS} chars")
        total += len(content)
    if total > MAX_TOTAL_CHARS:
        raise invalid(f"messages must not exceed {MAX_TOTAL_CHARS} characters in total")


def _validate_run(
    decision: RouteDecision,
    ports: Mapping[str, LLMPort],
    max_tokens: int,
    max_candidates: int,
    retry_on: tuple[type[BaseException], ...],
    on_event: EventCallback | None,
) -> None:
    if not isinstance(decision, RouteDecision) or not decision.candidates:
        raise invalid("decision must be a RouteDecision with at least one candidate")
    if not isinstance(ports, Mapping):
        raise invalid("ports must be a mapping of model id to LLMPort")
    check_int("max_tokens", max_tokens, 1, MAX_TOKENS)
    check_int("max_candidates", max_candidates, 1, MAX_PROFILES)
    if not _valid_retry_on(retry_on):
        raise invalid("retry_on must be a tuple of exception classes")
    if on_event is not None and not callable(on_event):
        raise invalid("on_event must be callable or None")


def _emit(on_event: EventCallback | None, model_id: str, ok: bool, error: str | None) -> None:
    if on_event is None:
        return
    payload: dict[str, object] = {
        "type": "attempt",
        "model_id": model_id,
        "ok": ok,
        "error_type": error,
    }
    try:
        on_event(payload)
    except Exception as exc:
        logger.warning("on_event callback failed: %s", type(exc).__name__)


def run(
    decision: RouteDecision,
    ports: Mapping[str, LLMPort],
    messages: Sequence[Message],
    *,
    max_tokens: int,
    max_candidates: int = 3,
    retry_on: tuple[type[BaseException], ...] = (RetryableError, TimeoutError, ConnectionError),
    on_event: EventCallback | None = None,
) -> RunResult:
    """Call the decision's candidates in order until one answers.

    At most ``max_candidates`` candidates are tried. Only exceptions in ``retry_on`` move on
    to the next candidate; any other exception propagates unchanged. Every selected
    candidate must have a port, checked before any port is called. ``on_event`` receives
    ``{'type': 'attempt', 'model_id', 'ok', 'error_type'}`` after each call; a failing
    callback is logged by exception type and ignored. Attempts record exception type names
    only, never exception text or message content.

    Raises:
        RouterError: ``invalid_argument`` for bad input, ``no_port`` when a candidate has no port.
        AllModelsFailedError: when every attempted candidate failed with a retryable error.
    """
    _validate_run(decision, ports, max_tokens, max_candidates, retry_on, on_event)
    _validate_messages(messages)
    selected = decision.candidates[:max_candidates]
    for profile in selected:
        if profile.id not in ports:
            raise RouterError("no_port", f"no port registered for model {profile.id}")
    attempts: list[tuple[str, str]] = []
    for profile in selected:
        port = ports[profile.id]
        try:
            output = port.complete(messages, max_tokens=max_tokens)
        except retry_on as retry_exc:
            name = type(retry_exc).__name__
            attempts.append((profile.id, name))
            logger.debug("model %s failed with %s", profile.id, name)
            _emit(on_event, profile.id, False, name)
        except BaseException as fatal_exc:
            _emit(on_event, profile.id, False, type(fatal_exc).__name__)
            raise
        else:
            _emit(on_event, profile.id, True, None)
            return RunResult(model_id=profile.id, output=output, attempts=len(attempts) + 1)
    raise AllModelsFailedError(attempts)
