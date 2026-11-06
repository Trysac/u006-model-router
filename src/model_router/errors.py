"""Typed errors raised by the model router."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Literal

ErrorCode = Literal["no_eligible_model", "all_models_failed", "no_port", "invalid_argument"]

_MAX_MESSAGE = 200


class RouterError(Exception):
    """Base class of every error raised by this library.

    Attributes:
        code: Stable machine-readable error code.
    """

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message[:_MAX_MESSAGE])
        self.code: ErrorCode = code


class NoEligibleModelError(RouterError):
    """Raised by ``route`` when every profile was excluded.

    Attributes:
        excluded: Read-only mapping of model id to exclusion reason code.
    """

    def __init__(self, excluded: Mapping[str, str]) -> None:
        super().__init__("no_eligible_model", f"no model is eligible ({len(excluded)} excluded)")
        self.excluded: Mapping[str, str] = MappingProxyType(dict(excluded))


class AllModelsFailedError(RouterError):
    """Raised by ``run`` when every attempted candidate failed with a retryable error.

    Attributes:
        attempts: ``(model_id, exception_type_name)`` pairs in attempt order. Exception
            text is never stored because it can carry request data.
    """

    def __init__(self, attempts: Sequence[tuple[str, str]]) -> None:
        super().__init__("all_models_failed", f"all {len(attempts)} attempted models failed")
        self.attempts: tuple[tuple[str, str], ...] = tuple(attempts)


class RetryableError(Exception):
    """Raise from an adapter to signal that the next candidate should be tried."""
