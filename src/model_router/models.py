"""Immutable data types and input validation for profiles, requests and decisions."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal

from .errors import RouterError

MAX_PROFILES = 1000
MAX_TOKENS = 10**12
MAX_LATENCY_MS = 10**9
MAX_CAPABILITIES = 256
MAX_NAME_LENGTH = 128
MAX_PRICE = Decimal(10**12)

_ID_PATTERN = re.compile(r"[A-Za-z0-9._:/-]{1,128}")


def invalid(message: str) -> RouterError:
    """Build an ``invalid_argument`` error with a fixed, input-free message."""
    return RouterError("invalid_argument", message)


def check_int(name: str, value: object, low: int, high: int) -> None:
    """Raise unless ``value`` is a non-boolean integer within ``[low, high]``."""
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise invalid(f"{name} must be an integer in [{low}, {high}]")


def _check_unit(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0.0 <= value <= 1.0:
        raise invalid(f"{name} must be a number in [0, 1]")


def _check_money(name: str, value: object) -> None:
    if not isinstance(value, Decimal) or not value.is_finite() or not 0 <= value <= MAX_PRICE:
        raise invalid(f"{name} must be a finite Decimal in [0, {MAX_PRICE}]")


def _check_text(name: str, value: object) -> None:
    if not isinstance(value, str) or not 1 <= len(value) <= MAX_NAME_LENGTH:
        raise invalid(f"{name} must be a string of 1 to {MAX_NAME_LENGTH} characters")


def _check_names(name: str, value: object) -> None:
    if not isinstance(value, frozenset) or len(value) > MAX_CAPABILITIES:
        raise invalid(f"{name} must be a frozenset of at most {MAX_CAPABILITIES} names")
    for item in value:
        _check_text(name, item)


@dataclass(frozen=True)
class ModelProfile:
    """Caller-supplied facts about one model.

    The library ships no prices or latencies; every value comes from the caller.

    Attributes:
        id: Unique identifier matching ``[A-Za-z0-9._:/-]{1,128}``.
        provider: Free-form provider label.
        input_per_mtok: Price per million input tokens.
        output_per_mtok: Price per million output tokens.
        p50_latency_ms: Median latency in milliseconds.
        quality: Quality score in ``[0, 1]``.
        context_window: Maximum input plus output tokens.
        capabilities: Capability names such as ``vision``.
    """

    id: str
    provider: str
    input_per_mtok: Decimal
    output_per_mtok: Decimal
    p50_latency_ms: int
    quality: float
    context_window: int
    capabilities: frozenset[str]

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or _ID_PATTERN.fullmatch(self.id) is None:
            raise invalid("id must match [A-Za-z0-9._:/-]{1,128}")
        _check_text("provider", self.provider)
        _check_money("input_per_mtok", self.input_per_mtok)
        _check_money("output_per_mtok", self.output_per_mtok)
        check_int("p50_latency_ms", self.p50_latency_ms, 0, MAX_LATENCY_MS)
        _check_unit("quality", self.quality)
        check_int("context_window", self.context_window, 1, MAX_TOKENS)
        _check_names("capabilities", self.capabilities)


@dataclass(frozen=True)
class RouteRequest:
    """What a caller needs from a model.

    Attributes:
        input_tokens: Expected prompt tokens.
        expected_output_tokens: Expected completion tokens.
        required_capabilities: Capabilities every candidate must have.
        min_quality: Minimum acceptable quality in ``[0, 1]``.
        max_cost: Maximum estimated cost, or ``None`` for no limit.
        max_latency_ms: Maximum median latency, or ``None`` for no limit.
    """

    input_tokens: int
    expected_output_tokens: int
    required_capabilities: frozenset[str] = field(default_factory=frozenset)
    min_quality: float = 0.0
    max_cost: Decimal | None = None
    max_latency_ms: int | None = None

    def __post_init__(self) -> None:
        check_int("input_tokens", self.input_tokens, 0, MAX_TOKENS)
        check_int("expected_output_tokens", self.expected_output_tokens, 0, MAX_TOKENS)
        _check_names("required_capabilities", self.required_capabilities)
        _check_unit("min_quality", self.min_quality)
        if self.max_cost is not None:
            _check_money("max_cost", self.max_cost)
        if self.max_latency_ms is not None:
            check_int("max_latency_ms", self.max_latency_ms, 0, MAX_LATENCY_MS)


@dataclass(frozen=True)
class RouteDecision:
    """Explained outcome of ``route``.

    Attributes:
        candidates: Eligible profiles, best first.
        excluded: Model id to exclusion reason code for every rejected profile.
        scores: Model id to score for every candidate; higher is better.
    """

    candidates: tuple[ModelProfile, ...]
    excluded: Mapping[str, str]
    scores: Mapping[str, float]
