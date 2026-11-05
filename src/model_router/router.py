"""Deterministic, explainable rule-based routing over caller-supplied profiles."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from decimal import Context, Decimal
from types import MappingProxyType
from typing import Literal

from .errors import NoEligibleModelError
from .models import MAX_PROFILES, ModelProfile, RouteDecision, RouteRequest, invalid

logger = logging.getLogger(__name__)

Policy = Literal["cheapest", "fastest", "best", "balanced"]
Weights = tuple[float, float, float]

_POLICIES = frozenset({"cheapest", "fastest", "best", "balanced"})
_MAX_WEIGHT = 1e12
_MILLION = Decimal(1_000_000)
_PRECISION = 60


def _estimate_cost(ctx: Context, profile: ModelProfile, request: RouteRequest) -> Decimal:
    tokens_in = Decimal(request.input_tokens)
    tokens_out = Decimal(request.expected_output_tokens)
    cost_in = ctx.divide(ctx.multiply(tokens_in, profile.input_per_mtok), _MILLION)
    cost_out = ctx.divide(ctx.multiply(tokens_out, profile.output_per_mtok), _MILLION)
    return ctx.add(cost_in, cost_out)


def _exclusion_reason(profile: ModelProfile, request: RouteRequest, cost: Decimal) -> str | None:
    missing = request.required_capabilities - profile.capabilities
    if missing:
        return f"missing_capability:{min(missing)}"
    if request.input_tokens + request.expected_output_tokens > profile.context_window:
        return "context_window"
    if profile.quality < request.min_quality:
        return "min_quality"
    if request.max_cost is not None and cost > request.max_cost:
        return "max_cost"
    if request.max_latency_ms is not None and profile.p50_latency_ms > request.max_latency_ms:
        return "max_latency"
    return None


def _normalize(values: list[float]) -> list[float]:
    low, high = min(values), max(values)
    if low == high:
        return [0.0] * len(values)
    span = high - low
    return [(value - low) / span for value in values]


def _normalize_costs(ctx: Context, values: list[Decimal]) -> list[float]:
    low, high = min(values), max(values)
    if low == high:
        return [0.0] * len(values)
    span = ctx.subtract(high, low)
    return [float(ctx.divide(ctx.subtract(value, low), span)) for value in values]


def _valid_weight(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    return 0 <= value <= _MAX_WEIGHT


def _normalize_weights(weights: Weights) -> Weights:
    if not isinstance(weights, tuple) or len(weights) != 3:
        raise invalid("weights must be a tuple of three numbers")
    if not all(_valid_weight(w) for w in weights):
        raise invalid("weights must be numbers in [0, 1e12]")
    total = math.fsum(weights)
    if total <= 0:
        raise invalid("weights must not sum to zero")
    return (weights[0] / total, weights[1] / total, weights[2] / total)


def _validate(
    profiles: Sequence[ModelProfile],
    request: RouteRequest,
    policy: str,
    weights: Weights,
) -> Weights:
    if isinstance(profiles, str | bytes) or not isinstance(profiles, Sequence):
        raise invalid("profiles must be a sequence of ModelProfile")
    if not 1 <= len(profiles) <= MAX_PROFILES:
        raise invalid(f"profiles must contain between 1 and {MAX_PROFILES} items")
    seen: set[str] = set()
    for profile in profiles:
        if not isinstance(profile, ModelProfile):
            raise invalid("every profile must be a ModelProfile")
        if profile.id in seen:
            raise invalid("profile ids must be unique")
        seen.add(profile.id)
    if not isinstance(request, RouteRequest):
        raise invalid("request must be a RouteRequest")
    if policy not in _POLICIES:
        raise invalid("policy must be cheapest, fastest, best or balanced")
    return _normalize_weights(weights)


def _score(policy: str, weights: Weights, q: float, c: float, lat: float) -> float:
    if policy == "cheapest":
        return 1.0 - c
    if policy == "fastest":
        return 1.0 - lat
    if policy == "best":
        return q
    w_q, w_c, w_l = weights
    return w_q * q + w_c * (1.0 - c) + w_l * (1.0 - lat)


def route(
    profiles: Sequence[ModelProfile],
    request: RouteRequest,
    policy: Policy = "balanced",
    weights: Weights = (0.5, 0.3, 0.2),
) -> RouteDecision:
    """Order the eligible models for ``request`` and explain every exclusion.

    Estimated cost is ``input_tokens * input_per_mtok / 10^6 + expected_output_tokens *
    output_per_mtok / 10^6`` in ``Decimal``. A profile is excluded with the first matching
    reason, checked in this order: ``missing_capability:<name>`` (alphabetically first
    missing name), ``context_window``, ``min_quality``, ``max_cost``, ``max_latency``.

    Eligible models are ordered by ``policy``: ``cheapest`` by estimated cost, ``fastest``
    by median latency, ``best`` by quality (descending) and ``balanced`` by the weighted
    score ``w_q * q_norm + w_c * (1 - c_norm) + w_l * (1 - l_norm)`` with min-max
    normalization over the eligible models (a norm is 0 when min equals max). ``weights``
    is ordered ``(w_q, w_c, w_l)``, must be non-negative and is divided by its sum. Ties
    break by id ascending. ``RouteDecision.scores`` holds a higher-is-better value per
    candidate: the policy's normalized criterion, or the balanced score.

    Raises:
        RouterError: with code ``invalid_argument`` for bad input.
        NoEligibleModelError: when every profile was excluded.
    """
    norm_weights = _validate(profiles, request, policy, weights)
    ctx = Context(prec=_PRECISION)
    excluded: dict[str, str] = {}
    eligible: list[ModelProfile] = []
    costs: dict[str, Decimal] = {}
    for profile in profiles:
        cost = _estimate_cost(ctx, profile, request)
        reason = _exclusion_reason(profile, request, cost)
        if reason is None:
            eligible.append(profile)
            costs[profile.id] = cost
        else:
            excluded[profile.id] = reason
    if not eligible:
        logger.debug("no eligible model among %d profiles", len(profiles))
        raise NoEligibleModelError(excluded)
    c_norm = _normalize_costs(ctx, [costs[p.id] for p in eligible])
    q_norm = _normalize([float(p.quality) for p in eligible])
    l_norm = _normalize([float(p.p50_latency_ms) for p in eligible])
    scores = {
        p.id: _score(policy, norm_weights, q, c, lat)
        for p, q, c, lat in zip(eligible, q_norm, c_norm, l_norm, strict=True)
    }
    ordered = sorted(eligible, key=lambda p: p.id)
    if policy == "cheapest":
        ordered.sort(key=lambda p: costs[p.id])
    elif policy == "fastest":
        ordered.sort(key=lambda p: p.p50_latency_ms)
    elif policy == "best":
        ordered.sort(key=lambda p: -p.quality)
    else:
        ordered.sort(key=lambda p: -scores[p.id])
    logger.debug("routed %d eligible, %d excluded (%s)", len(ordered), len(excluded), policy)
    return RouteDecision(
        candidates=tuple(ordered),
        excluded=MappingProxyType(excluded),
        scores=MappingProxyType({p.id: scores[p.id] for p in ordered}),
    )
