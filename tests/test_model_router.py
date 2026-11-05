"""Tests for the public API of model_router."""

from __future__ import annotations

import dataclasses
import itertools
import logging
from collections.abc import Sequence
from decimal import Decimal

import pytest

import model_router
from model_router import (
    AllModelsFailedError,
    FakeLLM,
    Message,
    ModelProfile,
    NoEligibleModelError,
    RecordedLLM,
    RecordingExhaustedError,
    RetryableError,
    RouteDecision,
    RouteRequest,
    RouterError,
    RunResult,
    route,
    run,
)

ZERO = Decimal("0")
ONE = Decimal("1")
NO_CAPS: frozenset[str] = frozenset()
MESSAGES: list[Message] = [{"role": "user", "content": "SENTINEL-msg"}]
ABC_REQUEST = RouteRequest(input_tokens=1_000_000, expected_output_tokens=0)


def make_profile(
    model_id: str,
    *,
    input_per_mtok: Decimal = ONE,
    output_per_mtok: Decimal = ZERO,
    latency: int = 100,
    quality: float = 0.5,
    window: int = 1_000_000,
    capabilities: frozenset[str] = NO_CAPS,
) -> ModelProfile:
    return ModelProfile(
        id=model_id,
        provider="acme",
        input_per_mtok=input_per_mtok,
        output_per_mtok=output_per_mtok,
        p50_latency_ms=latency,
        quality=quality,
        context_window=window,
        capabilities=capabilities,
    )


def abc_profiles() -> list[ModelProfile]:
    return [
        make_profile("a", input_per_mtok=Decimal("1"), latency=300, quality=0.5),
        make_profile("b", input_per_mtok=Decimal("3"), latency=100, quality=0.9),
        make_profile("c", input_per_mtok=Decimal("2"), latency=200, quality=0.8),
    ]


GOOD = make_profile("fine", latency=100, quality=0.9)


def ids(decision: RouteDecision) -> list[str]:
    return [profile.id for profile in decision.candidates]


def failing(error: BaseException) -> FakeLLM:
    def raise_error(messages: Sequence[Message]) -> str:
        raise error

    return FakeLLM(raise_error)


def cheapest_pair() -> RouteDecision:
    return route(abc_profiles()[:2], ABC_REQUEST, "cheapest")


def good_ports() -> dict[str, RecordedLLM]:
    return {"a": RecordedLLM(["ok"]), "b": RecordedLLM(["ok"])}


# --- routing ---------------------------------------------------------------------------


def test_capability_filter_keeps_only_vision_models() -> None:
    profiles = [
        make_profile("small"),
        make_profile("large", capabilities=frozenset({"vision"})),
        make_profile("mid", capabilities=frozenset({"vision", "tools"})),
    ]
    req = RouteRequest(
        input_tokens=1000,
        expected_output_tokens=100,
        required_capabilities=frozenset({"vision"}),
    )
    decision = route(profiles, req)
    assert sorted(ids(decision)) == ["large", "mid"]
    assert decision.excluded == {"small": "missing_capability:vision"}


def test_policy_orders() -> None:
    profiles = abc_profiles()
    assert ids(route(profiles, ABC_REQUEST, "cheapest")) == ["a", "c", "b"]
    assert ids(route(profiles, ABC_REQUEST, "fastest")) == ["b", "c", "a"]
    assert ids(route(profiles, ABC_REQUEST, "best")) == ["b", "c", "a"]
    assert ids(route(profiles, ABC_REQUEST, "balanced")) == ["b", "c", "a"]


def test_balanced_scores_and_default_policy() -> None:
    decision = route(abc_profiles(), ABC_REQUEST)
    assert ids(decision) == ["b", "c", "a"]
    expected = {"a": 0.3, "b": 0.7, "c": 0.625}
    assert set(decision.scores) == set(expected)
    for model_id, value in expected.items():
        assert decision.scores[model_id] == pytest.approx(value, abs=1e-9)


def test_scores_of_single_criterion_policies() -> None:
    profiles = abc_profiles()
    cheapest = route(profiles, ABC_REQUEST, "cheapest").scores
    assert cheapest == pytest.approx({"a": 1.0, "c": 0.5, "b": 0.0})
    fastest = route(profiles, ABC_REQUEST, "fastest").scores
    assert fastest == pytest.approx({"b": 1.0, "c": 0.5, "a": 0.0})
    best = route(profiles, ABC_REQUEST, "best").scores
    assert best == pytest.approx({"b": 1.0, "c": 0.75, "a": 0.0})


def test_weights_are_normalized_and_ordered_quality_cost_latency() -> None:
    profiles = abc_profiles()
    only_quality = route(profiles, ABC_REQUEST, "balanced", (1, 0, 0))
    assert only_quality.scores == pytest.approx({"b": 1.0, "c": 0.75, "a": 0.0})
    only_cost = route(profiles, ABC_REQUEST, "balanced", (0, 4, 0))
    assert ids(only_cost) == ["a", "c", "b"]
    only_latency = route(profiles, ABC_REQUEST, "balanced", (0, 0, 2))
    assert ids(only_latency) == ["b", "c", "a"]
    scaled = route(profiles, ABC_REQUEST, "balanced", (5, 3, 2))
    assert scaled.scores == pytest.approx(route(profiles, ABC_REQUEST).scores)


def test_single_eligible_model() -> None:
    decision = route([make_profile("only")], ABC_REQUEST)
    assert ids(decision) == ["only"]
    assert decision.scores["only"] == pytest.approx(0.5)


@pytest.mark.parametrize("policy", ["cheapest", "fastest", "best", "balanced"])
def test_equal_models_tie_break_by_id(policy: str) -> None:
    profiles = [make_profile("z"), make_profile("m"), make_profile("a")]
    decision = route(profiles, ABC_REQUEST, policy)  # type: ignore[arg-type]
    assert ids(decision) == ["a", "m", "z"]
    assert len(set(decision.scores.values())) == 1


def test_input_order_does_not_change_candidates() -> None:
    profiles = abc_profiles()
    for policy in ("cheapest", "fastest", "best", "balanced"):
        baseline = ids(route(profiles, ABC_REQUEST, policy))  # type: ignore[arg-type]
        for permutation in itertools.permutations(profiles):
            decision = route(permutation, ABC_REQUEST, policy)  # type: ignore[arg-type]
            assert ids(decision) == baseline


def test_max_cost_excludes() -> None:
    bad = make_profile("bad", input_per_mtok=Decimal("10"))
    req = RouteRequest(input_tokens=1_000_000, expected_output_tokens=0, max_cost=Decimal("5"))
    assert route([GOOD, bad], req).excluded == {"bad": "max_cost"}


def test_max_latency_excludes() -> None:
    bad = make_profile("bad", latency=900)
    req = RouteRequest(input_tokens=10, expected_output_tokens=10, max_latency_ms=500)
    assert route([GOOD, bad], req).excluded == {"bad": "max_latency"}


def test_min_quality_excludes() -> None:
    bad = make_profile("bad", quality=0.2)
    req = RouteRequest(input_tokens=10, expected_output_tokens=10, min_quality=0.5)
    assert route([GOOD, bad], req).excluded == {"bad": "min_quality"}


def test_context_window_excludes() -> None:
    bad = make_profile("bad", window=1000)
    req = RouteRequest(input_tokens=900, expected_output_tokens=200)
    decision = route([GOOD, bad], req)
    assert decision.excluded == {"bad": "context_window"}
    assert ids(decision) == ["fine"]


def test_limits_are_inclusive() -> None:
    edge = make_profile(
        "edge",
        input_per_mtok=Decimal("3"),
        latency=500,
        quality=0.5,
        window=600_000,
    )
    req = RouteRequest(
        input_tokens=500_000,
        expected_output_tokens=100_000,
        min_quality=0.5,
        max_cost=Decimal("1.5"),
        max_latency_ms=500,
    )
    decision = route([edge], req)
    assert ids(decision) == ["edge"]
    assert decision.excluded == {}
    tighter = dataclasses.replace(req, max_cost=Decimal("1.4999"))
    with pytest.raises(NoEligibleModelError) as info:
        route([edge], tighter)
    assert info.value.excluded == {"edge": "max_cost"}


def test_cost_includes_output_tokens() -> None:
    profile = make_profile("p", input_per_mtok=Decimal("2"), output_per_mtok=Decimal("10"))
    req = RouteRequest(input_tokens=500_000, expected_output_tokens=100_000, max_cost=Decimal("2"))
    assert ids(route([profile], req)) == ["p"]
    cheaper = dataclasses.replace(req, max_cost=Decimal("1.99"))
    with pytest.raises(NoEligibleModelError) as info:
        route([profile], cheaper)
    assert info.value.excluded == {"p": "max_cost"}


def test_first_matching_reason_wins_in_fixed_order() -> None:
    caps = frozenset({"audio", "vision"})
    profiles = [
        make_profile(
            "p1",
            input_per_mtok=Decimal("10"),
            latency=900,
            quality=0.1,
            window=10,
            capabilities=frozenset({"zzz"}),
        ),
        make_profile(
            "p2",
            input_per_mtok=Decimal("10"),
            latency=900,
            quality=0.1,
            window=10,
            capabilities=caps,
        ),
        make_profile(
            "p3",
            input_per_mtok=Decimal("10"),
            latency=900,
            quality=0.1,
            capabilities=caps,
        ),
        make_profile(
            "p4",
            input_per_mtok=Decimal("10"),
            latency=900,
            quality=0.9,
            capabilities=caps,
        ),
        make_profile("p5", latency=900, quality=0.9, capabilities=caps),
        make_profile("ok", latency=100, quality=0.9, capabilities=caps),
    ]
    req = RouteRequest(
        input_tokens=1_000_000,
        expected_output_tokens=0,
        required_capabilities=frozenset({"vision", "audio"}),
        min_quality=0.5,
        max_cost=Decimal("5"),
        max_latency_ms=500,
    )
    decision = route(profiles, req)
    assert decision.excluded == {
        "p1": "missing_capability:audio",
        "p2": "context_window",
        "p3": "min_quality",
        "p4": "max_cost",
        "p5": "max_latency",
    }
    assert ids(decision) == ["ok"]


def test_no_eligible_model_raises_with_full_reason_map() -> None:
    profiles = [make_profile("a", quality=0.1), make_profile("b", window=5)]
    req = RouteRequest(input_tokens=10, expected_output_tokens=0, min_quality=0.5)
    with pytest.raises(NoEligibleModelError) as info:
        route(profiles, req)
    assert info.value.code == "no_eligible_model"
    assert info.value.excluded == {"a": "min_quality", "b": "context_window"}
    assert isinstance(info.value, RouterError)


def test_decision_is_immutable() -> None:
    decision = route(abc_profiles(), ABC_REQUEST)
    with pytest.raises(dataclasses.FrozenInstanceError):
        decision.candidates = ()  # type: ignore[misc]
    with pytest.raises(TypeError):
        decision.excluded["x"] = "y"  # type: ignore[index]
    with pytest.raises(TypeError):
        decision.scores["x"] = 1.0  # type: ignore[index]


# --- input validation ------------------------------------------------------------------


def test_empty_and_duplicate_profiles_are_invalid() -> None:
    with pytest.raises(RouterError) as empty:
        route([], ABC_REQUEST)
    assert empty.value.code == "invalid_argument"
    with pytest.raises(RouterError) as duplicate:
        route([make_profile("a"), make_profile("a")], ABC_REQUEST)
    assert duplicate.value.code == "invalid_argument"


def test_profile_count_is_bounded() -> None:
    many = [make_profile(f"m{i}") for i in range(1000)]
    assert len(route(many, ABC_REQUEST).candidates) == 1000
    with pytest.raises(RouterError) as info:
        route([*many, make_profile("extra")], ABC_REQUEST)
    assert info.value.code == "invalid_argument"


@pytest.mark.parametrize("profiles", ["abc", b"abc", 5, [1], [make_profile("a"), None]])
def test_bad_profile_containers(profiles: object) -> None:
    with pytest.raises(RouterError) as info:
        route(profiles, ABC_REQUEST)  # type: ignore[arg-type]
    assert info.value.code == "invalid_argument"


def test_bad_request_policy_and_weights_types() -> None:
    with pytest.raises(RouterError) as bad_request:
        route(abc_profiles(), "request")  # type: ignore[arg-type]
    assert bad_request.value.code == "invalid_argument"
    with pytest.raises(RouterError) as bad_policy:
        route(abc_profiles(), ABC_REQUEST, "random")  # type: ignore[arg-type]
    assert bad_policy.value.code == "invalid_argument"


@pytest.mark.parametrize(
    "weights",
    [
        (0, 0, 0),
        (-1, 1, 1),
        (1, 1),
        (0.5, 0.3, 0.2, 0.1),
        [0.5, 0.3, 0.2],
        (True, 1, 1),
        (float("nan"), 1, 1),
        (float("inf"), 1, 1),
        (1e13, 0, 0),
        ("a", 1, 1),
    ],
)
def test_invalid_weights(weights: object) -> None:
    with pytest.raises(RouterError) as info:
        route(abc_profiles(), ABC_REQUEST, "balanced", weights)  # type: ignore[arg-type]
    assert info.value.code == "invalid_argument"


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("id", ""),
        ("id", "has space"),
        ("id", "a\n"),
        ("id", "x" * 129),
        ("id", 7),
        ("provider", ""),
        ("input_per_mtok", Decimal("-1")),
        ("input_per_mtok", Decimal("NaN")),
        ("input_per_mtok", 1.5),
        ("output_per_mtok", Decimal("Infinity")),
        ("p50_latency_ms", -1),
        ("p50_latency_ms", True),
        ("quality", 1.5),
        ("quality", -0.1),
        ("quality", float("nan")),
        ("quality", True),
        ("context_window", 0),
        ("capabilities", {"vision"}),
        ("capabilities", frozenset({""})),
        ("capabilities", frozenset(str(i) for i in range(257))),
    ],
)
def test_invalid_profile_fields(field_name: str, value: object) -> None:
    with pytest.raises(RouterError) as info:
        dataclasses.replace(make_profile("a"), **{field_name: value})
    assert info.value.code == "invalid_argument"


def test_valid_id_characters_and_error_does_not_echo_input() -> None:
    assert make_profile("vendor/model-1.5:beta_x").id == "vendor/model-1.5:beta_x"
    with pytest.raises(RouterError) as info:
        make_profile("SECRET!")
    assert "SECRET" not in str(info.value)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("input_tokens", -1),
        ("input_tokens", 1.5),
        ("input_tokens", True),
        ("input_tokens", 10**12 + 1),
        ("expected_output_tokens", -1),
        ("required_capabilities", {"x"}),
        ("min_quality", 2.0),
        ("min_quality", float("nan")),
        ("max_cost", Decimal("-1")),
        ("max_cost", 1.0),
        ("max_cost", Decimal("NaN")),
        ("max_latency_ms", -5),
        ("max_latency_ms", 1.5),
    ],
)
def test_invalid_request_fields(field_name: str, value: object) -> None:
    base = RouteRequest(input_tokens=1, expected_output_tokens=1)
    with pytest.raises(RouterError) as info:
        dataclasses.replace(base, **{field_name: value})
    assert info.value.code == "invalid_argument"


# --- run -------------------------------------------------------------------------------


def test_timeout_falls_back_to_next_candidate() -> None:
    ports = {"a": failing(TimeoutError()), "b": RecordedLLM(["ok"])}
    result = run(cheapest_pair(), ports, MESSAGES, max_tokens=64)
    assert result == RunResult(model_id="b", output="ok", attempts=2)
    assert ports["b"].calls == [[{"role": "user", "content": "SENTINEL-msg"}]]


def test_non_retryable_error_propagates_unchanged() -> None:
    error = ValueError("boom")
    second = RecordedLLM(["ok"])
    ports = {"a": failing(error), "b": second}
    events: list[dict[str, object]] = []
    with pytest.raises(ValueError) as info:
        run(cheapest_pair(), ports, MESSAGES, max_tokens=64, on_event=events.append)
    assert info.value is error
    assert second.calls == []
    assert events == [
        {"type": "attempt", "model_id": "a", "ok": False, "error_type": "ValueError"},
    ]


def test_all_models_failed_reports_type_names_only() -> None:
    ports = {
        "a": failing(TimeoutError("EXC-TEXT")),
        "b": failing(RetryableError("EXC-TEXT")),
    }
    events: list[dict[str, object]] = []
    with pytest.raises(AllModelsFailedError) as info:
        run(cheapest_pair(), ports, MESSAGES, max_tokens=64, on_event=events.append)
    assert info.value.code == "all_models_failed"
    assert info.value.attempts == (("a", "TimeoutError"), ("b", "RetryableError"))
    assert "EXC-TEXT" not in str(info.value)
    assert "SENTINEL-msg" not in repr(events)
    assert "EXC-TEXT" not in repr(events)
    assert [event["model_id"] for event in events] == ["a", "b"]
    assert all(event["ok"] is False for event in events)


def test_success_event_payload() -> None:
    events: list[dict[str, object]] = []
    run(cheapest_pair(), good_ports(), MESSAGES, max_tokens=8, on_event=events.append)
    assert events == [
        {"type": "attempt", "model_id": "a", "ok": True, "error_type": None},
    ]


def test_missing_port_raises_before_any_call() -> None:
    port_a = RecordedLLM(["ok"])
    with pytest.raises(RouterError) as info:
        run(cheapest_pair(), {"a": port_a}, MESSAGES, max_tokens=10)
    assert info.value.code == "no_port"
    assert port_a.calls == []


def test_max_candidates_limits_attempts_and_port_check() -> None:
    decision = route(abc_profiles(), ABC_REQUEST, "best")
    ports = {"b": failing(TimeoutError()), "c": failing(TimeoutError())}
    with pytest.raises(AllModelsFailedError) as info:
        run(decision, ports, MESSAGES, max_tokens=10, max_candidates=2)
    assert info.value.attempts == (("b", "TimeoutError"), ("c", "TimeoutError"))


def test_max_candidates_larger_than_candidates() -> None:
    ports = {"a": failing(TimeoutError()), "b": RecordedLLM(["ok"])}
    result = run(cheapest_pair(), ports, MESSAGES, max_tokens=10, max_candidates=50)
    assert result.model_id == "b"


@pytest.mark.parametrize("error", [ConnectionError(), RetryableError()])
def test_default_retry_set(error: BaseException) -> None:
    ports = {"a": failing(error), "b": RecordedLLM(["ok"])}
    assert run(cheapest_pair(), ports, MESSAGES, max_tokens=10).attempts == 2


def test_custom_retry_on() -> None:
    ports = {"a": failing(KeyError("k")), "b": RecordedLLM(["ok"])}
    result = run(cheapest_pair(), ports, MESSAGES, max_tokens=10, retry_on=(KeyError,))
    assert result.model_id == "b"
    ports = {"a": failing(TimeoutError()), "b": RecordedLLM(["ok"])}
    with pytest.raises(TimeoutError):
        run(cheapest_pair(), ports, MESSAGES, max_tokens=10, retry_on=(KeyError,))


def test_recording_exhausted_propagates_from_run() -> None:
    ports = {"a": RecordedLLM([]), "b": RecordedLLM(["ok"])}
    with pytest.raises(RecordingExhaustedError):
        run(cheapest_pair(), ports, MESSAGES, max_tokens=10)


def test_failing_callback_is_logged_by_type_and_ignored(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="model_router")

    def boom(payload: dict[str, object]) -> None:
        raise RuntimeError("callback-detail")

    result = run(cheapest_pair(), good_ports(), MESSAGES, max_tokens=10, on_event=boom)
    assert result.output == "ok"
    assert "RuntimeError" in caplog.text
    assert "callback-detail" not in caplog.text


def test_logging_never_contains_message_text(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="model_router")
    ports = {"a": failing(TimeoutError("EXC-TEXT")), "b": RecordedLLM(["ok"])}
    run(cheapest_pair(), ports, MESSAGES, max_tokens=10)
    route(abc_profiles(), ABC_REQUEST)
    assert "TimeoutError" in caplog.text
    assert "SENTINEL-msg" not in caplog.text
    assert "EXC-TEXT" not in caplog.text


def test_run_rejects_bad_decision_and_ports() -> None:
    empty = RouteDecision(candidates=(), excluded={}, scores={})
    for decision in ("nope", empty):
        with pytest.raises(RouterError) as info:
            run(decision, good_ports(), MESSAGES, max_tokens=10)  # type: ignore[arg-type]
        assert info.value.code == "invalid_argument"
    with pytest.raises(RouterError) as bad_ports:
        run(cheapest_pair(), ["a"], MESSAGES, max_tokens=10)  # type: ignore[arg-type]
    assert bad_ports.value.code == "invalid_argument"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("max_tokens", 0),
        ("max_tokens", -1),
        ("max_tokens", True),
        ("max_tokens", 1.5),
        ("max_candidates", 0),
        ("max_candidates", True),
        ("max_candidates", 1001),
        ("retry_on", [TimeoutError]),
        ("retry_on", (ValueError, "x")),
        ("retry_on", (int,)),
        ("on_event", "callback"),
    ],
)
def test_run_rejects_bad_scalars(name: str, value: object) -> None:
    kwargs: dict[str, object] = {"max_tokens": 10, name: value}
    with pytest.raises(RouterError) as info:
        run(cheapest_pair(), good_ports(), MESSAGES, **kwargs)  # type: ignore[arg-type]
    assert info.value.code == "invalid_argument"


def test_empty_retry_on_is_allowed() -> None:
    ports = {"a": failing(TimeoutError()), "b": RecordedLLM(["ok"])}
    with pytest.raises(TimeoutError):
        run(cheapest_pair(), ports, MESSAGES, max_tokens=10, retry_on=())


BAD_MESSAGES = [
    [],
    "hello",
    ["not a mapping"],
    [{"role": "tool", "content": "x"}],
    [{"role": ["user"], "content": "x"}],
    [{"role": "user", "content": 5}],
    [{"role": "user"}],
    [{"role": "user", "content": "x" * 1_000_001}],
    [{"role": "user", "content": "x"}] * 1001,
    [{"role": "user", "content": "x" * 900_000}] * 5,
]


@pytest.mark.parametrize("messages", BAD_MESSAGES)
def test_run_rejects_bad_messages_before_calling_ports(messages: object) -> None:
    ports = good_ports()
    with pytest.raises(RouterError) as info:
        run(cheapest_pair(), ports, messages, max_tokens=10)  # type: ignore[arg-type]
    assert info.value.code == "invalid_argument"
    assert ports["a"].calls == []


def test_run_accepts_all_roles() -> None:
    messages: list[Message] = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    assert run(cheapest_pair(), good_ports(), messages, max_tokens=10).output == "ok"


# --- doubles, errors, package ----------------------------------------------------------


def test_recorded_llm_returns_replies_in_order_then_raises() -> None:
    llm = RecordedLLM(["one", "two"])
    assert llm.complete(MESSAGES, max_tokens=5) == "one"
    assert llm.complete(MESSAGES, max_tokens=5) == "two"
    with pytest.raises(RecordingExhaustedError) as info:
        llm.complete(MESSAGES, max_tokens=5)
    assert info.value.code == "invalid_argument"
    assert len(llm.calls) == 3


@pytest.mark.parametrize("replies", ["abc", [1], ["ok", None]])
def test_recorded_llm_rejects_bad_replies(replies: object) -> None:
    with pytest.raises(RouterError):
        RecordedLLM(replies)  # type: ignore[arg-type]


def test_fake_llm_calls_function_and_records_copies() -> None:
    llm = FakeLLM(lambda messages: messages[-1]["content"].upper())
    messages: list[Message] = [{"role": "user", "content": "hello"}]
    assert llm.complete(messages, max_tokens=5) == "HELLO"
    messages[0]["content"] = "changed"
    assert llm.calls == [[{"role": "user", "content": "hello"}]]


def test_recorded_calls_are_capped() -> None:
    llm = FakeLLM(lambda messages: "x")
    for _ in range(1005):
        llm.complete(MESSAGES, max_tokens=1)
    assert len(llm.calls) == 1000


def test_error_messages_are_truncated_and_coded() -> None:
    error = RouterError("invalid_argument", "x" * 500)
    assert len(str(error)) == 200
    assert error.code == "invalid_argument"


def test_package_surface() -> None:
    assert model_router.__version__ == "0.1.0"
    for name in model_router.__all__:
        assert hasattr(model_router, name)
    handlers = logging.getLogger("model_router").handlers
    assert any(isinstance(handler, logging.NullHandler) for handler in handlers)
