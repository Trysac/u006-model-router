# model-router

[![PyPI version](https://img.shields.io/pypi/v/trysac-model-router)](https://pypi.org/project/trysac-model-router/)
[![License: MIT](https://img.shields.io/pypi/l/trysac-model-router)](LICENSE)
[![CI](https://github.com/Trysac/u006-model-router/actions/workflows/ci.yml/badge.svg)](https://github.com/Trysac/u006-model-router/actions/workflows/ci.yml)

Choose an LLM by cost, latency and quality rules over caller-supplied model profiles, with capability filters, explained exclusions and ordered fallback

## Problem

Applications hard-code one model and overpay for trivial requests, or break when that provider
has an outage. Routing rules scattered across if-statements cannot be tested or explained. This
library takes model profiles as plain data, returns a deterministic, explained candidate order
for a request, and runs it with ordered fallback through a small LLM port.

Compared with LiteLLM Router or RouteLLM: those need a proxy or a trained router; this is
explainable rule-based routing over plain data, with no dependencies.

## Install

```sh
pip install trysac-model-router
# or
uv add trysac-model-router
```

Requires Python 3.10 or newer. No runtime dependencies.

## Usage

The library ships no prices or latencies. Profiles are your data; the figures below are
illustrative only (example values modelled on vendor pricing pages, retrieved 2026-11-05).
Refresh them from your providers' current pricing and your own latency measurements before use.

```python
from decimal import Decimal

from model_router import ModelProfile, RecordedLLM, RouteRequest, route, run

# Illustrative figures, retrieved 2026-11-05; refresh before use.
profiles = [
    ModelProfile(
        id="small", provider="acme", input_per_mtok=Decimal("0.15"),
        output_per_mtok=Decimal("0.60"), p50_latency_ms=400, quality=0.6,
        context_window=128_000, capabilities=frozenset({"tools"}),
    ),
    ModelProfile(
        id="large", provider="acme", input_per_mtok=Decimal("3"),
        output_per_mtok=Decimal("15"), p50_latency_ms=1200, quality=0.95,
        context_window=200_000, capabilities=frozenset({"tools", "vision"}),
    ),
]

request = RouteRequest(
    input_tokens=2000, expected_output_tokens=300,
    required_capabilities=frozenset({"tools"}),
)
decision = route(profiles, request, policy="balanced")
print([m.id for m in decision.candidates])
print(dict(decision.excluded))
print(dict(decision.scores))

ports = {"small": RecordedLLM(["hello"]), "large": RecordedLLM(["hello"])}
result = run(decision, ports, [{"role": "user", "content": "Hi"}], max_tokens=50)
print(result.model_id, result.output, result.attempts)
```

If a model fails with `RetryableError`, `TimeoutError` or `ConnectionError`, `run` falls back to
the next candidate; any other exception propagates unchanged.

## Documentation

- [Architecture](docs/ARCHITECTURE.md) and [decisions](docs/decisions/)
- [Testing](docs/TESTING.md) and [security](docs/SECURITY.md)
- [Project record](docs/PROJECT.md) and [stack](docs/STACK.md)
- [Changelog](CHANGELOG.md) and [contributing guide](CONTRIBUTING.md)

## License

[MIT](LICENSE) (c) 2026 Miguel Gomez
