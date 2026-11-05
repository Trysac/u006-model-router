# model-router

[![PyPI version](https://img.shields.io/pypi/v/trysac-model-router)](https://pypi.org/project/trysac-model-router/)
[![License: MIT](https://img.shields.io/pypi/l/trysac-model-router)](LICENSE)
[![CI](https://github.com/Trysac/u006-model-router/actions/workflows/ci.yml/badge.svg)](https://github.com/Trysac/u006-model-router/actions/workflows/ci.yml)

Choose an LLM by cost, latency and quality rules over caller-supplied model profiles, with capability filters, explained exclusions and ordered fallback

## Problem

Describe in two or three sentences the concrete problem this utility solves, who hits it, and what
they do today without it. Replace this paragraph when the first real feature lands.

## Install

```sh
pip install trysac-model-router
# or
uv add trysac-model-router
```

Requires Python 3.10 or newer. No runtime dependencies.

## Usage

```python
import model_router

# normalize() trims surrounding whitespace and lower-cases the text.
print(model_router.normalize("  Hello World "))  # -> "hello world"

# Blank input is rejected explicitly instead of returning an empty string.
try:
    model_router.normalize("   ")
except ValueError as exc:
    print(f"rejected: {exc}")
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) and [decisions](docs/decisions/)
- [Testing](docs/TESTING.md) and [security](docs/SECURITY.md)
- [Project record](docs/PROJECT.md) and [stack](docs/STACK.md)
- [Changelog](CHANGELOG.md) and [contributing guide](CONTRIBUTING.md)

## License

[MIT](LICENSE) (c) 2026 Miguel Gomez
