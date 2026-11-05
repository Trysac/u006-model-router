# Project record

## Objective

Choose an LLM by cost, latency and quality rules over caller-supplied model profiles, with capability filters, explained exclusions and ordered fallback

## Scope

- In scope: the public API documented in [README.md](../README.md), typed and tested.
- Out of scope: anything that needs runtime dependencies or I/O without an accepted ADR.

## Timeline

| Date | Event |
|---|---|
| 2026 | Repository created from the factory utility template. |

## Effort metrics

Measured values only; a value that has not been measured stays "pending measurement".

| Metric | Value |
|---|---|
| Tokens (input / output) | 570011 / 92245 |
| Hours (human) | pending measurement |
| Cost (USD) | 0.8380 |

## Factory run u006 (measured from API usage)

| Phase | Model | Batch | Input | Output | Cache write | Cache read | Cost (USD) |
|---|---|---|---|---|---|---|---|
| generation | claude-sonnet-5-5 | yes | 2552 | 72593 | 0 | 4246 | 0.3659 |
| fix-1 | claude-sonnet-5-5 | no | 8 | 1405 | 18459 | 66322 | 0.0668 |
| fix-2 | claude-sonnet-5-5 | no | 8 | 1425 | 19145 | 66834 | 0.0688 |
| audit | claude-sonnet-5-5 | no | 14 | 3706 | 42573 | 196968 | 0.1632 |
| fix-3 | claude-sonnet-5-5 | no | 10 | 2562 | 11809 | 83399 | 0.0635 |
| fix-4 | claude-sonnet-5-5 | no | 6 | 851 | 7435 | 41172 | 0.0312 |
| docs | claude-haiku-5-5 | no | 108472 | 9703 | 0 | 0 | 0.0785 |

Total: $0.8380. Prices: Anthropic list prices.
