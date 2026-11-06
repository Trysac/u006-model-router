# Routing rules

`model_router` chooses among caller-supplied model profiles with plain, explainable rules.
It ships no prices, latencies or quality scores: profiles are caller data, so record where
the numbers came from and refresh them when a provider changes its pricing.

## Estimated cost

`input_tokens * input_per_mtok / 10^6 + expected_output_tokens * output_per_mtok / 10^6`,
computed in `Decimal`.

## Exclusion reasons

Checked in this order, the first match wins. Every excluded profile appears in
`RouteDecision.excluded` (or `NoEligibleModelError.excluded` when nothing is left).

| Order | Reason code | Meaning |
|---|---|---|
| 1 | `missing_capability:<name>` | A required capability is missing; the alphabetically first one is reported. |
| 2 | `context_window` | `input_tokens + expected_output_tokens` exceeds the context window. |
| 3 | `min_quality` | Quality is below `min_quality`. |
| 4 | `max_cost` | Estimated cost is above `max_cost`. |
| 5 | `max_latency` | Median latency is above `max_latency_ms`. |

Limits are inclusive: a value equal to the bound is still eligible.

## Policies

- `cheapest`: lowest estimated cost first.
- `fastest`: lowest `p50_latency_ms` first.
- `best`: highest quality first.
- `balanced`: highest `w_q * q_norm + w_c * (1 - c_norm) + w_l * (1 - l_norm)` first.

Normalization is min-max over the eligible models (cost is normalized in `Decimal`, then
converted to `float`); a norm is 0 when the minimum equals the maximum. Weights are ordered
`(w_q, w_c, w_l)`, must be non-negative and are divided by their sum. Ties always break by
id ascending, so the order never depends on the order of the input profiles.

`RouteDecision.scores` holds a higher-is-better value for each candidate: `1 - c_norm` for
`cheapest`, `1 - l_norm` for `fastest`, `q_norm` for `best` and the weighted score for
`balanced`.

## Running with fallback

`run` calls the first `max_candidates` candidates in order through the `LLMPort`. Only
exceptions listed in `retry_on` (default: `RetryableError`, `TimeoutError`,
`ConnectionError`) move on to the next candidate; anything else propagates unchanged.
Every selected candidate needs a port, checked before any call. Attempts and events store
exception type names only, never exception text or message content.

## Input limits

| Input | Limit |
|---|---|
| Profiles per `route` call | 1000 |
| Model id | `[A-Za-z0-9._:/-]{1,128}` |
| Capability names | at most 256 per set, 1 to 128 characters each |
| Token counts and context windows | at most 10^12 |
| Messages per `run` call | 1000 |
| Characters per message / in total | 1,000,000 / 4,000,000 |
| Recorded calls kept by the test doubles | 1000 |

Backoff, timeouts, learned routing, live latency measurement and vendor SDKs are out of scope.
