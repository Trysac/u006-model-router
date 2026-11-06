# Rule-based routing over caller-supplied profiles with ordered fallback

Status: accepted (generated, pending human review)

## Context

Applications hard-code one model or scatter routing if statements that cannot be tested or explained. Existing routers need a proxy (LiteLLM Router) or a trained model (RouteLLM). The library must give a deterministic, explainable choice from plain data and run it with fallback, without shipping prices that go stale.

## Decision

Model profiles are caller data. route() applies fixed exclusion rules in a documented order, with the first match winning and a stable reason code, and then orders the eligible models by a policy using min-max normalization, with ties broken by id. Cost is computed in Decimal. run() walks the candidates through the canonical LLMPort and only moves on for exceptions in retry_on. Attempts and events record exception type names only, never exception text. Inputs are validated at construction and call time with named size limits. Errors share one base class with stable codes, and logging uses a NullHandler and never includes input text.

## Consequences

Routing is reproducible, independent of input order, and every exclusion is explained, so it is easy to test and audit. Quality and latency are static numbers the caller must keep fresh, and there is no learned or live adaptation. Because errors carry only type names, debugging a failed attempt needs the caller's own logging of the exception. Scores of the non-balanced policies are a documented single-criterion value, not a comparable cross-policy measure.
