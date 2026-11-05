"""Explainable, rule-based LLM model routing with ordered fallback."""

from __future__ import annotations

import logging

from .errors import AllModelsFailedError, NoEligibleModelError, RetryableError, RouterError
from .llm import FakeLLM, LLMPort, Message, RecordedLLM, RecordingExhaustedError
from .models import ModelProfile, RouteDecision, RouteRequest
from .router import route
from .runner import RunResult, run

__all__ = [
    "AllModelsFailedError",
    "FakeLLM",
    "LLMPort",
    "Message",
    "ModelProfile",
    "NoEligibleModelError",
    "RecordedLLM",
    "RecordingExhaustedError",
    "RetryableError",
    "RouteDecision",
    "RouteRequest",
    "RouterError",
    "RunResult",
    "__version__",
    "route",
    "run",
]

__version__ = "0.1.0"

logging.getLogger(__name__).addHandler(logging.NullHandler())
