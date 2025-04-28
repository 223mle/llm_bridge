"""core.abc

Abstract base class that *all* provider adapters must implement.

Design goals
============
1. **Provider-agnostic public API** - callers interact exclusively via
    `generate()` passing domain models (`Message`, `GenerationParams`).
    They never touch provider-specific payloads.
2. **Built-in retry** - the public methods are wrapped in the `with_retry()`
    decorator so every adapter inherits robust back-off behaviour by default.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from llm_bridge.core.retry import RetryStrategy, with_retry

if TYPE_CHECKING:
    from llm_bridge.core.types import GenerationParams, Message


class AbstractLLMClient(ABC):
    """Provider-independent LLM client interface."""

    # ---------------------------------------------------------------------
    # Construction
    # ---------------------------------------------------------------------

    def __init__(
        self,
        model: str,
        *,
        retry_strategy: RetryStrategy | None = None,
    ) -> None:
        """Store *model* name and optional strategy / adapter-specific options."""
        self._model: str = model
        self._retry_strategy: RetryStrategy = retry_strategy or RetryStrategy(
            max_attempts=5,
            base_backoff_sec=1.0,
            max_backoff_sec=60.0,
            jitter=True,
        )

    # ------------------------------------------------------------------
    # Public synchronous API
    # ------------------------------------------------------------------

    def generate(
        self,
        messages: list[Message],
        params: GenerationParams,
    ) -> str:
        """Generate a completion synchronously.

        Subclasses **must not** override this - override `_invoke()` instead.
        """
        bound_params = params

        @with_retry(self._retry_strategy)
        def _call() -> str:  # inner closure captures args
            return self._invoke(messages, bound_params)  # type: ignore[return-value]

        return _call()

    # ------------------------------------------------------------------
    # Methods to implement in concrete adapters
    # ------------------------------------------------------------------

    @abstractmethod
    def _invoke(self, messages: list[Message], params: GenerationParams) -> str | None:
        """Provider-specific **blocking** implementation (to be overridden)."""

    # ------------------------------------------------------------------
    # Helper - string representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f'<{self.__class__.__name__} model={self._model!r}>'
