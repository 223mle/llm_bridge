"""core.exceptions

Centralised exception hierarchy for *llm_bridge*.

Each error carries an `http_status` attribute so that upper layers (REST API
controllers, FastAPI exception handlers, etc.) can translate exceptions to
appropriate HTTP responses *without* scattering status-code logic throughout
business code.
"""

from __future__ import annotations

from http import HTTPStatus
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from collections.abc import Mapping


# ---------------------------------------------------------------------------
# Base mixin with HTTP status information
# ---------------------------------------------------------------------------


class LLMBridgeError(Exception):
    """Base class for all *llm_bridge* domain errors."""

    #: Default HTTP status if not overridden by subclass.
    http_status: ClassVar[HTTPStatus] = HTTPStatus.INTERNAL_SERVER_ERROR

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.__class__.__name__)

    def to_json(self) -> dict[str, dict[str, str]]:
        """Unified error body as defined in the requirements."""
        return {'error': {'type': self.__class__.__name__, 'message': str(self)}}


# ---------------------------------------------------------------------------
# Concrete error classes
# ---------------------------------------------------------------------------


class ProviderNotFoundError(LLMBridgeError):
    """Raised when `ProviderRegistry` cannot find a requested provider key."""

    http_status: ClassVar[HTTPStatus] = HTTPStatus.NOT_IMPLEMENTED  # 501


class ModelNotFoundError(LLMBridgeError):
    """Raised when a model is unknown for a valid provider."""

    http_status: ClassVar[HTTPStatus] = HTTPStatus.BAD_REQUEST  # 400


class RateLimitExceededError(LLMBridgeError):
    """Raised when provider rate limits persist beyond retry strategy."""

    http_status: ClassVar[HTTPStatus] = HTTPStatus.TOO_MANY_REQUESTS  # 429


class LLMClientError(LLMBridgeError):
    """Generic upstream provider error (e.g., unexpected 5xx)."""

    http_status: ClassVar[HTTPStatus] = HTTPStatus.BAD_GATEWAY  # 502


class GenerationTimeoutError(LLMBridgeError):
    """Raised when retry attempts exceed maximum backoff window."""

    http_status: ClassVar[HTTPStatus] = HTTPStatus.GATEWAY_TIMEOUT  # 504


HTTP_STATUS_MAP: Mapping[type[LLMBridgeError], HTTPStatus] = {
    ProviderNotFoundError: ProviderNotFoundError.http_status,
    ModelNotFoundError: ModelNotFoundError.http_status,
    RateLimitExceededError: RateLimitExceededError.http_status,
    LLMClientError: LLMClientError.http_status,
    GenerationTimeoutError: GenerationTimeoutError.http_status,
}
