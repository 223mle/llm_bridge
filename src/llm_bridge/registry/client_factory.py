"""registry.client_factory

Factory responsible for converting a ModelId (or raw string) into a fully
initialized adapter instance (subclass of AbstractLLMClient).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, overload

from llm_bridge.core.model_id import ModelId, parse_model_id
from llm_bridge.registry.provider_registry import provider_registry

if TYPE_CHECKING:
    from llm_bridge.core.abc import AbstractLLMClient
    from llm_bridge.core.retry import RetryStrategy

# Type variable for better type annotations
ClientT = TypeVar('ClientT', bound='AbstractLLMClient')


class LLMClientFactory:
    """Factory for creating provider-specific LLM clients.

    This class is stateless; all information resides in provider_registry.
    An explicit class is provided rather than a bare function to maintain
    a symmetrical API with future extensions (e.g. pooled client caching).
    """

    @staticmethod
    @overload
    def initialize_client(model_id: str, **adapter_kwargs: RetryStrategy | None) -> AbstractLLMClient: ...

    @staticmethod
    @overload
    def initialize_client(model_id: ModelId, **adapter_kwargs: RetryStrategy | None) -> AbstractLLMClient: ...

    @staticmethod
    def initialize_client(model_id: str | ModelId, **adapter_kwargs: RetryStrategy | None) -> AbstractLLMClient:
        """Return a concrete adapter for model_id.

        Parameters
        ----------
        model_id
            Either a raw string ("provider:model") or a pre-parsed
            ModelId instance.
        **adapter_kwargs
            Arbitrary keyword arguments forwarded to the adapter's
            constructor. This allows callers to specify timeouts,
            custom HTTP sessions, etc. without changing the factory signature.

        """
        # Normalize input: convert string to ModelId if needed
        model_identifier = parse_model_id(model_id) if isinstance(model_id, str) else model_id

        # Look up the appropriate adapter class for this provider
        adapter_class = provider_registry.get_adapter_cls(model_identifier.provider)

        # Instantiate and return the adapter with the specified model and options
        return adapter_class(model=model_identifier.model, **adapter_kwargs)
