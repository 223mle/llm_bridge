"""registry.provider_registry

Global registry that maps provider slugs (e.g. "openai") to their
concrete Adapter classes (subclasses of AbstractLLMClient).

The registry is intentionally implemented as a pure domain helper—no external
SDK imports—so that it can be imported freely by adapters without risk of
circular-dependency explosions.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Generic, TypeVar

from llm_bridge.core.exceptions import ProviderNotFoundError

if TYPE_CHECKING:
    from collections.abc import Mapping, MutableMapping

    from llm_bridge.core.abc import AbstractLLMClient

# Type variable for better type annotations
ClientT = TypeVar('ClientT', bound='AbstractLLMClient')


class _ThreadSafeSingleton(type):
    """Metaclass ensuring a single registry instance across threads."""

    _instance: ProviderRegistry | None = None
    _lock = threading.Lock()

    def __call__(cls, *args: object, **kwargs: object) -> ProviderRegistry:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__call__(*args, **kwargs)
        return cls._instance


class ProviderRegistry(Generic[ClientT], metaclass=_ThreadSafeSingleton):
    """Centralised look-up and registration for provider → adapter mappings.

    Usage (typically inside adapter modules):

    ```python
    from llm_bridge.registry import ProviderRegistry

    class OpenAIAdapter(AbstractLLMClient):
        ...

    ProviderRegistry.register("openai", OpenAIAdapter)
    ```
    """

    _registry: MutableMapping[str, type[ClientT]]

    def __init__(self) -> None:  # pragma: no cover - called once via singleton
        self._registry = {}

    def register(self, provider_key: str, adapter_cls: type[ClientT]) -> None:
        """Register adapter_cls under provider_key.

        Parameters
        ----------
        provider_key
            Slug such as "openai" or "anthropic". Normalised to lower-case.
        adapter_cls
            Concrete subclass implementing AbstractLLMClient.

        """
        key = provider_key.lower()
        # Late import to avoid heavy SDKs at module import-time
        from llm_bridge.core.abc import AbstractLLMClient  # local import avoids cycles

        if not issubclass(adapter_cls, AbstractLLMClient):
            raise TypeError('adapter_cls must subclass AbstractLLMClient')
        self._registry[key] = adapter_cls

    def get_adapter_cls(self, provider_key: str) -> type[ClientT]:
        """Return the adapter class registered for provider_key.

        Raises
        ------
        ProviderNotFoundError
            If provider_key hasn't been registered.

        """
        key = provider_key.lower()
        try:
            return self._registry[key]
        except KeyError as exc:
            raise ProviderNotFoundError(f'Unsupported provider: {provider_key}') from exc

    def available_providers(self) -> list[str]:
        """Return a sorted list of registered providers (for introspection)."""
        return sorted(self._registry)

    # Expose mapping as *read-only* to outsiders (defensive copy)
    def mapping(self) -> Mapping[str, type[ClientT]]:
        """Return a read-only copy of the provider registry mapping."""
        return dict(self._registry)


# Re-export a module-level instance for ergonomic usage
provider_registry: ProviderRegistry = ProviderRegistry()
