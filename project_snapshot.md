# Project Snapshot for `/Users/daikitsutsumi/llm_bridge/src`

## Directory Tree
```
src/
└── llm_bridge
    ├── adapters
    │   ├── __init__.py
    │   └── openai_adapter.py
    ├── core
    │   ├── __init__.py
    │   ├── abc.py
    │   ├── exceptions.py
    │   ├── model_id.py
    │   ├── retry.py
    │   └── types.py
    ├── registry
    │   ├── __init__.py
    │   ├── client_factory.py
    │   └── provider_registry.py
    └── __init__.py
```

### llm_bridge/__init__.py

```python

```

### llm_bridge/adapters/__init__.py

```python

```

### llm_bridge/adapters/openai_adapter.py

```python
"""adapters.openai_adapter

Concrete adapter that bridges :class:`llm_bridge.core.abc.AbstractLLMClient`
with the **OpenAI Chat Completions** HTTP API.

This implementation targets *openai==1.x* (the new "unified" client).  For the
older 0.x series, please update the `_invoke` call-sites.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import openai
from dotenv import load_dotenv

from llm_bridge.core.abc import AbstractLLMClient
from llm_bridge.core.exceptions import (
    LLMClientError,
    ModelNotFoundError,
    RateLimitExceededError,
)
from llm_bridge.registry.provider_registry import provider_registry

if TYPE_CHECKING:
    from llm_bridge.core.types import GenerationParams, Message

load_dotenv()
# ---------------------------------------------------------------------------
# Adapter implementation
# ---------------------------------------------------------------------------


class OpenAIAdapter(AbstractLLMClient):
    """Adapter for OpenAI ChatCompletion API."""

    # NOTE: The OpenAI Python client automatically picks up `OPENAI_API_KEY`.

    def __init__(self, model: str) -> None:
        super().__init__(model)
        self._client = openai.Client(api_key=os.getenv('OPENAI_API_KEY'))

    # ------------------------------------------------------------------
    # Synchronous path
    # ------------------------------------------------------------------

    def _invoke(self, messages: list[Message], params: GenerationParams) -> str:
        try:
            response = self._client.responses.create(
                model=self._model,
                input=messages,  # type: ignore[arg-type]
                temperature=params.temperature,
                top_p=params.top_p,
                max_output_tokens=params.max_tokens,
                **params.model_dump(exclude={'temperature', 'top_p', 'max_tokens'}),
            )
        except openai.RateLimitError as exc:
            raise RateLimitExceededError('Rate limit exceeded') from exc
        except openai.NotFoundError as exc:
            raise ModelNotFoundError('Unknown model for provider') from exc
        except openai.OpenAIError as exc:  # generic fallback
            raise LLMClientError('Upstream provider error') from exc

        return response.output_text


# ---------------------------------------------------------------------------
# Automatic registration
# ---------------------------------------------------------------------------

provider_registry.register('openai', OpenAIAdapter)

```

### llm_bridge/core/__init__.py

```python

```

### llm_bridge/core/abc.py

```python
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

```

### llm_bridge/core/exceptions.py

```python
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

```

### llm_bridge/core/model_id.py

```python
"""core.model_id

Utility for validating and parsing LLM model identifiers of the canonical form

    "<provider>:<model_name>"

This module is intentionally free of external dependencies (apart from Pydantic)
so that it can live in the **core** domain layer and be imported by any other
layer without causing circular-import issues.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Regular-expression helpers
# ---------------------------------------------------------------------------

_MODEL_ID_REGEX: re.Pattern[str] = re.compile(
    r'^(?P<provider>[a-z0-9_-]+):(?P<model>[a-z0-9_.-]+)$',
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class ModelId(BaseModel):
    """Value-object representing a model identifier.

    * `provider` … external service slug (e.g. ``openai``)
    * `model` … concrete model name (e.g. ``gpt-4o``)

    The *raw* string is preserved for logging/debugging purposes.
    """

    provider: str = Field(..., pattern=r'^[a-z0-9_-]+$', description='provider slug')
    model: str = Field(..., pattern=r'^[a-z0-9_.-]+$', description='model name')
    raw: str = Field(..., description='original, unmodified identifier')

    model_config = {
        'frozen': True,  # hashable / usable as dict key
        'str_strip_whitespace': True,
    }
    # --------------------------- Validators ---------------------------

    @field_validator('provider', mode='before')
    @classmethod
    def _provider_to_lower(cls, v: str) -> str:
        """Force lower-case for case-insensitive matching."""
        return v.lower()

    @field_validator('model', mode='before')
    @classmethod
    def _model_to_lower(cls, v: str) -> str:
        return v.lower()

    # --------------------------- Constructors -------------------------

    @classmethod
    def parse(cls, raw: str) -> ModelId:
        """Parse and validate a *raw* identifier string.

        >>> ModelId.parse("openai:gpt-4o")
        ModelId(provider='openai', model='gpt-4o')
        """
        if (m := _MODEL_ID_REGEX.match(raw.strip())) is None:
            raise ValueError(f"Invalid ModelId format. Expected '<provider>:<model>', got: {raw}")
        return cls(provider=m.group('provider'), model=m.group('model'), raw=raw)

    # --------------------------- Dunder helpers -----------------------

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f'{self.provider}:{self.model}'


# Convenience alias so callers don't need to import the class explicitly
parse_model_id = ModelId.parse

```

### llm_bridge/core/retry.py

```python
"""core.retry

Reusable retry utilities with exponential back-off + optional jitter.
Designed to run in the **core** layer and depends only on Python stdlib + Pydantic.
"""

from __future__ import annotations

import functools
import secrets
import time
from typing import TYPE_CHECKING, NoReturn, ParamSpec, TypeVar

from pydantic import BaseModel, Field

from llm_bridge.core.exceptions import GenerationTimeoutError, RateLimitExceededError

if TYPE_CHECKING:
    from collections.abc import Callable

P = ParamSpec('P')
T = TypeVar('T')


class RetryStrategy(BaseModel):
    """Configuration for exponential back-off retry with optional jitter."""

    max_attempts: int = Field(default=5, ge=1, description='Total attempts including the first call')
    base_backoff_sec: float = Field(default=1.0, ge=0.0, description='Initial delay before first retry (seconds)')
    max_backoff_sec: float = Field(default=60.0, ge=0.0, description='Upper bound for any sleep interval')
    jitter: bool = Field(default=True, description='Add random jitter (0-1s) to each interval')

    model_config = {
        'frozen': True,
        'use_enum_values': True,
    }

    def compute_delay(self, attempt_number: int) -> float:
        """Calculate sleep duration for the given attempt number (1-indexed)."""
        # Calculate exponential backoff with upper bound
        delay = min(self.base_backoff_sec * (2 ** (attempt_number - 1)), self.max_backoff_sec)

        # Add random jitter if enabled
        if self.jitter:
            # Add random value between 0 and 1
            delay += secrets.randbelow(101) / 100

        return delay


def with_retry(  # noqa: C901
    strategy: RetryStrategy | None = None,
    retry_on: tuple[type[Exception], ...] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Retry decorator that retries a function according to the specified strategy.

    Parameters
    ----------
    strategy
        Retry policy. Defaults to RetryStrategy() if None.
    retry_on
        Exception types that trigger a retry. Defaults to
        (RateLimitExceededError, GenerationTimeoutError, ConnectionError).

    """
    retry_strategy = strategy or RetryStrategy()
    retry_exceptions = retry_on or (RateLimitExceededError, GenerationTimeoutError, ConnectionError)

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            def raise_timeout(message: str) -> NoReturn:
                raise GenerationTimeoutError(message)

            for attempt_number in range(1, retry_strategy.max_attempts + 1):
                try:
                    result = func(*args, **kwargs)
                except retry_exceptions:
                    if attempt_number == retry_strategy.max_attempts:
                        raise_timeout('Retry limit exceeded')
                    # Calculate and apply backoff delay before next attempt
                    time.sleep(retry_strategy.compute_delay(attempt_number))
                    continue
                except Exception:
                    # Any other exceptions bubble up immediately
                    raise
                else:
                    # No exceptions occurred, check the result
                    if result is None:
                        # Handle None result outside of try block
                        if attempt_number == retry_strategy.max_attempts:
                            raise_timeout('Retry limit exceeded due to None response')
                        time.sleep(retry_strategy.compute_delay(attempt_number))
                        continue
                    return result

            # This is technically unreachable, but added for explicit return
            return raise_timeout('Retry limit exceeded (sync)')

        return wrapper

    return decorator

```

### llm_bridge/core/types.py

```python
"""core.types

Shared DTOs and enums used throughout *llm_bridge*.

These models live in the **core** layer so that *adapters*, *registry*, and
higher application layers can depend on them without causing circular imports.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Chat roles (OpenAI-style for broad compatibility)
# ---------------------------------------------------------------------------


class Role(StrEnum):
    system = 'system'
    user = 'user'
    assistant = 'assistant'


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


class Message(BaseModel):
    """Single chat message."""

    role: Role
    content: str

    # Immutable value-object
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)


# ---------------------------------------------------------------------------
# Generation parameters (provider-agnostic)
#   • Extra fields are permitted so callers can pass provider-specific knobs
# ---------------------------------------------------------------------------


class GenerationParams(BaseModel):
    """Temperature, top-p, etc. Extra params are allowed."""

    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    max_tokens: int = Field(1024, ge=1, description='Maximum tokens in completion')

    # Allow provider-specific parameters without schema errors
    model_config = ConfigDict(extra='allow')

```

### llm_bridge/registry/__init__.py

```python

```

### llm_bridge/registry/client_factory.py

```python
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

```

### llm_bridge/registry/provider_registry.py

```python
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

```
