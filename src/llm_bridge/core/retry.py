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
