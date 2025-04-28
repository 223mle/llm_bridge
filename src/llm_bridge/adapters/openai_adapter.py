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
