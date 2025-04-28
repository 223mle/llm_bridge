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
