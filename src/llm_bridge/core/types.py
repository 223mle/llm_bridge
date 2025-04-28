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
