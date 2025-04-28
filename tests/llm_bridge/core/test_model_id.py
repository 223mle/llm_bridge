from __future__ import annotations

import pytest

from llm_bridge.core.model_id import ModelId, parse_model_id


def test_valid_parse_and_str() -> None:
    mid: ModelId = ModelId.parse('OpenAI:GPT-4o')
    assert mid.provider == 'openai'
    assert mid.model == 'gpt-4o'
    assert mid.raw == 'OpenAI:GPT-4o'
    assert str(mid) == 'openai:gpt-4o'


@pytest.mark.parametrize('bad_id', ['openai', 'openai-gpt4o', ':', 'openai:'])
def test_invalid_parse(bad_id: str) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        ModelId.parse(bad_id)


def test_function_alias() -> None:
    assert isinstance(parse_model_id('anthropic:claude-3-5-sonnet'), ModelId)
