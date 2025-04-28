from __future__ import annotations

from llm_bridge.core.abc import AbstractLLMClient
from llm_bridge.core.types import GenerationParams, Message, Role
from llm_bridge.registry.client_factory import LLMClientFactory
from llm_bridge.registry.provider_registry import ProviderRegistry


class DummyAdapter(AbstractLLMClient):
    """テスト用の簡易アダプタ."""

    def __init__(self, model: str, *, greeting: str = 'hello', **_: object) -> None:
        super().__init__(model)
        self._greeting: str = greeting

    def _invoke(self, messages: list[Message], params: GenerationParams) -> str:  # noqa: ARG002
        return self._greeting


def test_initialize_client_and_generate() -> None:
    # アダプタをレジストリへ登録
    ProviderRegistry().register('dummyprov', DummyAdapter)

    client: DummyAdapter = LLMClientFactory.initialize_client(
        'dummyprov:my-model',
        greeting='hiya',  # 任意 kwargs がアダプタに渡るか検証
        retry_strategy=None,
    )
    assert isinstance(client, DummyAdapter)

    out: str = client.generate(
        [Message(role=Role.user, content='ping')],
        GenerationParams(),
    )
    assert out == 'hiya'
