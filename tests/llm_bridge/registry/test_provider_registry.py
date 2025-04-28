import pytest

from llm_bridge.core.abc import AbstractLLMClient
from llm_bridge.core.exceptions import ProviderNotFoundError
from llm_bridge.registry.provider_registry import ProviderRegistry


class DummyAdapter(AbstractLLMClient):
    def _invoke(self, messages: list[dict], params: dict) -> str:  # noqa: ARG002
        return 'dummy'


def test_register_and_fetch() -> None:
    reg = ProviderRegistry()
    reg.register('dummy', DummyAdapter)
    assert 'dummy' in reg.available_providers()
    assert reg.get_adapter_cls('dummy') is DummyAdapter


def test_register_type_validation() -> None:
    reg = ProviderRegistry()
    with pytest.raises(TypeError):
        reg.register('bad', object)  # type: ignore[arg-type]


def test_unknown_provider() -> None:
    reg = ProviderRegistry()
    with pytest.raises(ProviderNotFoundError):
        reg.get_adapter_cls('no-such')
