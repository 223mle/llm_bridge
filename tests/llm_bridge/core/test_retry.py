import time

import pytest

from llm_bridge.core.exceptions import (
    GenerationTimeoutError,
    RateLimitExceededError,
)
from llm_bridge.core.retry import RetryStrategy, with_retry


def test_compute_delay_without_jitter() -> None:
    strategy = RetryStrategy(base_backoff_sec=1.0, jitter=False)
    assert strategy.compute_delay(1) == 1.0  # 1 * 2^(1-1)
    assert strategy.compute_delay(2) == 2.0  # 1 * 2^(2-1)  # noqa: PLR2004
    # 上限チェック
    assert strategy.compute_delay(10) == strategy.max_backoff_sec


def test_wrapper_success_first_try() -> None:
    calls = {'cnt': 0}

    @with_retry(RetryStrategy(max_attempts=3, base_backoff_sec=0, jitter=False))
    def _fn() -> str:
        calls['cnt'] += 1
        return 'ok'

    assert _fn() == 'ok'
    assert calls['cnt'] == 1


def test_wrapper_eventual_success(monkeypatch: pytest.MonkeyPatch) -> None:
    # time.sleep をスタブ化して高速化
    monkeypatch.setattr(time, 'sleep', lambda *_: None)
    calls = {'cnt': 0}

    @with_retry(RetryStrategy(max_attempts=3, base_backoff_sec=0, jitter=False))
    def _fn() -> str:
        calls['cnt'] += 1
        if calls['cnt'] < 3:  # noqa: PLR2004
            raise RateLimitExceededError('busy')
        return 'done'

    assert _fn() == 'done'
    assert calls['cnt'] == 3  # noqa: PLR2004


def test_wrapper_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, 'sleep', lambda *_: None)

    @with_retry(RetryStrategy(max_attempts=2, base_backoff_sec=0, jitter=False))
    def _always_fail() -> None:
        raise RateLimitExceededError('still busy')

    with pytest.raises(GenerationTimeoutError):
        _always_fail()
