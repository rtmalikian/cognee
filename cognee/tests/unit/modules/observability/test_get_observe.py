import asyncio
import inspect
import warnings
from unittest.mock import patch

import pytest

from cognee.modules.observability.exceptions import UnsupportedObserverError
from cognee.modules.observability.get_observe import _wrap_with_otel, get_observe
from cognee.modules.observability.observers import Observer


def test_get_observe_raises_for_unsupported_observer():
    """Unsupported observer (e.g. LLMLITE, LANGSMITH) raises UnsupportedObserverError."""
    with patch("cognee.modules.observability.get_observe.get_base_config") as get_config:
        get_config.return_value = type("Config", (), {"monitoring_tool": Observer.LLMLITE})()

        with pytest.raises(UnsupportedObserverError, match="Unsupported observer"):
            get_observe()


def test_observe_keeps_async_wrapper_for_coroutine_functions():
    """The deprecated ``asyncio.iscoroutinefunction`` was replaced with
    ``inspect.iscoroutinefunction``; wrapper selection must be unchanged and
    must not emit a DeprecationWarning."""
    observe = _wrap_with_otel(lambda func: func)

    async def my_coro():
        return 1

    def my_sync():
        return 2

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        wrapped_coro = observe(my_coro)
        wrapped_sync = observe(my_sync)

    assert inspect.iscoroutinefunction(wrapped_coro)
    assert not inspect.iscoroutinefunction(wrapped_sync)
    assert asyncio.run(wrapped_coro()) == 1
    assert wrapped_sync() == 2


def test_observe_with_as_type_keeps_async_wrapper():
    """Same cover for the parameterised ``@observe(as_type=...)`` path."""
    observe = _wrap_with_otel(lambda *args, **kwargs: lambda func: func)

    async def my_coro():
        return 3

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        wrapped = observe(as_type="generation")(my_coro)

    assert inspect.iscoroutinefunction(wrapped)
    assert asyncio.run(wrapped()) == 3
