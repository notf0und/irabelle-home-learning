"""Run the sentence-file work off the event loop, one job at a time.

Reading, globbing and writing `custom_sentences/` blocks, so Home Assistant wants it in the
executor. The files are read, changed and written back, though, and two of those overlapping
in executor threads would lose one of the changes - which running them on the event loop used
to rule out by running them in turn. The lock keeps that: one file job at a time.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from functools import partial
from typing import Any, TypeVar

from homeassistant.core import HomeAssistant
from homeassistant.util.hass_dict import HassKey

_T = TypeVar("_T")

_LOCK: HassKey[asyncio.Lock] = HassKey("irabelle_home_learning_file_lock")


async def async_run(
    hass: HomeAssistant, func: Callable[..., _T], /, *args: Any, **kwargs: Any
) -> _T:
    """Run blocking file work in the executor, after any file job already running."""
    lock = hass.data.setdefault(_LOCK, asyncio.Lock())
    async with lock:
        return await hass.async_add_executor_job(partial(func, *args, **kwargs))
