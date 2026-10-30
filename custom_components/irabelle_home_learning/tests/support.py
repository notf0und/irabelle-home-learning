"""Shared helpers for the tests.

These run **inside the Home Assistant container**, against the real Home Assistant the
component will meet::

    scripts/test.sh                             # from the repo, via docker
    docker exec homeassistant python3 /config/custom_components/irabelle_home_learning/tests/run.py

No pytest and no test harness are involved. A real `HomeAssistant` object is cheap to build
and the component's own code is the thing under test, so the standard library is enough.
That matters more than usual here: the facts worth pinning down are Home Assistant's own
behaviour — what hassil matches, what Jinja scopes, what the template engine renders — and a
mock would answer the wrong question.
"""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest
from typing import Any

#: The container mounts the repository at /config, so the integration imports as a package
#: from there. `custom_components` has no `__init__.py`; namespace packages cover that.
#: Whatever holds `custom_components/` - /config in the container, the repository root
#: in a checkout.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from homeassistant.core import HomeAssistant  # noqa: E402


class FakeEntry:
    """The parts of a ConfigEntry the learner reads.

    A real ConfigEntry needs Home Assistant's config-entry machinery to exist at all, and
    every field this component touches is a plain value, so a stub is honest here.
    """

    def __init__(self, entry_id: str = "test_entry", **data: Any) -> None:
        self.entry_id = entry_id
        self.data: dict[str, Any] = dict(data)
        self.options: dict[str, Any] = {}
        self.runtime_data: Any = None
        self.state = None


async def make_hass() -> HomeAssistant:
    """A started Home Assistant with a temp config dir and stub reload services.

    The learner reloads the conversation integration after every write. That integration is
    not set up here, so a no-op service is registered instead — the write paths do not care
    what the service does, only that calling it does not raise. `custom_sentences/` lives
    under the temp config dir, so every file test is isolated.
    """
    hass = HomeAssistant(tempfile.mkdtemp(prefix="irabelle_home_learning_test_"))
    await hass.async_start()
    hass.services.async_register("conversation", "reload", lambda call: None)
    hass.services.async_register("intent_script", "reload", lambda call: None)
    return hass


def entry_for(hass: HomeAssistant) -> FakeEntry:
    """A stub entry whose `runtime_data` is a real LearnerData with a real (temp) store.

    The store key is built the way `async_load` builds it, so a save in one call is found by
    a load in the next — which is what the reply round-trip test depends on.
    """
    from custom_components.irabelle_home_learning.const import STORAGE_KEY, STORAGE_VERSION
    from custom_components.irabelle_home_learning.learner import LearnerData
    from homeassistant.helpers.storage import Store

    entry = FakeEntry()
    entry.runtime_data = LearnerData(
        store=Store(hass, STORAGE_VERSION, f"{STORAGE_KEY}.{entry.entry_id}")
    )
    # When the test world stubs the config-entry manager, the entry belongs in it: that is
    # how the component finds its entry at all.
    register = getattr(getattr(hass, "config_entries", None), "register", None)
    if register is not None:
        register(entry)
    return entry


class HassCase(unittest.TestCase):
    """A test with a started Home Assistant and a stub entry.

    `run.py` awaits `asyncSetUp`/`asyncTearDown` itself, because the standard library's
    unittest has no idea what an event loop is.
    """

    hass: HomeAssistant
    entry: FakeEntry

    async def asyncSetUp(self) -> None:  # noqa: N802 - unittest spelling
        self.hass = await make_hass()
        self.entry = entry_for(self.hass)

    async def asyncTearDown(self) -> None:  # noqa: N802
        await self.hass.async_stop()

    def sentences_dir(self):
        return self.hass.config.path("custom_sentences", "en")

    def write_sentences_file(self, name: str, text: str) -> str:
        import pathlib

        directory = pathlib.Path(self.sentences_dir())
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / name
        path.write_text(text, encoding="utf-8")
        return str(path)
