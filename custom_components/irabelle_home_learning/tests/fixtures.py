"""Entities a test owns, in a Home Assistant the test owns.

A test here never touches a live installation and never needs a cleanup step at the end:
every test gets its own `HomeAssistant` with a throwaway config directory, and it is thrown
away with the test. What a test *does* have to build is the same world the component reads in
production — registries, aliases, exposure — because half of this component's decisions are
made from the entity registry rather than from states:

* ownership comes from the `person` -> `device_tracker` -> device chain;
* the name a model is shown comes from the alias, not the entity id;
* `_exposed_entities` and `ListEntities` read the Assist exposure setting.

Registries are set up here the same way Home Assistant sets them up (the data key, then
`async_load`). Devices are deliberately *not* built: `device_registry.async_get_or_create`
insists on a real `ConfigEntry` from `hass.config_entries`, which does not exist outside a
bootstrapped Home Assistant, and stubbing it means chasing every attribute HA reads off it
(`domain`, `disabled_by`, ...). So the tests cover the person -> tracker link and leave the
device-name bridge to `scripts/` against a live house.
"""

from __future__ import annotations

import pathlib
import unittest
from typing import Any

from support import entry_for, make_hass

from homeassistant.components.homeassistant import exposed_entities
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er


class StubConfigEntries:
    """The one config-entry method the component calls, over a list tests can add to.

    A real `ConfigEntries` manager needs a bootstrapped Home Assistant; the component only
    ever asks this for the entries of its own domain, so the stub is exactly that question.
    """

    def __init__(self) -> None:
        self.entries: list[Any] = []

    def register(self, entry: Any) -> None:
        self.entries.append(entry)

    def async_entries(self, domain: str | None = None) -> list[Any]:
        return list(self.entries)

    async def async_wait_initialized(self) -> None:
        """Home Assistant 2026.11's entity registry waits for the entries on load."""

    async def async_unload_platforms(self, entry: Any, platforms: Any) -> bool:
        """The one platform call the unload path makes; a test world has no platforms."""
        return True


async def world() -> Any:
    """A started hass with the registries, the exposure store and the entries it reads."""
    hass = await make_hass()
    hass.config_entries = StubConfigEntries()
    for module, registry in (
        (dr, dr.DeviceRegistry(hass)),
        (er, er.EntityRegistry(hass)),
        (ar, ar.AreaRegistry(hass)),
    ):
        hass.data[module.DATA_REGISTRY] = registry
        await registry.async_load()
    exposed = exposed_entities.ExposedEntities(hass)
    hass.data[exposed_entities.DATA_EXPOSED_ENTITIES] = exposed
    await exposed.async_initialize()
    return hass


def add_entity(
    hass: Any,
    entity_id: str,
    state: str = "on",
    *,
    name: str | None = None,
    aliases: set[str] | None = None,
    attributes: dict[str, Any] | None = None,
    exposed: bool | None = None,
) -> str:
    """Create a state and, when given a name or alias, a registry row to go with it.

    The registry row is what makes the entity *legible*: an alias is how a model is shown
    "Area Last Seen aka Person's watch" without the entity being renamed, and the exposure
    flag is what Assist lists. A plain state is enough for a template to read.
    """
    domain, _, object_id = entity_id.partition(".")
    # Every entity in a real house is in the registry, so every one here is too: ownership and
    # aliases are read from the registry, and a bare state has neither.
    registry = er.async_get(hass)
    registry.async_get_or_create(
        domain, "test", object_id, suggested_object_id=object_id, original_name=name
    )
    if aliases:
        # Aliases are set by updating the row; `async_get_or_create` does not take them.
        registry.async_update_entity(entity_id, aliases=set(aliases))
    hass.states.async_set(
        entity_id, state, {"friendly_name": name or object_id, **(attributes or {})}
    )
    if exposed is not None:
        # None leaves the Assist default for the domain alone, which is a third state worth
        # being able to test: "not exposed" is a decision, "never asked" is a fallback.
        exposed_entities.async_expose_entity(hass, "conversation", entity_id, exposed)
    return entity_id


def add_person(hass: Any, name: str, trackers: list[str]) -> str:
    """A person entity listing its trackers, which is how ownership is derived."""
    entity_id = f"person.{name.lower()}"
    hass.states.async_set(
        entity_id,
        "home",
        {"friendly_name": name, "device_trackers": list(trackers)},
    )
    return entity_id


class WorldCase(unittest.TestCase):
    """A test with a world of its own: `self.hass`, `self.entry`, and no cleanup to forget.

    The temp config directory goes with the hass, so a file a test writes cannot outlive it
    and cannot collide with another test's.
    """

    hass: Any
    entry: Any

    async def asyncSetUp(self) -> None:  # noqa: N802 - unittest spelling
        self.hass = await world()
        self.entry = entry_for(self.hass)

    async def asyncTearDown(self) -> None:  # noqa: N802
        await self.hass.async_stop()

    def sentences_dir(self) -> str:
        return self.hass.config.path("custom_sentences", "en")

    def write_file(self, name: str, text: str) -> pathlib.Path:
        directory = pathlib.Path(self.sentences_dir())
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / name
        path.write_text(text, encoding="utf-8")
        return path
