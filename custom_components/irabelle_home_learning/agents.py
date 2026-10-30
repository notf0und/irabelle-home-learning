"""Agents: one hub of learning, any number of named agents in front of it.

The hub (the config entry) owns everything that is the *house's*: the sentence files, the
store of what was learned, the sensors, the services. That has to be one thing - two hubs
would write the same `custom_sentences/` files and fight over them.

An **agent** (a config subentry) is what a voice pipeline picks as its conversation agent:
a name, the agent the sentence is handed to, and the two models that learn from the turn.
"Jev -> DeepSeek" and "ChatGPT -> DeepSeek" are two agents of one hub, and whatever either
learns, both - and the local matcher - answer from then on.

Code that learns takes a config entry and reads its settings through `settings()`. A turn
passes an `AgentView` - the hub, seen through one agent - so the settings are that agent's.
Anything not started by a turn (a service, the audit) reads the hub, which resolves to the
first agent: there has to be one answer to "which model reviews this", and the first agent
is the one a person set up first.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from homeassistant.config_entries import ConfigEntry, ConfigSubentry

#: The subentry type of an agent.
SUBENTRY_AGENT = "agent"


def agent_subentries(entry: Any) -> list[ConfigSubentry]:
    """The hub's agents, oldest first."""
    subentries = getattr(entry, "subentries", None) or {}
    return [
        subentry
        for subentry in subentries.values()
        if subentry.subentry_type == SUBENTRY_AGENT
    ]


class AgentView:
    """The hub, seen through one agent: its settings win, everything else is the hub's.

    Only ``options`` and ``title`` differ. ``entry_id`` and ``runtime_data`` are the hub's,
    so the store, the spend counters and the rate limits stay shared - which is the point.
    """

    def __init__(self, entry: ConfigEntry, subentry: ConfigSubentry) -> None:
        self._entry = entry
        self.subentry = subentry

    @property
    def options(self) -> Mapping[str, Any]:
        return MappingProxyType({**self._entry.options, **self.subentry.data})

    @property
    def title(self) -> str:
        return self.subentry.title

    def __getattr__(self, name: str) -> Any:
        return getattr(self._entry, name)


def settings(entry: Any) -> Mapping[str, Any]:
    """Every setting that applies to this entry or agent, agent settings first."""
    if isinstance(entry, AgentView):
        return entry.options
    agents = agent_subentries(entry)
    base = dict(agents[0].data) if agents else {}
    return {**dict(getattr(entry, "data", {}) or {}), **base, **dict(entry.options)}
