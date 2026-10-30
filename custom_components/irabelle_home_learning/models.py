"""Any model the house has: an `ai_task` entity or a conversation agent.

Both the processing model and the classification model are picked from one list, by the
name a person knows them by ("DeepSeek AI Task", "Groq Cloud API"), not by entity id.
Many integrations offer only a conversation agent and no `ai_task`, so both kinds count.

`async_generate_data` is `ai_task.generate_data` for either kind. An `ai_task` entity is
asked directly, with the structure as its output schema. A conversation agent has no
output schema, so the structure is written into the request as a list of JSON keys and
the reply is read back as JSON - the same answer, by a longer road. A conversation agent
keeps whatever tools it was given, so the request says plainly that it must not act.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from homeassistant.components import conversation
from homeassistant.core import Context, HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.selector import SelectOptionDict
from homeassistant.util import ulid

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

#: Integrations whose conversation agent is not a model to be asked for JSON: this one,
#: and Jev, which routes to intents and is offered as a classifier through `jev.ask`.
NOT_MODELS = frozenset({DOMAIN, "jev"})


def label_of(hass: HomeAssistant, entity_id: str) -> str:
    """The name a person knows a model by: its own name, else its device's."""
    registry = er.async_get(hass)
    row = registry.async_get(entity_id)
    if row is not None:
        if row.name:
            return row.name
        device = dr.async_get(hass).async_get(row.device_id) if row.device_id else None
        if device is not None and (device.name_by_user or device.name):
            name = device.name_by_user or device.name
            # The way Home Assistant itself names it: the device, then the entity's own
            # name when it has one that adds something.
            if row.has_entity_name and row.original_name and row.original_name != name:
                return f"{name} {row.original_name}"
            return name
        if row.original_name:
            return row.original_name
    state = hass.states.get(entity_id)
    if state is not None and state.name:
        return state.name
    return entity_id


def model_choices(hass: HomeAssistant, current: str | None = None) -> list[SelectOptionDict]:
    """Every `ai_task` entity and conversation agent that can be asked, by name."""
    registry = er.async_get(hass)
    rows = [
        row
        for row in registry.entities.values()
        if row.domain in ("ai_task", "conversation")
        and row.platform not in NOT_MODELS
        # Home Assistant's own agent matches sentences; it cannot be asked a question.
        and row.entity_id != "conversation.home_assistant"
        and not row.disabled_by
    ]
    choices = [
        SelectOptionDict(label=label_of(hass, row.entity_id), value=row.entity_id)
        for row in rows
    ]
    if current and current.startswith(("ai_task.", "conversation.")) and current not in {
        choice["value"] for choice in choices
    }:
        choices.append(SelectOptionDict(label=f"{current} (missing)", value=current))
    # AI tasks first - they have an output schema - then by name.
    return sorted(
        choices,
        key=lambda choice: (not choice["value"].startswith("ai_task."), choice["label"].lower()),
    )


def agent_choices(hass: HomeAssistant) -> list[SelectOptionDict]:
    """Conversation agents a sentence can be handed to, by name. Never one of ours."""
    registry = er.async_get(hass)
    choices = [
        SelectOptionDict(label="Home Assistant", value="conversation.home_assistant")
    ]
    for row in registry.entities.values():
        if row.domain != "conversation" or row.platform == DOMAIN or row.disabled_by:
            continue
        if row.entity_id == "conversation.home_assistant":
            continue
        choices.append(SelectOptionDict(label=label_of(hass, row.entity_id), value=row.entity_id))
    return [choices[0], *sorted(choices[1:], key=lambda choice: choice["label"].lower())]


# --------------------------------------------------------------------------
# Asking
# --------------------------------------------------------------------------


def _describe_field(name: str, spec: dict[str, Any]) -> str:
    selector = spec.get("selector") or {}
    kind = next(iter(selector), "text")
    detail = selector.get(kind) or {}
    if kind == "select":
        shape = "one of: " + ", ".join(json.dumps(str(option)) for option in detail.get("options", []))
    elif kind == "number":
        shape = f"a number from {detail.get('min', 0)} to {detail.get('max', 1)}"
    elif kind == "boolean":
        shape = "true or false"
    else:
        shape = "a string"
    required = " (required)" if spec.get("required") else ""
    return f'- "{name}"{required}: {shape}. {spec.get("description") or ""}'.rstrip()


def _json_object(text: str) -> dict[str, Any] | None:
    """The JSON object in a reply, wherever the model put it."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = [fenced.group(1)] if fenced else []
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


async def async_generate_data(hass: HomeAssistant, data: dict[str, Any]) -> dict[str, Any]:
    """`ai_task.generate_data` for an `ai_task` entity or a conversation agent.

    Takes the same service data and returns the same ``{"data": ...}`` response, so the
    callers do not know which kind of model answered.
    """
    entity_id = str(data.get("entity_id") or "")
    if entity_id.startswith("ai_task."):
        return await hass.services.async_call(
            "ai_task", "generate_data", data, blocking=True, return_response=True
        ) or {}
    if not entity_id.startswith("conversation."):
        raise HomeAssistantError(f"{entity_id!r} is not an ai_task entity or a conversation agent")

    structure: dict[str, Any] = data.get("structure") or {}
    request = (
        f"{data.get('instructions') or ''}\n\n"
        "This is a data task, not a conversation. Do not call any tool and do not change "
        "anything in the house. Reply with ONLY one JSON object - no prose, no markdown - "
        "with these keys:\n"
        + "\n".join(_describe_field(name, spec) for name, spec in structure.items())
    )
    result = await conversation.async_converse(
        hass,
        request,
        # A fresh conversation each time: nothing from one task may leak into the next.
        ulid.ulid_now(),
        Context(),
        agent_id=entity_id,
    )
    speech = result.response.speech.get("plain", {}).get("speech", "") if result else ""
    payload = _json_object(speech)
    if payload is None:
        _LOGGER.debug("%s did not answer %r with JSON: %r", entity_id, data.get("task_name"), speech[:300])
        raise HomeAssistantError(f"{label_of(hass, entity_id)} did not answer with JSON")
    return {"data": payload}
