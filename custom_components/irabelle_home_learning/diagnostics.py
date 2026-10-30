"""The integration's diagnostics: everything it knows, in one download.

Home Assistant offers this from the integration's page (Download diagnostics), which is
where a person looks when the entities are not enough - the exact templates, the phrasings,
what was heard and what was decided about it, what it cost, and what it found wrong with
itself. The API key is never included.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from homeassistant.components.diagnostics import async_redact_data
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import CONF_JEV_API_KEY
from .judge import describe, usage_of
from .learner import LearnerData

TO_REDACT = {CONF_JEV_API_KEY}


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: ConfigEntry
) -> dict[str, Any]:
    """Return everything the learner knows, for a bug report."""
    data: LearnerData = entry.runtime_data
    return {
        "options": async_redact_data(dict(entry.options), TO_REDACT),
        "judge": describe(hass, entry),
        # `JudgeUsage` is a slots dataclass, so `vars()` raises on it.
        "judge_since_restart": dataclasses.asdict(usage_of(entry)),
        "spend_today": data.spend,
        "last_learned": data.last_learned,
        "last_result": data.last_result,
        "counts": {
            "phrases": len(data.phrases),
            "intents": len(data.intents),
            "observations": len(data.observations),
            "quarantined": len(data.quarantined),
            "drift": len(data.drift),
            "orphans": data.orphans,
            "consolidated": data.consolidated,
            "turns_seen": data.turns_seen,
        },
        "intents": [
            {
                "name": spec.name,
                "template": spec.template,
                "response": spec.response or spec.template,
                "file": spec.file,
                "learned_at": spec.learned_at,
                "note": spec.note,
                "source": spec.source,
            }
            for spec in data.intents
        ],
        "phrases": [row.as_dict() for row in data.phrases],
        "observations": [row.as_dict() for row in data.observations[-50:]],
        "repeats": data.repeats,
        "drift": data.drift,
    }
