"""Observability: what the house learned, what it cost, and what it is waiting on.

These are entities rather than attributes on one entity because that is where the
information is actually useful: every one of them appears on the integration's device page,
and a sensor whose *state* changes is what puts an entry in the Activity log. "Learned
phrases: 58" is a number nobody can act on; "Last learned: where is my phone at" is one
line in the log that says what happened and when.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from homeassistant.components.sensor import SensorEntity, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util import dt as dt_util

from .const import DOMAIN, JEV_PRICE_PER_MTOK, SENSOR_ATTRIBUTE_LIMIT, SIGNAL_UPDATED
from .learner import LearnerData


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Add the learner's entities."""
    async_add_entities(
        [
            LearnedPhrasesSensor(entry),
            LastLearnedSensor(entry),
            HeardOnceSensor(entry),
            JudgeCallsSensor(entry),
            JudgeCostSensor(entry),
        ]
    )


class HomeLearningSensor(SensorEntity):
    """Shared plumbing: one device, updated by the integration's own signal."""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, entry: ConfigEntry, key: str) -> None:
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_{key}"
        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
            "name": "Irabelle Home Learning",
            "manufacturer": "local",
            "model": "learned voice phrasings",
        }

    async def async_added_to_hass(self) -> None:
        self.async_on_remove(
            async_dispatcher_connect(self.hass, SIGNAL_UPDATED, self._handle_update)
        )

    @callback
    def _handle_update(self) -> None:
        self.async_write_ha_state()

    @property
    def _data(self) -> LearnerData:
        return self._entry.runtime_data

    @property
    def _spend(self) -> dict[str, Any]:
        return self._data.spend or {}


class LearnedPhrasesSensor(HomeLearningSensor):
    """How many phrasings this house answers locally because it learned them."""

    _attr_name = "Learned phrases"
    _attr_icon = "mdi:school-outline"

    def __init__(self, entry: ConfigEntry) -> None:
        super().__init__(entry, "learned_phrases")

    @property
    def native_value(self) -> int:
        return len(self._data.phrases)

    @property
    def extra_state_attributes(self) -> dict:
        data = self._data
        recent = list(reversed(data.phrases))[:SENSOR_ATTRIBUTE_LIMIT]
        return {
            "learned": [row.as_dict() for row in recent],
            "phrase_intents": sorted({row.intent for row in data.phrases}),
            "files": sorted({row.file for row in data.phrases if row.file}),
            "learned_intents": [
                {
                    "name": spec.name,
                    "description": spec.description,
                    "note": spec.note,
                    "source": spec.source,
                    "learned_at": spec.learned_at,
                    "file": spec.file,
                }
                for spec in data.intents
            ],
            "quarantined": data.quarantined,
            "last_result": data.last_result,
            # What the system thinks of itself: what it spent, what it made of the
            # requests it saw, what it found wrong with what is already installed, and
            # which phrases people had to ask for twice.
            "stats": {
                **data.spend,
                "drift": data.drift,
                "repeats": data.repeats,
                "orphans": data.orphans,
                "consolidated": data.consolidated,
            },
        }


class LastLearnedSensor(HomeLearningSensor):
    """What the house learned most recently - one line per learning in the log.

    The state is the phrase itself, so the Activity tab of the device reads as a list of
    what it learned and when, which is the question "what has it learned?" actually being
    asked when a count is not enough.
    """

    _attr_name = "Last learned"
    _attr_icon = "mdi:lightbulb-on-outline"

    def __init__(self, entry: ConfigEntry) -> None:
        super().__init__(entry, "last_learned")

    @property
    def native_value(self) -> str:
        learned = self._data.last_learned or {}
        return str(learned.get("phrase") or "nothing yet")

    @property
    def extra_state_attributes(self) -> dict:
        learned = dict(self._data.last_learned or {})
        learned.setdefault("how", "nothing learned since this integration started")
        return learned


#: How long a phrase heard once is still "waiting". After this it is history: the window
#: is what makes the number mean something now, rather than counting everything the house
#: has ever failed to answer.
WAITING_DAYS = 7


class HeardOnceSensor(HomeLearningSensor):
    """Phrases heard once and not yet decided, recently enough to still be waiting.

    Called "pending" first, which promised a queue that drains. What it counted was every
    phrase heard once ever recorded - most of them one-offs and instructions about the
    assistant, none of which would ever be said twice, so the number only grew. The
    window, and the name, are the fix.
    """

    _attr_name = "Heard once"
    _attr_icon = "mdi:timer-sand"
    _attr_native_unit_of_measurement = "requests"

    def __init__(self, entry: ConfigEntry) -> None:
        super().__init__(entry, "heard_once")

    def _waiting(self) -> list:
        cutoff = dt_util.now() - dt.timedelta(days=WAITING_DAYS)
        recent = []
        for row in self._data.observations:
            if row.state != "new":
                continue
            try:
                seen = dt_util.parse_datetime(row.last_seen)
            except (TypeError, ValueError):
                seen = None
            if seen is not None and dt_util.as_local(seen) >= cutoff:
                recent.append(row)
        return recent

    @property
    def native_value(self) -> int:
        return len(self._waiting())

    @property
    def extra_state_attributes(self) -> dict:
        data = self._data
        waiting = self._waiting()
        all_new = [row for row in data.observations if row.state == "new"]
        return {
            "window_days": WAITING_DAYS,
            "heard_within_the_window": [
                row.utterance for row in waiting[-SENSOR_ATTRIBUTE_LIMIT:]
            ],
            "heard_ever_once_and_undecided": len(all_new),
            "oldest_undecided": min(
                (row.last_seen for row in all_new), default=None
            ),
            "waiting_for": (
                "a second sighting, unless the judge is certain it is a bind; after the "
                "window it is history rather than something pending"
            ),
            "repeats": data.repeats,
        }


class JudgeCallsSensor(HomeLearningSensor):
    """How many questions the judge has been asked today."""

    _attr_name = "Judge calls today"
    _attr_icon = "mdi:comment-question-outline"
    _attr_native_unit_of_measurement = "calls"
    _attr_state_class = SensorStateClass.TOTAL
    _attr_suggested_display_precision = 0

    def __init__(self, entry: ConfigEntry) -> None:
        super().__init__(entry, "judge_calls_today")

    @property
    def native_value(self) -> int:
        return int(self._spend.get("calls", 0))

    @property
    def last_reset(self):
        # The count is per day, so midnight is where it goes back to zero - which is what
        # `TOTAL` needs to be told, or the statistics record a fall to zero as a fault.
        return dt_util.start_of_local_day()

    @property
    def extra_state_attributes(self) -> dict:
        spend = self._spend
        return {
            "day": spend.get("day"),
            "input_tokens": spend.get("input_tokens"),
            "output_tokens": spend.get("output_tokens"),
            "audit_position": spend.get("audit_position"),
            "last_audit": spend.get("last_audit"),
        }


class JudgeCostSensor(HomeLearningSensor):
    """What the judge has cost today, estimated from the tokens it was sent."""

    _attr_name = "Judge cost today"
    _attr_icon = "mdi:currency-usd"
    _attr_native_unit_of_measurement = "USD"
    _attr_state_class = SensorStateClass.TOTAL
    _attr_suggested_display_precision = 6

    def __init__(self, entry: ConfigEntry) -> None:
        super().__init__(entry, "judge_cost_today")

    @property
    def native_value(self) -> float:
        return float(self._spend.get("estimated_cost", 0.0))

    @property
    def last_reset(self):
        return dt_util.start_of_local_day()

    @property
    def extra_state_attributes(self) -> dict:
        spend = self._spend
        return {
            "priced_at": f"${JEV_PRICE_PER_MTOK}/Mtok input, output free (Jev)",
            "estimated": (
                "an estimate from the input tokens Jev reports, at a constant price in "
                "const.py; a classification model behind ai_task reports no tokens, so its "
                "calls are counted but not priced"
            ),
            "input_tokens": spend.get("input_tokens"),
            "calls": spend.get("calls"),
            "learned_today": spend.get("learned_today"),
            "ignored_today": spend.get("ignored_today"),
            "still_new": spend.get("still_new"),
        }
