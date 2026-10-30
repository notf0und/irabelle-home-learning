"""The startup passes that keep voice working, and the undo paths.

Two of these prevent the failure mode this component fears most: a sentence whose intent
nothing answers makes *recognition itself* fail, so one leftover takes local voice down for
every agent. The other repairs a sentence that was written before punctuation was understood,
which would otherwise sit in a file looking learned and matching nothing.
"""

from __future__ import annotations

import pathlib
import unittest

import fixtures
import yaml

from custom_components.irabelle_home_learning.intents import (
    async_forget_intent,
    async_learn_intent,
    learner_sentences,
    register_intent,
)
from custom_components.irabelle_home_learning.learner import (
    LearnedIntent,
    LearnedPhrase,
    async_prune_orphans,
    normalize_sentence_files,
)


def load(path) -> dict:
    return yaml.safe_load(open(path, encoding="utf-8")) or {}


class StartupRepair(fixtures.WorldCase):
    """`normalize_sentence_files`: a stored sentence has to be one hassil accepts."""

    async def test_a_stored_question_mark_is_repaired(self):
        path = self.write_file(
            "learned-broken.yaml",
            "language: en\nintents:\n  Broken:\n    data:\n    - sentences:\n      - is it cold?\n",
        )
        self.assertEqual(normalize_sentence_files(self.hass, "en"), ["learned-broken.yaml"])
        self.assertEqual(
            load(path)["intents"]["Broken"]["data"][0]["sentences"], ["is it cold"]
        )

    async def test_a_matching_sentence_is_left_alone(self):
        self.write_file(
            "learned-fine.yaml",
            "language: en\nintents:\n  Fine:\n    data:\n    - sentences:\n      - is it cold\n",
        )
        self.assertEqual(normalize_sentence_files(self.hass, "en"), [])


class Pruning(fixtures.WorldCase):
    """`async_prune_orphans`: remove ours, only ever report someone else's."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        # "Kept" is a real intent; "Gone" is not registered by anything.
        register_intent(self.hass, LearnedIntent(name="Kept", template="ok"))
        self.ours = self.write_file(
            "learned-orphan.yaml",
            "language: en\nintents:\n  Gone:\n    data:\n    - sentences:\n      - never answered\n"
            "  Kept:\n    data:\n    - sentences:\n      - still answered\n",
        )

    async def test_our_sentence_for_an_unregistered_intent_is_removed(self):
        result = await async_prune_orphans(self.hass, self.entry)
        self.assertEqual(result["removed"], ["Gone (learned-orphan.yaml)"])
        remaining = load(self.ours)["intents"]
        self.assertNotIn("Gone", remaining)
        self.assertIn("Kept", remaining)

    async def test_a_learner_file_holding_only_orphans_is_deleted(self):
        path = self.write_file(
            "learned-only-orphan.yaml",
            "language: en\nintents:\n  Ghost:\n    data:\n    - sentences:\n      - boo\n",
        )
        await async_prune_orphans(self.hass, self.entry)
        self.assertFalse(path.exists())

    async def test_a_name_a_hand_written_file_declares_is_never_removed(self):
        # The intent belongs to whoever wrote that file; ours is a binding to it, and its
        # handler may simply not be registered yet when this runs.
        self.write_file(
            "house.yaml",
            "language: en\nintents:\n  Gone:\n    data:\n    - sentences:\n      - asked here\n",
        )
        result = await async_prune_orphans(self.hass, self.entry)
        self.assertEqual(result["removed"], [])
        self.assertIn("Gone", load(self.ours)["intents"])

    async def test_a_foreign_file_with_no_handler_is_reported_not_rewritten(self):
        foreign = self.write_file(
            "house.yaml",
            "language: en\nintents:\n  Nowhere:\n    data:\n    - sentences:\n      - asked here\n",
        )
        result = await async_prune_orphans(self.hass, self.entry)
        self.assertEqual(result["reported"], ["Nowhere"])
        self.assertIn("Nowhere", load(foreign)["intents"])


class Forgetting(fixtures.WorldCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        fixtures.add_entity(self.hass, "weather.home", "rainy", attributes={"temperature": 14.0})
        self.result = await async_learn_intent(
            self.hass,
            self.entry,
            name="WeatherForecast",
            template="It is {{ states('weather.home') }}.",
            utterances=["what's the weather"],
        )
        self.entry.runtime_data.phrases.append(
            LearnedPhrase(utterance="what's the weather", intent="WeatherForecast", file=self.result["file"])
        )

    async def test_forgetting_removes_the_handler_the_sentences_and_the_record(self):
        from homeassistant.helpers import intent as intent_helper

        await async_forget_intent(self.hass, self.entry, "WeatherForecast")

        registered = {handler.intent_type for handler in intent_helper.async_get(self.hass)}
        self.assertNotIn("WeatherForecast", registered)
        self.assertEqual(learner_sentences(self.hass, "en", {"WeatherForecast"}), {})
        self.assertEqual(self.entry.runtime_data.intents, [])
        self.assertEqual(self.entry.runtime_data.phrases, [])
        # The file is deleted with the intent, or its sentences would fail the pipeline.
        self.assertFalse(pathlib.Path(self.hass.config.path(self.result["file"])).exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
