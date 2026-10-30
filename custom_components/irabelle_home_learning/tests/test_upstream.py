"""What Home Assistant's own sentence files tell us, instead of guessing.

Two things come out of the bundled intents package: the slot names a built-in intent's handler
reads, and the vocabulary a sentence may lean on. The first is a correctness matter - a phrase
that fills a slot the handler never reads is understood and then answered with nothing.
"""

from __future__ import annotations

import unittest

from support import HassCase

from custom_components.irabelle_home_learning.learner import (
    _slots_the_intent_ignores,
    _values_the_intent_needs,
)
from custom_components.irabelle_home_learning.upstream import (
    expected_slots,
    reference,
    slot_name,
    vocabulary,
)


class WhatTheBundleSays(unittest.TestCase):
    """Read from `home_assistant_intents`, which ships with Home Assistant."""

    def test_a_reference_names_the_slot_not_the_list(self):
        # `{brightness_level:brightness}` fills the slot `brightness`.
        self.assertEqual(slot_name("brightness_level:brightness"), "brightness")
        self.assertEqual(slot_name("area"), "area")

    def test_the_slots_a_handler_reads_come_from_its_own_sentences(self):
        slots = expected_slots("en", "HassLightSet")
        self.assertIn("brightness", slots)
        self.assertIn("color", slots)
        # ... and the ones every sentence may use, which Home Assistant resolves itself.
        self.assertIn("area", slots)
        self.assertIn("name", slots)

    def test_an_intent_nobody_knows_is_empty_not_restrictive(self):
        # Nothing known must not read as "nothing allowed": a house integration's intent has
        # no bundled sentences, and a warning about it would be noise.
        self.assertEqual(expected_slots("en", "NotAnIntentAnywhere"), set())

    def test_the_vocabulary_a_sentence_may_lean_on(self):
        rules, lists = vocabulary("en")
        self.assertIn("the", rules)
        self.assertIn("color", lists)

    def test_the_reference_is_a_list_for_a_prompt(self):
        self.assertIn("brightness", reference("en", "HassLightSet"))
        self.assertEqual(reference("en", "NotAnIntentAnywhere"), "")


class SlotsTheHandlerIgnores(HassCase):
    """A phrasing that fills a slot nobody reads is understood, and the value is dropped."""

    def test_a_slot_the_intent_does_not_read_is_reported(self):
        (warning,) = _slots_the_intent_ignores(
            "en", "HassLightSet", ["set the office light to {level} percent"], []
        )
        self.assertIn("'level'", warning)
        self.assertIn("brightness", warning, "the warning names what it does read")

    def test_a_slot_the_intent_reads_is_silent(self):
        self.assertEqual(
            _slots_the_intent_ignores(
                "en", "HassLightSet", ["set the office light to {brightness} percent"], []
            ),
            [],
        )

    def test_a_pinned_value_is_a_value(self):
        # `slots: {temperature: 2700}` on the group is the other correct way to do it, so the
        # check must not refuse the very fix it recommends.
        self.assertIsNone(
            _values_the_intent_needs(
                "en", "HassLightSet", ["set the office colour temperature to warm"], [],
                {"temperature"},
            )
        )

    def test_a_learned_intent_is_its_own_authority(self):
        # Its own file declares its slots; there is no bundled sentence to compare against.
        class Spec:
            name = "WeatherForecast"

        self.assertEqual(
            _slots_the_intent_ignores(
                "en", "WeatherForecast", ["what's the weather {day}"], [Spec()]
            ),
            [],
        )

    def test_an_unknown_intent_says_nothing(self):
        self.assertEqual(
            _slots_the_intent_ignores("en", "NotAnIntentAnywhere", ["a {thing} here"], []), []
        )


class ValuesTheIntentNeeds(HassCase):
    """A phrasing bound to an intent that always acts on a value has to supply one.

    Measured on the live house: "set the office colour temperature to warm" was taught to
    `HassLightSet` as fixed wording. It matched, called `light.turn_on` with no temperature,
    changed nothing, and answered with no speech - silence after a request.
    """

    def test_a_phrasing_with_no_value_is_refused_with_both_ways_out(self):
        reason = _values_the_intent_needs(
            "en", "HassLightSet", ["set the office colour temperature to warm"], []
        )
        self.assertIsNotNone(reason)
        self.assertIn("do nothing", reason)
        self.assertIn("temperature", reason, "it names what the intent acts on")
        self.assertIn("slots: {temperature: 2700}", reason, "and how to pin it instead")
        self.assertIn("maps the words to numbers", reason, "and the slot form when the values can be mapped")

    def test_a_phrasing_that_supplies_the_value_is_fine(self):
        self.assertIsNone(
            _values_the_intent_needs(
                "en", "HassLightSet", ["set the office light to {brightness}"], []
            )
        )

    def test_an_intent_that_needs_no_value_is_fine(self):
        # HassTurnOn acts on a target, not a value; "turn on the office light" is a good bind.
        self.assertIsNone(
            _values_the_intent_needs("en", "HassTurnOn", ["turn on the office light"], [])
        )

    def test_a_learned_intent_is_its_own_authority(self):
        class Spec:
            name = "OfficeLightStatus"

        self.assertIsNone(
            _values_the_intent_needs("en", "OfficeLightStatus", ["what is the office light"], [Spec()])
        )
