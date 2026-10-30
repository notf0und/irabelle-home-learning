"""Slots: what a model may declare, and how example phrasings fold onto `{day}`.

`normalise_slots` is the only door a list of values comes through, and each refusal in it was
written after the failure it prevents: a one-value list installed a slot that matched the
literal word "day", a `{name}` with no list stops matching for every agent, and a list of
forty free-form values is not a word-class.
"""

from __future__ import annotations

import unittest

from custom_components.irabelle_home_learning.autolearn import _slot_lines
from custom_components.irabelle_home_learning.intents import (
    concrete,
    normalise_slots,
    parameterise,
    slot_sample,
)
from custom_components.irabelle_home_learning.learner import LearnerError

DAYS = {
    "day": {
        "values": ["today", "tomorrow", "after tomorrow", "monday", "friday"],
    }
}


class Normalise(unittest.TestCase):
    def test_the_shapes_a_model_writes_all_work(self):
        expected = {"day": {"values": ["today", "tomorrow"]}}
        self.assertEqual(normalise_slots({"day": ["today", "tomorrow"]}), expected)
        self.assertEqual(normalise_slots({"day": "today, tomorrow"}), expected)
        self.assertEqual(normalise_slots({"day": {"values": ["today", "tomorrow"]}}), expected)

    def test_nothing_declared_is_an_empty_dict(self):
        self.assertEqual(normalise_slots(None), {})
        self.assertEqual(normalise_slots({}), {})

    def test_a_wildcard_is_the_shape_hassil_needs(self):
        # No `values` key: hassil silently stops matching a wildcard that carries one.
        self.assertEqual(normalise_slots({"speaker": "wildcard"}), {"speaker": {"wildcard": True}})
        self.assertEqual(
            normalise_slots({"speaker": {"wildcard": True}}), {"speaker": {"wildcard": True}}
        )

    def test_one_value_is_refused(self):
        for raw in ({"day": "day"}, {"day": ["today"]}):
            with self.assertRaises(LearnerError):
                normalise_slots(raw)

    def test_an_unusable_name_is_refused(self):
        with self.assertRaises(LearnerError):
            normalise_slots({"Day Light": ["a", "b"]})

    def test_a_value_that_is_not_a_spoken_word_is_refused(self):
        for bad in ("{day}", "a|b", "["):
            with self.assertRaises(LearnerError):
                normalise_slots({"day": ["today", bad]})

    def test_duplicates_fold(self):
        self.assertEqual(
            normalise_slots({"day": ["Today", "today", "tomorrow"]}),
            {"day": {"values": ["Today", "tomorrow"]}},
        )


class Parameterise(unittest.TestCase):
    """Example phrasings in, one sentence with `{day}` out."""

    def test_a_value_is_replaced_by_the_slot(self):
        self.assertEqual(
            parameterise("how is the weather on Monday", DAYS),
            "how is the weather on {day}",
        )

    def test_a_word_that_is_not_a_declared_value_is_left_alone(self):
        # Only values the list declares fold; the rest of the sentence is the person's words.
        self.assertEqual(
            parameterise("how is the weather on Wednesday", DAYS),
            "how is the weather on Wednesday",
        )

    def test_the_longest_value_wins(self):
        # "after tomorrow" must not be eaten by "tomorrow", or the sentence matches the
        # wrong phrase and captures "after" as a separate word.
        self.assertEqual(
            parameterise("what's the weather after tomorrow", DAYS),
            "what's the weather {day}",
        )

    def test_word_boundaries(self):
        self.assertEqual(parameterise("todays forecast", DAYS), "todays forecast")

    def test_no_slots_leaves_the_wording_alone(self):
        self.assertEqual(parameterise("who am i", None), "who am i")


class SampleAndConcrete(unittest.TestCase):
    def test_the_sample_is_the_first_value(self):
        self.assertEqual(slot_sample(DAYS), {"day": "today"})
        self.assertEqual(slot_sample({"speaker": {"wildcard": True}}), {"speaker": "someone"})
        self.assertEqual(slot_sample(None), {})

    def test_concrete_fills_every_slot_for_the_matcher(self):
        # `async_verify` asks the real recogniser, so it needs a sayable sentence.
        self.assertEqual(
            concrete("what's the weather {day}", DAYS), "what's the weather today"
        )


class SlotLines(unittest.TestCase):
    """The generaliser asks for slots in plain text, one line per slot."""

    def test_equals_and_colon_both_parse(self):
        self.assertEqual(
            _slot_lines("day = today, tomorrow\nwho: Alex, Sam"),
            {"day": ["today", "tomorrow"], "who": ["Alex", "Sam"]},
        )

    def test_repeated_lines_merge_rather_than_replace(self):
        self.assertEqual(
            _slot_lines("day = today, tomorrow\nday: monday"), {"day": ["today", "tomorrow", "monday"]}
        )

    def test_prose_and_noise_are_ignored(self):
        self.assertEqual(_slot_lines("I think a day slot would help\n"), {})
        self.assertEqual(_slot_lines(""), {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
