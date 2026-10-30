"""The sentence rules that are really *hassil's* rules.

Calling hassil directly is the only way to know them: the component can write a sentence and
report it as learned while the matcher quietly never accepts it, which is what happened with
punctuation and what the undeclared-slot refusal exists to prevent.
"""

from __future__ import annotations

import unittest

from hassil import Intents, recognize
from hassil.errors import MissingListError


def match(sentences, text, lists=None):
    """What hassil makes of one utterance, as `{slot: value}` or None."""
    config = {
        "language": "en",
        "intents": {"X": {"data": [{"sentences": sentences}]}},
    }
    if lists:
        config["lists"] = lists
    return recognize(text, Intents.from_dict(config))


class Punctuation(unittest.TestCase):
    """A stored sentence with `, ; ? !` matches nothing — not even the same input.

    Home Assistant strips that punctuation from the *input* and not from the stored sentence,
    so the two can never meet. This is why `sentence_for_storage` exists.
    """

    def test_a_question_mark_in_the_stored_sentence_never_matches(self):
        self.assertIsNone(match(["is it cold?"], "is it cold"))
        self.assertIsNone(match(["is it cold?"], "is it cold?"))

    def test_the_same_sentence_without_punctuation_matches(self):
        self.assertIsNotNone(match(["is it cold"], "is it cold"))

    def test_a_comma_behaves_the_same_way(self):
        self.assertIsNone(match(["what time is it, please"], "what time is it please"))


class Slots(unittest.TestCase):
    def test_a_declared_list_matches_any_value_and_is_case_insensitive(self):
        result = match(
            ["what's the weather {day}"],
            "what's the weather tuesday",
            {"day": {"values": ["monday", "Tuesday"]}},
        )
        self.assertIsNotNone(result)

    def test_the_value_comes_back_as_the_list_spells_it(self):
        # Not as it was said — which is why a template that prints `day` needs `| lower`.
        result = match(
            ["what's the weather {day}"],
            "what's the weather tuesday",
            {"day": {"values": ["monday", "Tuesday"]}},
        )
        self.assertEqual(result.entities["day"].value, "Tuesday")

    def test_an_undeclared_slot_raises_at_match_time(self):
        # Not at load time: the file loads, and then recognition throws for *every* agent.
        # That is why `_validate` refuses a `{slot}` whose list the same proposal omits.
        with self.assertRaises(MissingListError):
            match(["weather {day}"], "weather monday")

    def test_a_pinned_value_is_applied_without_a_placeholder(self):
        result = match(
            ["what's the weather tomorrow"],
            "what's the weather tomorrow",
            None,
        )
        self.assertIsNotNone(result)

    def test_an_optional_group_matches_with_and_without_its_words(self):
        result = match(["[please ]turn it off"], "turn it off")
        self.assertIsNotNone(result)
        self.assertIsNotNone(match(["[please ]turn it off"], "please turn it off"))


class WildcardLists(unittest.TestCase):
    """A wildcard swallows arbitrary words — and `values: []` silently disables it.

    The house's hand-written files used `[{speaker} ]who am i` so a stray leading word (a
    wake word that reached the transcript, a name) would not stop the phrase matching. That
    only works when the list is declared with `wildcard: true` and **no** `values` key, which
    is the shape `normalise_slots` writes.
    """

    SPEAKER = {"speaker": {"wildcard": True}}

    def test_the_optional_wildcard_prefix_matches_bare_and_prefixed(self):
        self.assertIsNotNone(match(["[{speaker} ]who am i"], "who am i", self.SPEAKER))
        prefixed = match(["[{speaker} ]who am i"], "please who am i", self.SPEAKER)
        self.assertIsNotNone(prefixed)
        self.assertEqual(prefixed.entities["speaker"].value, "please")

    def test_a_values_key_kills_the_wildcard(self):
        crippled = {"speaker": {"wildcard": True, "values": []}}
        self.assertIsNone(match(["[{speaker} ]who am i"], "please who am i", crippled))

    def test_a_wildcard_needs_the_optional_group_to_match_nothing(self):
        # Without the brackets it demands at least one word, so the bare phrase fails.
        self.assertIsNone(match(["{speaker} who am i"], "who am i", self.SPEAKER))


if __name__ == "__main__":
    unittest.main(verbosity=2)
