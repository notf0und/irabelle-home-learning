"""Folding phrasings into one line: arithmetic, a judgement, and the matcher's veto.

The three parts fail differently and are tested apart: the candidate is string arithmetic and
must not invent wrdings that say something else, the judgement is the judge's and is stubbed
here, and the matcher is the real one - a fold it does not accept must never be written.
"""

from __future__ import annotations

import pathlib
import unittest

import fixtures

from support import HassCase

from custom_components.irabelle_home_learning import combine
from custom_components.irabelle_home_learning.combine import candidates, covers, group, pins_of
from custom_components.irabelle_home_learning.intents import _write_sentences
from custom_components.irabelle_home_learning.learner import LearnedIntent, LearnerError


def fold(sentences):
    return {item.merged for item in candidates("X", sentences)}


class Candidates(unittest.TestCase):
    """What string arithmetic may propose. Nothing here is allowed to need a model."""

    def test_one_word_apart_becomes_an_alternative(self):
        self.assertEqual(
            fold(["what's the weather today", "what's the weather tomorrow"]),
            {"what's the weather (today | tomorrow)"},
        )

    def test_a_span_of_words_apart_becomes_an_alternative(self):
        self.assertEqual(
            fold(["how was the office light turned on", "how was the office light switched on"]),
            {"how was the office light (turned | switched) on"},
        )

    def test_a_word_only_one_has_becomes_optional(self):
        self.assertEqual(
            fold(["what's the weather", "what's the weather like"]),
            {"what's the weather [like]"},
        )

    def test_two_extra_words_become_optional_together(self):
        self.assertEqual(
            fold(["is the office light on", "is the office light on right now"]),
            {"is the office light on [right now]"},
        )

    def test_sentences_that_share_nothing_are_left_alone(self):
        self.assertEqual(fold(["what's the weather today", "where is my phone"]), set())

    def test_a_sentence_already_folded_is_not_folded_again(self):
        # Otherwise the second pass writes `((a | b) | c)`.
        self.assertEqual(fold(["what's the weather (today | tomorrow)", "what's the weather friday"]), set())

    def test_every_proposal_names_the_two_it_replaces(self):
        (candidate,) = candidates("Weather", ["what's the weather", "what's the weather like"])
        self.assertEqual(candidate.intent, "Weather")
        self.assertEqual({candidate.a, candidate.b}, {"what's the weather", "what's the weather like"})
        self.assertEqual(candidate.kind, "optional")


class Grouping(unittest.TestCase):
    """Three phrasings that differ in one place are one line, not three overlapping ones."""

    def test_variants_at_one_place_become_one_line(self):
        (fold,) = group(candidates("X", ["is the TV on", "is the telly on", "is the tv on"]))
        self.assertEqual(fold.merged, "is the (TV | telly | tv) on")
        self.assertEqual(
            set(fold.sources), {"is the TV on", "is the telly on", "is the tv on"}
        )

    def test_a_family_too_big_to_read_keeps_a_line_for_the_rest(self):
        # Nothing may be lost to the limit: the phrasings that do not fit stay where they are.
        sentences = [f"is the {word} on" for word in ("TV", "telly", "set", "screen", "panel")]
        folds = group(candidates("X", sentences), limit=2)
        self.assertEqual(folds[0].merged, "is the (TV | telly) on")
        covered = {source for fold in folds for source in fold.sources}
        self.assertEqual(covered, {"is the TV on", "is the telly on"})
        self.assertNotIn("is the set on", covered, "left for its own line, not dropped")

    def test_a_phrasing_is_claimed_by_one_line_only(self):
        # The same three phrasings can be split at two different places. Only one line may have
        # them, or the file gains a redundant line; here the family that takes all three wins,
        # and the tauter-looking split is the one dropped.
        folds = group(
            candidates(
                "X",
                ["is the living room TV on", "is the living room telly on", "is the lounge tv on"],
            )
        )
        claimed = [source for fold in folds for source in fold.sources]
        self.assertEqual(len(claimed), len(set(claimed)), f"claimed twice: {claimed}")
        self.assertEqual(len(folds), 1)
        self.assertEqual(
            folds[0].merged, "is the (living room TV | lounge tv | living room telly) on"
        )
        self.assertEqual(len(folds[0].sources), 3, "one line replaces all three")

    def test_optional_words_do_not_share_a_phrasing(self):
        folds = group(
            candidates("X", ["is the light on", "is the light on right now", "is the light on now"])
        )
        seen = [source for fold in folds for source in fold.sources]
        self.assertEqual(len(seen), len(set(seen)), f"a phrasing was taken twice: {seen}")


class TheMatcherDecides(unittest.TestCase):
    """`covers` is the real recogniser, offline, one concrete value per slot."""

    def test_a_good_fold_covers_both_originals(self):
        self.assertTrue(
            covers(
                "X",
                "what's the weather (today | tomorrow)",
                ["what's the weather today", "what's the weather tomorrow"],
                {},
            )
        )

    def test_a_fold_that_misses_one_original_is_not_a_fold(self):
        self.assertFalse(
            covers(
                "X",
                "what's the weather (today | tomorrow)",
                ["what's the weather today", "what's the weather friday"],
                {},
            )
        )

    def test_a_slot_is_filled_before_the_matcher_is_asked(self):
        # `{day}` is not an utterance; the matcher is asked with one of its values.
        lists = {"day": {"values": ["monday"]}}
        self.assertTrue(
            covers("X", "what's the weather {day}", ["what's the weather {day}"], lists)
        )
        self.assertTrue(
            covers("X", "what's the weather {day}", ["what's the weather monday"], lists)
        )
        # ... and a fold that drops a word the original had is not a fold.
        self.assertFalse(
            covers("X", "what's the weather {day}", ["what's the weather on {day}"], lists)
        )


class PinnedWordings(fixtures.WorldCase):
    """A fold must carry the `slots:` pin of the wordings it replaces.

    Measured on the live house: folding `what's the weather tomorrow` (pinned `day: tomorrow`)
    into an unpinned group still *matched* - and then answered about today. The sentence was
    right, the slot was gone, and the old verifier could not see it because it only asked
    whether the words matched. So these tests look at the file, not at the match.
    """

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.entry.options["judge"] = "jev"
        self.spec = LearnedIntent(
            name="Weather", template="x", file="custom_sentences/en/learned-weather.yaml"
        )
        self.entry.runtime_data.intents.append(self.spec)
        self.path = pathlib.Path(self.hass.config.path(self.spec.file))
        self.original = combine.async_ask

        class Answer:
            noul = 0.95

        async def fake_ask(hass, entry, state, questions, **kwargs):
            return {key: Answer() for key in questions}

        combine.async_ask = fake_ask

    async def asyncTearDown(self) -> None:
        combine.async_ask = self.original
        await super().asyncTearDown()

    def write(self, groups) -> None:
        import yaml

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            yaml.safe_dump(
                {"language": "en", "intents": {"Weather": {"data": groups}}}, sort_keys=False
            ),
            encoding="utf-8",
        )

    def groups(self):
        import yaml

        data = yaml.safe_load(self.path.read_text())
        return data["intents"]["Weather"]["data"]

    async def test_the_folded_line_lands_in_the_group_with_its_pin(self):
        self.write(
            [
                {"sentences": ["what's the weather {day}"]},
                {
                    "sentences": ["what's the weather tomorrow", "whats the weather tomorrow"],
                    "slots": {"day": "tomorrow"},
                },
            ]
        )
        report = await combine.async_combine(self.hass, self.entry, "Weather", apply=True)
        self.assertTrue(report["applied"], report)
        merged = [item for item in self.groups() if any("(" in s for s in item["sentences"])]
        self.assertEqual(len(merged), 1)
        self.assertEqual(
            merged[0].get("slots"),
            {"day": "tomorrow"},
            "the pin has to travel with the folded sentence, or it answers about today",
        )

    async def test_wordings_that_pin_different_values_are_not_folded(self):
        self.write(
            [
                {"sentences": ["what's the weather today"], "slots": {"day": "today"}},
                {"sentences": ["what's the weather tomorrow"], "slots": {"day": "tomorrow"}},
            ]
        )
        report = await combine.async_combine(self.hass, self.entry, "Weather", apply=True)
        self.assertEqual(report["merged"], [])
        self.assertIn("pin different slot values", report["refused"][0]["why"])

    def test_the_pin_of_each_sentence_is_known(self):
        from custom_components.irabelle_home_learning.learner import normalize_sentence

        pins = pins_of(
            [
                {"sentences": ["what's the weather {day}"]},
                {"sentences": ["what's the weather tomorrow"], "slots": {"day": "tomorrow"}},
            ]
        )
        self.assertEqual(pins[normalize_sentence("what's the weather tomorrow")], {"day": "tomorrow"})


class Combining(fixtures.WorldCase):
    """The judge's verdict is respected, and the live matcher has the last word."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.entry.options["judge"] = "jev"
        self.spec = LearnedIntent(
            name="Weather",
            template="It is rainy.",
            file="custom_sentences/en/learned-weather.yaml",
        )
        self.entry.runtime_data.intents.append(self.spec)
        self.path = pathlib.Path(self.hass.config.path(self.spec.file))
        _write_sentences(
            self.path,
            "Weather",
            ["what's the weather today", "what's the weather tomorrow"],
        )
        self.original = combine.async_ask

    async def asyncTearDown(self) -> None:
        combine.async_ask = self.original
        await super().asyncTearDown()

    def answer(self, noul: float) -> None:
        """Stub the judge: every pair scores `noul`."""

        class Answer:
            def __init__(self, value: float) -> None:
                self.noul = value
                self.choice = None

        async def fake_ask(hass, entry, state, questions, **kwargs):
            return {key: Answer(noul) for key in questions}

        combine.async_ask = fake_ask

    def test_the_merged_line_survives_storage(self):
        # The fold is written through the same normaliser every sentence goes through; if that
        # stripped a bracket the house would answer with a template nobody can match.
        from custom_components.irabelle_home_learning.learner import sentence_for_storage

        for line in (
            "what's the weather (today | tomorrow)",
            "what's the weather [like]",
        ):
            with self.subTest(line=line):
                self.assertEqual(sentence_for_storage(line), line)

    async def test_a_confident_judge_gets_a_proposal_counted(self):
        self.answer(0.95)
        report = await combine.async_combine(self.hass, self.entry, "Weather")
        self.assertEqual(report["considered"], 1)
        self.assertEqual(len(report["merged"]), 1)
        self.assertEqual(report["merged"][0]["into"], "what's the weather (today | tomorrow)")
        self.assertFalse(report["applied"], "a dry run writes nothing")
        self.assertIn("what's the weather today", self.path.read_text())

    async def test_an_unsure_judge_is_refused_with_its_score(self):
        self.answer(0.30)
        report = await combine.async_combine(self.hass, self.entry, "Weather")
        self.assertEqual(report["merged"], [])
        self.assertIn("scored it 0.30", report["refused"][0]["why"])

    async def test_applying_replaces_the_two_with_the_one(self):
        self.answer(0.95)
        report = await combine.async_combine(self.hass, self.entry, "Weather", apply=True)
        self.assertTrue(report["applied"])
        import yaml

        stored = yaml.safe_load(self.path.read_text())
        self.assertEqual(
            stored["intents"]["Weather"]["data"][0]["sentences"],
            ["what's the weather (today | tomorrow)"],
        )

    async def test_without_the_judge_nothing_is_guessed(self):
        self.entry.options["judge"] = "none"
        with self.assertRaises(LearnerError) as caught:
            await combine.async_combine(self.hass, self.entry, "Weather")
        self.assertIn("needs the judge", str(caught.exception))
