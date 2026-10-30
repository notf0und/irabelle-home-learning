"""What the screen is shown, and what it makes of the answers.

The screen is a prompt, and its *state* is the prompt. Two of the tuning fixes in its history
were state fixes rather than threshold fixes — a speaker-aware template judged on its fallback
render, and a temperature answer judged from `home = rainy` — so the state is pinned here
where a change to it is visible, and the judge itself is stubbed: this tests what we ask, not
what the model answers.
"""

from __future__ import annotations

import unittest

import fixtures

from custom_components.irabelle_home_learning import autolearn
from custom_components.irabelle_home_learning.autolearn import (
    SCREEN_CLEAN,
    SCREEN_MARGIN,
    _screen_state,
    async_screen_learning,
)
from custom_components.irabelle_home_learning.const import CONF_JUDGE, JUDGE_JEV
from custom_components.irabelle_home_learning.judge import Answer
from custom_components.irabelle_home_learning.learner import LearnedIntent

QUESTIONS = (
    "answers_the_request",
    "same_scope",
    "reads_a_real_entity",
    "reads_the_right_entity",
    "answers_the_speaker",
    "time_looks_right",
    "reads_like_a_sentence",
)


def answer(noul: float = 0.9) -> Answer:
    return Answer(raw={"noul": noul, "confidence": 0.9})


class State(fixtures.WorldCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        fixtures.add_entity(self.hass, "weather.home", "rainy", name="Home Home Forecast",
                            attributes={"temperature": 14.0})
        fixtures.add_entity(self.hass, "sensor.home_temperature", "14", name="Home Temperature")
        fixtures.add_person(self.hass, "Sam", [])
        self.spec = LearnedIntent(
            name="WeatherForecast",
            template="On {{ day }} it will be {{ states('weather.home') }}.",
            slots={"day": {"values": ["today", "tomorrow"]}},
            file="custom_sentences/en/learned-weatherforecast.yaml",
        )
        self.entry.runtime_data.intents = [self.spec]
        # `learned_phrases` reads the sentence files, not the store: they are what the matcher
        # loads, and a wording a person edited by hand is still a wording.
        self.write_file(
            "learned-weatherforecast.yaml",
            "language: en\nintents:\n  WeatherForecast:\n    data:\n    - sentences:\n"
            "      - what is the temperature outside\n",
        )

    async def test_a_slotted_template_is_shown_both_ways_it_runs(self):
        state = _screen_state(self.hass, self.entry, None, self.spec)
        rendered = state["what_the_template_says"]
        # "what is the temperature outside" carries no slot; "what's the weather on monday"
        # does. Showing only the filled one made every person's answer the sample person's.
        self.assertIn("with no slot given - Sam", rendered)
        self.assertIn("with day=today - Sam", rendered)
        self.assertIn("On today it will be rainy.", rendered.values())

    async def test_the_entities_it_reads_come_with_their_attributes(self):
        state = _screen_state(self.hass, self.entry, None, self.spec)
        line = state["the_entities_this_template_reads"]["weather.home"]
        # Without the attribute names, a temperature answer looks like a rain answer.
        self.assertIn("temperature", line)

    async def test_the_alternatives_are_shown_beside_what_it_reads(self):
        state = _screen_state(self.hass, self.entry, None, self.spec)
        alternatives = state["the_real_entities_that_could_answer_this"]
        self.assertIn("sensor.home_temperature", alternatives)

    async def test_the_request_falls_back_to_the_phrasings_for_an_audit(self):
        # An audit has no observation behind it; asking "does this answer '(auditing)'" is
        # how a correct learning got flagged too_wide.
        state = _screen_state(self.hass, self.entry, None, self.spec)
        self.assertEqual(state["request"], "what is the temperature outside")


class Verdict(fixtures.WorldCase):
    """`async_screen_learning` with the judge stubbed, so the reasons are testable."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        fixtures.add_entity(self.hass, "weather.home", "rainy", attributes={"temperature": 14.0})
        self.spec = LearnedIntent(
            name="WeatherForecast",
            template="It is {{ states('weather.home') }}.",
            file="custom_sentences/en/learned-weatherforecast.yaml",
        )
        self.entry.runtime_data.intents = [self.spec]
        self.write_file(
            "learned-weatherforecast.yaml",
            "language: en\nintents:\n  WeatherForecast:\n    data:\n    - sentences:\n"
            "      - what's the weather\n",
        )
        self.entry.options[CONF_JUDGE] = JUDGE_JEV
        self.original = autolearn.async_ask

    async def asyncTearDown(self) -> None:
        autolearn.async_ask = self.original
        await super().asyncTearDown()

    def stub(self, **nouls: float) -> dict:
        """Answer every question 0.9 unless told otherwise; remember the state we sent."""
        captured: dict = {}

        async def fake_ask(hass, entry, state, questions, **kwargs):
            captured["state"] = state
            captured["questions"] = list(questions)
            answers = {}
            for name in questions:
                if name == "what_is_wrong":
                    answers[name] = Answer(
                        raw={"choice": "correct", "confidence": 0.9,
                             "probabilities": {"correct": 0.9, "too_wide": 0.1}}
                    )
                else:
                    answers[name] = answer(nouls.get(name, 0.9))
            return answers

        autolearn.async_ask = fake_ask
        return captured

    async def screen(self, **nouls: float):
        self.stub(**nouls)
        return await async_screen_learning(self.hass, self.entry, None, self.spec)

    async def test_a_clean_screen_has_no_reasons(self):
        screen = await self.screen()
        self.assertTrue(screen.checked)
        self.assertTrue(screen.clean)
        self.assertEqual(screen.reasons, [])

    async def test_a_low_answer_becomes_a_reason_naming_the_question(self):
        screen = await self.screen(same_scope=0.1)
        self.assertFalse(screen.clean)
        self.assertIn(f"same_scope=0.1", screen.reasons)

    async def test_every_question_can_flag_on_its_own(self):
        for name in QUESTIONS:
            screen = await self.screen(**{name: SCREEN_CLEAN - 0.1})
            self.assertFalse(screen.clean, f"{name} did not flag")
            self.assertIn(f"{name}=", " ".join(screen.reasons))

    async def test_a_near_tie_on_correct_is_not_a_verdict(self):
        async def near_tie(hass, entry, state, questions, **kwargs):
            answers = {}
            for name in questions:
                if name == "what_is_wrong":
                    margin = SCREEN_MARGIN / 2
                    answers[name] = Answer(
                        raw={"choice": "correct", "confidence": 0.4,
                             "probabilities": {"correct": 0.4, "too_wide": 0.4 - margin}}
                    )
                else:
                    answers[name] = answer()
            return answers

        autolearn.async_ask = near_tie
        screen = await async_screen_learning(self.hass, self.entry, None, self.spec)
        self.assertFalse(screen.clean)
        self.assertIn("near-tie", " ".join(screen.reasons))

    async def test_the_verdict_is_recorded_with_its_numbers(self):
        # A threshold can only be tuned against numbers that were kept.
        screen = await self.screen(reads_like_a_sentence=0.55)
        recorded = screen.as_dict()["answers"]["reads_like_a_sentence"]
        self.assertEqual(recorded["noul"], 0.55)
        self.assertEqual(recorded["confidence"], 0.9)

    async def test_without_a_judge_there_is_no_screen_at_all(self):
        self.entry.options.pop(CONF_JUDGE)
        self.assertIsNone(await async_screen_learning(self.hass, self.entry, None, self.spec))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class WhatTheAnswerRead(unittest.TestCase):
    """The slots the answering turn used, and which of them the words actually say.

    Reported: "set the office light brightness to warm" did nothing and said nothing. The
    assistant had answered it by setting a temperature of 2700 K, and the learner saw that -
    then threw it away, because the words do not contain "2700". It learned the words without
    the meaning, and the phrase matched with no value to act on.
    """

    def test_the_value_the_answer_chose_is_kept(self):
        slots = autolearn._reproducible_slots(
            {"name": "office light", "temperature": 2700},
            "set the office light brightness to warm",
        )
        self.assertEqual(slots, {"name": "office light", "temperature": 2700})

    def test_a_named_value_is_still_named(self):
        slots = autolearn._reproducible_slots(
            {"name": "office light", "brightness": 40}, "set the office light to 40 percent"
        )
        self.assertEqual(slots["brightness"], 40)

    def test_a_phrasing_that_names_nothing_is_refused(self):
        # "make it nicer" does not say what it wants; pinning the model's guess would make the
        # words mean one thing for ever.
        self.assertIsNone(
            autolearn._reproducible_slots({"temperature": 2700}, "make it nicer")
        )
        self.assertIsNone(autolearn._reproducible_slots({}, "turn it off"))

    def test_context_slots_are_still_left_out(self):
        slots = autolearn._reproducible_slots(
            {"name": "office light", "preferred_area_id": "office"},
            "set the office light to warm",
        )
        self.assertEqual(slots, {"name": "office light"})
