"""The classification model is configuration: every backend answers in one shape.

The decisions downstream read `choice`, `confidence`, `margin()` and `noul` and never ask who
answered. These tests stand in a fake `ai_task.generate_data` and a fake `jev.ask` - the
other integrations' actions, not this component's code - and check that what each returns is
read back into the same `Answer`, and that what each is *sent* is what it expects.
"""

from __future__ import annotations

import unittest

from support import HassCase

from custom_components.irabelle_home_learning import judge
from custom_components.irabelle_home_learning.judge import (
    Choice,
    JudgeError,
    JudgeNotConfigured,
    Noul,
    Score,
    async_ask,
)
from homeassistant.core import SupportsResponse

QUESTIONS = {
    "route": Choice("Which intent?", {"Weather": "the weather", "none_of_these": "nothing"}),
    "one_request": Noul("One thing?", true="one", false="two"),
    "quality": Score("How good?", ["bad", "fine", "great"]),
}


class AiTaskBackend(HassCase):
    async def asyncSetUp(self) -> None:  # noqa: N802
        await super().asyncSetUp()
        self.calls: list[dict] = []
        self.reply: dict = {}

        async def generate(call):
            self.calls.append(dict(call.data))
            return {"data": self.reply}

        self.hass.services.async_register(
            "ai_task", "generate_data", generate, supports_response=SupportsResponse.ONLY
        )
        self.hass.states.async_set("ai_task.small", "unknown")
        self.entry.options["judge"] = "ai_task.small"

    async def test_answers_come_back_in_the_jev_shape(self):
        self.reply = {
            "route__choice": "Weather",
            "route__probability": 0.8,
            "route__runner_up": "none_of_these",
            "route__runner_up_probability": 0.15,
            "one_request__true": 0.9,
            "quality__level": "2",
            "quality__probability": 0.7,
        }
        answers = await async_ask(self.hass, self.entry, {"request": "x"}, QUESTIONS)
        self.assertEqual(answers["route"].choice, "Weather")
        self.assertAlmostEqual(answers["route"].confidence, 0.8)
        self.assertAlmostEqual(answers["route"].margin(), 0.65)
        self.assertAlmostEqual(answers["one_request"].noul, 0.9)
        self.assertEqual(answers["quality"].score, 2.0)

    async def test_one_call_carries_every_question(self):
        self.reply = {"one_request__true": 0.5}
        await async_ask(self.hass, self.entry, {"request": "x"}, QUESTIONS)
        self.assertEqual(len(self.calls), 1)
        sent = self.calls[0]
        self.assertEqual(sent["entity_id"], "ai_task.small")
        self.assertEqual(
            sent["structure"]["route__choice"]["selector"]["select"]["options"],
            ["Weather", "none_of_these"],
        )
        self.assertIn("one_request__true", sent["structure"])
        self.assertIn("true means: one", sent["instructions"])

    async def test_an_option_that_does_not_exist_is_no_answer(self):
        self.reply = {"route__choice": "Lights", "route__probability": 0.99}
        answers = await async_ask(self.hass, self.entry, "x", {"route": QUESTIONS["route"]})
        self.assertIsNone(answers["route"].choice)

    async def test_a_missing_runner_up_is_not_certainty(self):
        self.reply = {"route__choice": "Weather", "route__probability": 0.6}
        answers = await async_ask(self.hass, self.entry, "x", {"route": QUESTIONS["route"]})
        self.assertAlmostEqual(answers["route"].margin(), 0.2)

    async def test_percentages_are_read_as_probabilities(self):
        self.reply = {"one_request__true": 85}
        answers = await async_ask(
            self.hass, self.entry, "x", {"one_request": QUESTIONS["one_request"]}
        )
        self.assertAlmostEqual(answers["one_request"].noul, 0.85)

    async def test_the_processing_model_classifies_when_asked_to(self):
        self.entry.options["judge"] = "ai_task"
        self.entry.options["ai_task_entity"] = "ai_task.small"
        self.reply = {"one_request__true": 0.2}
        await async_ask(self.hass, self.entry, "x", {"one_request": QUESTIONS["one_request"]})
        self.assertEqual(self.calls[0]["entity_id"], "ai_task.small")

    async def test_a_missing_model_is_a_judge_error(self):
        self.entry.options["judge"] = "ai_task.gone"
        with self.assertRaises(JudgeNotConfigured):
            await async_ask(self.hass, self.entry, "x", QUESTIONS)

    async def test_none_asks_nothing(self):
        self.entry.options["judge"] = "none"
        self.assertFalse(judge.has_judge(self.entry))
        with self.assertRaises(JudgeError):
            await async_ask(self.hass, self.entry, "x", QUESTIONS)
        self.assertEqual(self.calls, [])


class JevIntegrationBackend(HassCase):
    async def asyncSetUp(self) -> None:  # noqa: N802
        await super().asyncSetUp()
        self.calls: list[dict] = []

        async def ask(call):
            self.calls.append(dict(call.data))
            return {
                "answers": {
                    "route": {
                        "type": "choice",
                        "choice": "Weather",
                        "probabilities": {"Weather": 0.9, "none_of_these": 0.1},
                        "confidence": 0.9,
                    },
                    "one_request": {"type": "noul", "noul": 0.97},
                },
                "usage": {"input_tokens": 120, "output_tokens": 0},
            }

        self.hass.services.async_register(
            "jev", "ask", ask, supports_response=SupportsResponse.ONLY
        )
        self.entry.options["judge"] = "jev_integration"
        # Loaded means a loaded jev config entry; the action alone stands in for it here.
        self._loaded = judge._jev_integration_loaded  # noqa: SLF001
        judge._jev_integration_loaded = lambda hass: True  # noqa: SLF001
        judge._USAGE.clear()  # noqa: SLF001

    async def asyncTearDown(self) -> None:  # noqa: N802
        judge._jev_integration_loaded = self._loaded  # noqa: SLF001
        await super().asyncTearDown()

    async def test_noul_labels_go_top_level(self):
        await async_ask(
            self.hass,
            self.entry,
            {"request": "x"},
            {"route": QUESTIONS["route"], "one_request": QUESTIONS["one_request"]},
        )
        sent = self.calls[0]["questions"]["one_request"]
        self.assertEqual((sent["true"], sent["false"]), ("one", "two"))
        self.assertNotIn("criteria", sent)

    async def test_answers_and_usage_are_read(self):
        answers = await async_ask(
            self.hass,
            self.entry,
            {"request": "x"},
            {"route": QUESTIONS["route"], "one_request": QUESTIONS["one_request"]},
        )
        self.assertAlmostEqual(answers["route"].margin(), 0.8)
        self.assertAlmostEqual(answers["one_request"].noul, 0.97)
        self.assertEqual(judge.usage_of(self.entry).input_tokens, 120)

    async def test_a_template_looking_state_is_not_rendered(self):
        await async_ask(
            self.hass,
            self.entry,
            "{{ states('sun.sun') }}",
            {"one_request": QUESTIONS["one_request"]},
        )
        self.assertEqual(self.calls[0]["state_template"], {"text": "{{ states('sun.sun') }}"})


if __name__ == "__main__":
    unittest.main()
