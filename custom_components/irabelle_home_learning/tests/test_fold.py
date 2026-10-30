"""Folding a group into templates: the model proposes, the matcher and the judge prove.

The processing model and the classification model are both a fake `ai_task.generate_data`
here, told apart by task name - what is under test is what this component does with their
answers, against the real hassil: what it accepts, what it refuses, and that a refused fold
leaves the file exactly as it was.
"""

from __future__ import annotations

import pathlib
import unittest

import yaml
from support import HassCase

from custom_components.irabelle_home_learning import fold
from homeassistant.core import SupportsResponse

D2 = {"name": "Divoom Ditoo Keyboard Toggle"}
LINES = [
    "Press the D2 keyboard",
    "Toggle the D2 keyboard",
    "Togl the D2 keyboard",
    "Toggle D2 keyboard",
    "Together the keyboard",
]


class Folding(HassCase):
    async def asyncSetUp(self) -> None:  # noqa: N802
        await super().asyncSetUp()
        self.proposal = "(press|toggle|togl|together) [the] [D2] keyboard"
        self.judged = 0.95
        self.calls: list[str] = []

        async def generate(call):
            self.calls.append(call.data["task_name"])
            if call.data["task_name"] == "classify":
                return {"data": {name.split("__")[0] + "__true": self.judged for name in call.data["structure"]}}
            return {"data": {"templates": self.proposal, "reason": "folded"}}

        self.hass.services.async_register(
            "ai_task", "generate_data", generate, supports_response=SupportsResponse.ONLY
        )
        self.hass.states.async_set("ai_task.model", "unknown")
        self.entry.options.update(judge="ai_task.model", ai_task_entity="ai_task.model")
        self.path = pathlib.Path(
            self.write_sentences_file(
                "learned-hassturnon.yaml",
                yaml.safe_dump(
                    {
                        "language": "en",
                        "intents": {
                            "HassTurnOn": {
                                "data": [
                                    {"sentences": ["turn on the office light"], "slots": {"name": "Office Light"}},
                                    {"sentences": list(LINES), "slots": dict(D2), "response": "d2"},
                                ]
                            }
                        },
                        "responses": {"intents": {"HassTurnOn": {"d2": "Pressed the D2 keyboard."}}},
                    },
                    sort_keys=False,
                ),
            )
        )

    def group(self) -> fold.Group:
        return next(g for g in fold.groups(self.hass, "en", "HassTurnOn") if g.pin == D2)

    def stored(self) -> dict:
        return yaml.safe_load(self.path.read_text())["intents"]["HassTurnOn"]["data"][1]

    async def test_a_proved_fold_replaces_the_group_and_keeps_its_target_and_reply(self):
        outcome = await fold.async_fold_group(self.hass, self.entry, self.group(), apply=True)
        self.assertTrue(outcome.applied, outcome.why)
        block = self.stored()
        self.assertEqual(block["sentences"], [self.proposal])
        self.assertEqual(block["slots"], D2)
        self.assertEqual(block["response"], "d2")

    async def test_the_judge_sees_only_what_the_fold_adds(self):
        outcome = await fold.async_fold_group(self.hass, self.entry, self.group(), apply=False)
        self.assertTrue(outcome.samples)
        said = {line.lower() for line in LINES}
        self.assertFalse(said & set(outcome.samples), "a sentence somebody said is not a question")

    async def test_a_fold_that_misses_a_phrasing_is_refused(self):
        self.proposal = "(press|toggle) the D2 keyboard"
        outcome = await fold.async_fold_group(self.hass, self.entry, self.group(), apply=True)
        self.assertFalse(outcome.applied)
        self.assertIn("do not answer", outcome.why)
        self.assertEqual(self.stored()["sentences"], LINES)

    async def test_a_fold_that_takes_another_groups_sentence_is_refused(self):
        self.proposal = "(press|toggle|togl|together|turn on) [the] [D2|office] (keyboard|light)"
        outcome = await fold.async_fold_group(self.hass, self.entry, self.group(), apply=True)
        self.assertFalse(outcome.applied)
        self.assertIn("already answers", outcome.why)

    async def test_a_doubting_judge_refuses_the_fold(self):
        self.judged = 0.3
        outcome = await fold.async_fold_group(self.hass, self.entry, self.group(), apply=True)
        self.assertFalse(outcome.applied)
        self.assertIn("doubts", outcome.why)
        self.assertEqual(self.stored()["sentences"], LINES)

    async def test_new_syntax_is_refused_before_anything_is_checked(self):
        self.proposal = "<press> the {device} keyboard"
        outcome = await fold.async_fold_group(self.hass, self.entry, self.group(), apply=True)
        self.assertFalse(outcome.applied)
        self.assertNotIn("classify", self.calls)

    async def test_without_a_classifier_nothing_is_proposed(self):
        self.entry.options["judge"] = "none"
        outcome = await fold.async_fold_group(self.hass, self.entry, self.group(), apply=True)
        self.assertFalse(outcome.applied)
        self.assertEqual(self.calls, [])

    async def test_the_automatic_pass_asks_once_per_change(self):
        fold._ATTEMPTED.clear()  # noqa: SLF001
        self.judged = 0.3
        await fold.async_fold(self.hass, self.entry, "HassTurnOn", apply=True, only_ready=True)
        asked = len(self.calls)
        await fold.async_fold(self.hass, self.entry, "HassTurnOn", apply=True, only_ready=True)
        self.assertEqual(len(self.calls), asked, "an unchanged group is not asked about again")


class Expansions(unittest.TestCase):
    def test_what_was_said_is_not_new_whatever_its_case(self):
        group = fold.Group(pathlib.Path("x.yaml"), "HassTurnOn", 0, ["Toggle D2 keyboard"], dict(D2), {})
        added = fold.added_sentences(group, ["(press|toggle) D2 keyboard"])
        self.assertEqual(added, ["press D2 keyboard"])

    def test_a_template_expands_to_every_sentence_it_accepts(self):
        found = fold.expansions("(press|toggle) [the] D2 keyboard", {})
        self.assertEqual(
            found,
            ["press D2 keyboard", "press the D2 keyboard", "toggle D2 keyboard", "toggle the D2 keyboard"],
        )


if __name__ == "__main__":
    unittest.main()
