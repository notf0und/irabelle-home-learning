"""A binding keeps the target the answer acted on - even when speech-to-text misheard it.

"Toggle the D2 keyword" ran HassTurnOn on the D2 keyboard. The words do not contain the
keyboard's name, so the literal check dropped the target, and the router then bound the words
to HassTurnOn with nothing to turn on: a sentence that matches and fails, for ever.
"""

from __future__ import annotations

import pathlib
import unittest

from support import HassCase

from custom_components.irabelle_home_learning import autolearn
from custom_components.irabelle_home_learning.learner import Observation
from homeassistant.core import SupportsResponse

D2 = {"name": "Divoom Ditoo Keyboard Toggle"}


class MisheardTargets(HassCase):
    async def asyncSetUp(self) -> None:  # noqa: N802
        await super().asyncSetUp()
        self.score = 0.95

        async def generate(call):
            return {"data": {"names_it__true": self.score}}

        self.hass.services.async_register(
            "ai_task", "generate_data", generate, supports_response=SupportsResponse.ONLY
        )
        self.hass.states.async_set("ai_task.model", "unknown")
        self.entry.options["judge"] = "ai_task.model"

    async def test_a_misheard_name_is_pinned_when_the_judge_hears_it(self):
        self.assertIsNone(autolearn._reproducible_slots(dict(D2), "Toggle the D2 keyword"))  # noqa: SLF001
        slots = await autolearn.async_heard_target(
            self.hass, self.entry, "Toggle the D2 keyword", {**D2, "domain": ["switch"]}
        )
        self.assertEqual(slots, D2, "the target is kept, the context is not")

    async def test_words_that_name_nothing_stay_refused(self):
        self.score = 0.1
        self.assertIsNone(
            await autolearn.async_heard_target(self.hass, self.entry, "turn it off", dict(D2))
        )

    async def test_no_judge_means_no_guess(self):
        self.entry.options["judge"] = "none"
        self.assertIsNone(
            await autolearn.async_heard_target(self.hass, self.entry, "Toggle the D2 keyword", dict(D2))
        )


class TargetlessBindings(HassCase):
    async def test_a_targeted_intent_is_never_bound_without_a_target(self):
        row = Observation(utterance="Toggle the D2 keyword", normalized="toggle the d2 keyword")
        result = await autolearn._bind(self.hass, self.entry, row, "HassTurnOn", {})  # noqa: SLF001
        self.assertFalse(result["ok"])
        self.assertEqual(row.state, "ignored")
        written = pathlib.Path(self.sentences_dir())
        self.assertFalse(written.is_dir() and any(written.iterdir()), "nothing was written")


if __name__ == "__main__":
    unittest.main()
