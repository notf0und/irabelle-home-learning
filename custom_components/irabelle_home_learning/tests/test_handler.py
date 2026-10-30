"""The answer path: a registered learned intent, asked the way Home Assistant asks it.

Everything else in the suite tests what gets written *down*. This tests what is actually said,
through `intent.async_handle` — the call the pipeline makes — including the two things that
only exist at answer time: the slots the sentence matched, and who is asking.
"""

from __future__ import annotations

import unittest

import fixtures

from homeassistant.helpers import intent as intent_helper

from custom_components.irabelle_home_learning.intents import register_intent, write_response
from custom_components.irabelle_home_learning.learner import LearnedIntent, async_load, async_save


class Answering(fixtures.WorldCase):
    def register(self, **spec) -> LearnedIntent:
        intent = LearnedIntent(**spec)
        self.entry.runtime_data.intents = [intent]
        register_intent(self.hass, intent)
        return intent

    async def say(self, slots: dict | None = None) -> str:
        response = await intent_helper.async_handle(
            self.hass, "conversation", "ProbeIntent", slots or {}, text_input="anything"
        )
        return response.speech["plain"]["speech"]

    async def test_a_slot_the_sentence_matched_reaches_the_template(self):
        # The whole point of a slot: the sentence matched "on monday", so the reply says
        # Monday without a template per day.
        self.register(name="ProbeIntent", template="On {{ day }} it will rain.")
        self.assertEqual(await self.say({"day": {"value": "monday"}}), "On monday it will rain.")

    async def test_an_explicit_reply_wins_over_the_template(self):
        # `set_response` writes one; the template keeps computing what it needs.
        self.register(
            name="ProbeIntent",
            template="On {{ day }} it will rain.",
            response="Rain expected on {{ day }}.",
        )
        self.assertEqual(
            await self.say({"day": {"value": "tuesday"}}), "Rain expected on tuesday."
        )

    async def test_the_speaker_variables_are_always_defined(self):
        # A template may read them without a guard, from a satellite with nobody signed in.
        self.register(
            name="ProbeIntent",
            template="{{ user_name }}/{{ person_name }}/{{ speaker_certain }}/{{ user_area }}",
        )
        self.assertEqual(await self.say(), "///")

    async def test_a_recognised_voice_sets_the_certainty_a_template_branches_on(self):
        fixtures.add_entity(
            self.hass, "sensor.voicebm_current_speaker", "Sam", attributes={"confidence": 0.9}
        )
        self.register(
            name="ProbeIntent",
            template="{% if speaker_certain %}You are "
            "{{ states('sensor.voicebm_current_speaker') }}."
            "{% else %}I am not sure who is asking.{% endif %}",
        )
        self.assertEqual(await self.say(), "You are Sam.")

    async def test_a_low_confidence_match_is_not_certain(self):
        fixtures.add_entity(
            self.hass, "sensor.voicebm_current_speaker", "Sam", attributes={"confidence": 0.2}
        )
        self.register(
            name="ProbeIntent",
            template="{% if speaker_certain %}sure{% else %}not sure{% endif %}",
        )
        self.assertEqual(await self.say(), "not sure")

    async def test_a_reply_edited_in_the_file_is_what_the_house_says(self):
        # The file is authoritative, not the store: a reply a person edited there wins on the
        # next restart, which is the whole reason replies live beside the sentences.
        self.register(name="ProbeIntent", template="The store says {{ day }}.")
        await async_save(self.hass, self.entry)
        write_response(
            self.write_file("learned-probeintent.yaml", "language: en\nintents:\n  ProbeIntent:\n    data:\n    - sentences:\n      - what's the weather on {day}\n"),
            "ProbeIntent",
            "The file says {{ day }}.",
        )

        reloaded = await async_load(self.hass, self.entry)

        self.assertEqual(reloaded.intents[0].response, "The file says {{ day }}.")
        self.entry.runtime_data = reloaded
        register_intent(self.hass, reloaded.intents[0])
        self.assertEqual(await self.say({"day": {"value": "friday"}}), "The file says friday.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
