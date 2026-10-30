"""The model-facing surface: the teaching tools, and what they hand back.

These are the tools a fallback model calls, so their output *is* the instructions it acts on.
The one that matters most is `ListEntities`: a model shown a friendly name writes an id that
does not exist, and this is the list that lets it copy a real one instead.
"""

from __future__ import annotations

import unittest

import fixtures

from homeassistant.core import Context
from homeassistant.helpers.llm import LLMContext, ToolInput

from custom_components.irabelle_home_learning.autolearn import _candidate_entities, _exposed_entities
from custom_components.irabelle_home_learning.learner import (
    LearnedIntent,
    entity_aliases,
    entity_label,
    entity_owners,
)


class EntityLegibility(fixtures.WorldCase):
    """What a model is *shown* about an entity — the alias, its owner, its exposure."""

    async def test_an_alias_is_what_a_model_sees_and_the_id_is_unchanged(self):
        fixtures.add_entity(
            self.hass,
            "sensor.mac_area_last_seen",
            "Office",
            name="Area Last Seen",
            aliases={"Sam's watch"},
        )
        self.assertEqual(entity_aliases(self.hass, "sensor.mac_area_last_seen"), ["Sam's watch"])
        label = entity_label(self.hass, "sensor.mac_area_last_seen", "Area Last Seen")
        self.assertIn("Sam's watch", label)
        # The id is untouched: an alias adds words, a rename would take the composed name away.
        self.assertIsNotNone(self.hass.states.get("sensor.mac_area_last_seen"))

    async def test_ownership_comes_from_the_person_tracker_link(self):
        fixtures.add_entity(self.hass, "sensor.mac_area_last_seen", "Office", aliases={"watch"})
        fixtures.add_entity(self.hass, "device_tracker.sams_phone", "home", name="Sam's Phone")
        fixtures.add_person(self.hass, "Sam", ["device_tracker.sams_phone"])
        owners = entity_owners(self.hass)
        self.assertEqual(owners.get("device_tracker.sams_phone"), "Sam")

    async def test_the_exposed_list_names_what_a_template_may_read(self):
        fixtures.add_entity(
            self.hass, "sensor.mac_area_last_seen", "Office", aliases={"Sam's watch"},
            exposed=True,
        )
        block = _exposed_entities(self.hass)
        self.assertIn("sensor.mac_area_last_seen", block)
        self.assertIn("Sam's watch", block)
        self.assertIn("Sam's", block)


class ListEntities(fixtures.WorldCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        from custom_components.irabelle_home_learning.llm import ListEntitiesTool

        self.tool = ListEntitiesTool()
        self.context = LLMContext(
            platform="deepseek_conversation", context=Context(), language="en",
            assistant="conversation", device_id=None,
        )

    async def call(self, **args) -> dict:
        result = await self.tool.async_call(
            self.hass, ToolInput(tool_name=self.tool.name, tool_args=args), self.context
        )
        return result["result"]

    async def test_it_returns_the_real_id_with_its_state_and_attributes(self):
        fixtures.add_entity(
            self.hass, "weather.home", "rainy", name="Home Home Forecast", attributes={"temperature": 14.0}
        )
        result = await self.call(search="weather")
        found = {row["entity_id"]: row for row in result["entities"]}
        self.assertIn("weather.home", found)
        self.assertEqual(found["weather.home"]["state"], "rainy")
        self.assertIn("temperature", found["weather.home"]["attributes"])
        # The friendly name is the trap this tool exists to defuse.
        self.assertEqual(found["weather.home"]["name"], "Home Home Forecast")

    async def test_a_search_only_returns_what_it_names(self):
        fixtures.add_entity(self.hass, "weather.home", "rainy", name="Home Home Forecast")
        fixtures.add_entity(self.hass, "light.kitchen", "on", name="Kitchen Light")
        result = await self.call(search="kitchen light")
        self.assertEqual([row["entity_id"] for row in result["entities"]], ["light.kitchen"])

    async def test_an_empty_search_lists_what_assist_may_read(self):
        fixtures.add_entity(self.hass, "light.kitchen", "on", name="Kitchen Light", exposed=True)
        # `light` is exposed by domain by default, so "not exposed" has to be said explicitly.
        fixtures.add_entity(self.hass, "light.hidden", "on", name="Hidden Light", exposed=False)
        ids = [row["entity_id"] for row in (await self.call())["entities"]]
        self.assertIn("light.kitchen", ids)
        self.assertNotIn("light.hidden", ids)

    async def test_a_search_can_find_something_assist_may_not_read(self):
        # Exposure decides what a person may ask for, not what an answer may look at.
        fixtures.add_entity(self.hass, "sensor.quiet", "1", name="Quiet Sensor", exposed=False)
        
        result = await self.call(search="quiet sensor")
        self.assertEqual(result["entities"][0]["entity_id"], "sensor.quiet")
        self.assertFalse(result["entities"][0]["assist_can_read"])

    async def test_stop_words_do_not_match_everything(self):
        fixtures.add_entity(self.hass, "light.kitchen", "on", name="Kitchen Light")
        result = await self.call(search="what is the weather like")
        self.assertEqual(result["entities"], [])

    async def test_it_tells_the_model_what_to_do_with_the_ids(self):
        result = await self.call(search="nothing at all")
        self.assertIn("exactly as written", result["note"])
        self.assertIn("inventing an id", result["note"])


class TeachingOnly(fixtures.WorldCase):
    """`ListEntities` is part of the teaching surface, so the gate covers it."""

    async def test_a_platform_that_may_not_teach_does_not_get_the_writing_tools(self):
        from custom_components.irabelle_home_learning.llm import (
            LLM_API_ASSIST,
            ListEntitiesTool,
            async_get_tools,
        )
        from custom_components.irabelle_home_learning.const import CONF_ALLOWED_PLATFORMS

        self.entry.options[CONF_ALLOWED_PLATFORMS] = ["deepseek_conversation"]

        def names(platform: str) -> set[str]:
            context = LLMContext(
                platform=platform, context=Context(), language="en",
                assistant="conversation", device_id=None,
            )
            tools = async_get_tools(self.hass, context, LLM_API_ASSIST)
            return {tool.name for tool in (tools.tools if tools else [])}

        self.assertIn(ListEntitiesTool.name, names("deepseek_conversation"))
        self.assertNotIn(ListEntitiesTool.name, names("some_other_agent"))


class Candidates(fixtures.WorldCase):
    """The alternatives the screen is shown, so a real-but-wrong entity can be spotted."""

    async def test_it_offers_the_other_entities_that_match_the_request(self):
        fixtures.add_entity(self.hass, "weather.home", "rainy", name="Home Home Forecast")
        fixtures.add_entity(self.hass, "sensor.home_temperature", "14", name="Home Temperature")
        found = _candidate_entities(self.hass, "what is the temperature outside", "{{ states('weather.home') }}")
        # The template's own entities are not repeated back; the alternatives are.
        self.assertIn("sensor.home_temperature", found)
        self.assertNotIn("weather.home", found)

    async def test_nothing_matching_is_an_empty_answer(self):
        fixtures.add_entity(self.hass, "light.kitchen", "on", name="Kitchen Light")
        self.assertEqual(_candidate_entities(self.hass, "who am i", "{{ person_name }}"), {})


if __name__ == "__main__":
    unittest.main(verbosity=2)


class GetIntent(fixtures.WorldCase):
    """A model asked to improve an answer has to be able to *read* it first."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        from custom_components.irabelle_home_learning.llm import GetIntentTool

        self.tool = GetIntentTool()
        self.context = LLMContext(
            platform="deepseek_conversation", context=Context(), language="en",
            assistant="conversation", device_id=None,
        )
        fixtures.add_entity(self.hass, "weather.home", "rainy", attributes={"temperature": 14.0})
        self.entry.runtime_data.intents.append(
            LearnedIntent(
                name="WeatherForecast",
                template="It is {{ states('weather.home') }}, {{ state_attr('weather.home','temperature') }} degrees.",
                response="It says {{ states('weather.home') }} at {{ state_attr('weather.home','temperature') }}.",
                slots={"day": {"values": ["today", "tomorrow"]}},
                file="custom_sentences/en/learned-weatherforecast.yaml",
            )
        )
        self.write_file(
            "learned-weatherforecast.yaml",
            "language: en\nintents:\n  WeatherForecast:\n    data:\n    - sentences:\n"
            "      - what's the weather\n",
        )

    async def call(self, **args) -> dict:
        return await self.tool.async_call(
            self.hass, ToolInput(tool_name=self.tool.name, tool_args=args), self.context
        )

    async def test_it_hands_back_the_text_to_edit_and_what_it_says_now(self):
        result = (await self.call(name="WeatherForecast"))["result"]
        # The reply written in the sentence file is what the house says, so that is the text
        # to hand back edited - not the template it was derived from.
        self.assertEqual(
            result["the_reply_to_edit"],
            "It says {{ states('weather.home') }} at {{ state_attr('weather.home','temperature') }}.",
        )
        self.assertEqual(result["says_right_now"], "It says rainy at 14.0.")
        self.assertEqual(result["reads"], {"weather.home": "rainy"})
        self.assertIn("what's the weather", result["phrasings"])
        self.assertIn("whole", result["note"])

    async def test_it_says_which_intents_exist_when_the_name_is_wrong(self):
        out = await self.call(name="NotAnIntent")
        self.assertFalse(out["success"])
        self.assertIn("WeatherForecast", out["error"])
        # And that a built-in's answer is not this component's to rewrite.
        self.assertIn("built-in", out["error"])

    async def test_it_warns_about_an_attribute_that_is_not_there(self):
        self.entry.runtime_data.intents[0].response = "The outlook is {{ state_attr('weather.home','forecast') }}."
        result = (await self.call(name="WeatherForecast"))["result"]
        self.assertEqual(len(result["warnings"]), 1)
        self.assertIn("forecast", result["warnings"][0])
