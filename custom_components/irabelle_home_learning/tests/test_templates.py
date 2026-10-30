"""What every template must pass before it is installed, and why each refusal exists.

The refusals are the component's teeth: a template that reads an entity which is not there
renders the word *unknown* and the house says it out loud, a variable nothing defines renders
blank for ever, and an attribute that does not exist degrades *silently* — which is the one
failure a mechanical check has to guess at, so it warns instead.
"""

from __future__ import annotations

import unittest

from support import HassCase

from custom_components.irabelle_home_learning.intents import (
    _check_template,
    _did_you_mean,
    _reads,
    missing_attributes,
    undeclared_variables,
)
from custom_components.irabelle_home_learning.learner import LearnerError


class Validator(HassCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.hass.states.async_set(
            "weather.home",
            "rainy",
            {
                "temperature": 14.0,
                "humidity": 61,
                "forecast_daily": [{"condition": "rainy", "temperature": "16", "templow": "12"}],
                "forecast_hourly": [],
            },
        )
        self.hass.states.async_set("sensor.home_temperature", "14.4")

    def test_a_real_template_is_accepted(self):
        _check_template(
            self.hass,
            "It is {{ states('weather.home') }}, {{ state_attr('weather.home','temperature') }} degrees.",
        )

    def test_a_missing_entity_is_refused_and_the_nearest_id_is_named(self):
        with self.assertRaises(LearnerError) as caught:
            _check_template(self.hass, "{{ state_attr('weather.home_home_forecast','forecast') }}")
        message = str(caught.exception)
        self.assertIn("weather.home_home_forecast", message)
        # This is the hint that breaks a model's retry loop: the friendly name is not an id.
        self.assertIn("-> weather.home", message)

    def test_a_variable_nothing_defines_is_refused(self):
        with self.assertRaises(LearnerError) as caught:
            _check_template(self.hass, "{% if true %}{% set x = 1 %}{% endif %}{{ x }}")
        self.assertIn("reads x", str(caught.exception))

    def test_a_declared_slot_is_allowed(self):
        source = "{% if day is defined %}On {{ day }}{% endif %}"
        with self.assertRaises(LearnerError):
            _check_template(self.hass, source)
        _check_template(self.hass, source, defined={"day"}, sample={"day": "today"})

    def test_a_template_that_renders_None_is_refused(self):
        # last_updated lives on the state, not in the attributes: this renders the word None.
        with self.assertRaises(LearnerError) as caught:
            _check_template(self.hass, "Seen {{ state_attr('weather.home','last_updated') }}")
        self.assertIn("renders", str(caught.exception))

    def test_strftime_on_a_utc_timestamp_is_refused(self):
        # It prints the right minute with the wrong half of the day.
        with self.assertRaises(LearnerError) as caught:
            _check_template(
                self.hass, "{{ states.sensor.home_temperature.last_changed.strftime('%H') }}"
            )
        self.assertIn("UTC", str(caught.exception))

    def test_undeclared_variables_honours_the_declared_slots(self):
        self.assertEqual(undeclared_variables(self.hass, "{{ day }}"), {"day"})
        self.assertEqual(undeclared_variables(self.hass, "{{ day }}", {"day"}), set())


class MissingAttributes(HassCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.hass.states.async_set("weather.home", "rainy", {"temperature": 14.0, "humidity": 61})

    def test_a_missing_attribute_is_reported_with_what_exists(self):
        warnings = missing_attributes(self.hass, "{{ state_attr('weather.home','forecast') }}")
        self.assertEqual(len(warnings), 1)
        self.assertIn("weather.home.forecast", warnings[0])
        self.assertIn("temperature", warnings[0])

    def test_a_real_attribute_is_silent(self):
        self.assertEqual(
            missing_attributes(self.hass, "{{ state_attr('weather.home','temperature') }}"), []
        )

    def test_reading_nothing_at_all_is_silent(self):
        self.assertEqual(missing_attributes(self.hass, "It is warm."), [])

    def test_a_state_field_is_not_an_attribute(self):
        # last_changed is on the state object; warning about it would be noise.
        self.assertEqual(
            missing_attributes(self.hass, "{{ states.weather.home.last_changed }}"), []
        )


class EntityReading(HassCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.hass.states.async_set("weather.home", "rainy", {"temperature": 14.0})
        self.hass.states.async_set("sensor.home_temperature", "14.4")

    def test_reads_names_the_ids_with_their_state(self):
        # Returned with every write, so a model sees the ids it actually used.
        self.assertEqual(
            _reads(self.hass, "{{ states('weather.home') }}"),
            {"weather.home": "rainy"},
        )

    def test_reads_marks_something_that_is_not_there(self):
        self.assertEqual(
            _reads(self.hass, "{{ states('sensor.gone') }}"), {"sensor.gone": "(not in this house)"}
        )

    def test_quoted_ids_are_seen_even_when_a_variable_carries_them(self):
        # `template_entities` cannot see this; the quoted-id check can, which is how a
        # template that picks between two sensors per speaker is still validated.
        source = "{% set e = 'sensor.home_temperature' %}{{ states(e) }}"
        self.assertEqual(_reads(self.hass, source), {"sensor.home_temperature": "14.4"})

    def test_the_nearest_id_is_suggested(self):
        self.assertEqual(
            _did_you_mean(self.hass, ["weather.home_home_forecast"]),
            "Did you mean: weather.home_home_forecast -> weather.home? ",
        )

    def test_nothing_to_suggest_says_nothing(self):
        self.assertEqual(_did_you_mean(self.hass, ["light.nowhere_at_all"]), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
