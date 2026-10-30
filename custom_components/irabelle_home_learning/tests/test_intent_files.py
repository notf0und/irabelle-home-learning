"""The files a learned intent lives in.

These are the tests that protect local voice. A sentence whose intent nothing answers makes
recognition itself fail for *every* agent, and a `lists:` block left behind when its sentences
move makes every file that uses those sentences unloadable — both of which this component has
shipped once and neither of which is visible until someone speaks to the house.

The rules pinned here:

* an intent's file is named after the intent, never after the note that came with the write;
* a slot list travels with the sentences that use it, through consolidation and rename;
* forgetting an intent removes its sentences from *every* file, and deletes a file it emptied;
* a merge moves wording rather than dropping it, and leaves nothing behind that could be
  matched with no handler.
"""

from __future__ import annotations

import pathlib
import unittest

import fixtures

from homeassistant.helpers import intent as intent_helper
from homeassistant.util import yaml as yaml_util

from support import HassCase

from custom_components.irabelle_home_learning.intents import (
    _write_lists,
    async_set_response,
    _write_sentences,
    async_consolidate_intents,
    async_learn_intent,
    async_merge_intents,
    async_rename_intent,
    learner_sentences,
    write_response,
)
from custom_components.irabelle_home_learning import learner
from custom_components.irabelle_home_learning.learner import (
    LearnedIntent,
    LearnerError,
    file_responses,
    remove_intent_sentences,
)


def load(path: str | pathlib.Path) -> dict:
    """Read a written file. Paths are always absolute (from the temp config dir) - a
    relative one would resolve against the process CWD and could touch a live install."""
    import yaml

    assert pathlib.Path(path).is_absolute(), f"tests must use absolute paths, got {path!r}"
    return yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8")) or {}


class FilePrimitives(HassCase):
    def test_a_reply_lands_beside_the_sentences(self):
        path = pathlib.Path(self.sentences_dir()) / "learned-thing.yaml"
        _write_sentences(path, "Thing", ["what is the thing"])
        write_response(path, "Thing", "It is the thing.")
        data = load(path)
        self.assertEqual(data["intents"]["Thing"]["data"][0]["sentences"], ["what is the thing"])
        self.assertEqual(data["responses"]["intents"]["Thing"]["default"], "It is the thing.")

    def test_writing_a_reply_keeps_what_is_already_in_the_file(self):
        path = pathlib.Path(self.sentences_dir()) / "learned-thing.yaml"
        _write_sentences(path, "Thing", ["what is the thing"])
        write_response(path, "Thing", "It is the thing.")
        write_response(path, "Thing", "It is still the thing.")
        data = load(path)
        self.assertEqual(data["intents"]["Thing"]["data"][0]["sentences"], ["what is the thing"])
        self.assertEqual(data["responses"]["intents"]["Thing"]["default"], "It is still the thing.")

    def test_sentences_are_stored_in_matchable_form_and_deduplicated(self):
        path = pathlib.Path(self.sentences_dir()) / "learned-thing.yaml"
        _write_sentences(path, "Thing", ["is the thing on?", "is the thing on", "and the thing"])
        self.assertEqual(
            load(path)["intents"]["Thing"]["data"][0]["sentences"],
            ["is the thing on", "and the thing"],
        )

    def test_a_reply_that_reads_a_slot_renders_with_slots_alone(self):
        """The reply has to work in Home Assistant's own renderer, not just ours.

        A reply is stored in `responses:`, and Home Assistant renders that with `slots` in
        scope and nothing else - which is exactly what answers a learned phrase after this
        component is deleted, through the exported `intent_script` stubs. A reply reading
        the bare slot name does not fail there, it silently takes the wrong branch: the
        weather answer asked about tomorrow would talk about today.
        """
        from homeassistant.helpers import template

        path = pathlib.Path(self.sentences_dir()) / "learned-weather.yaml"
        _write_sentences(
            path, "Weather", ["what's the weather {day}"], lists={"day": {"values": ["tomorrow"]}}
        )
        write_response(path, "Weather", "{% if day is defined and day %}On {{ day }}{% else %}Today{% endif %} it rains.")
        stored = load(path)["responses"]["intents"]["Weather"]["default"]

        # The component's own context: bare names and `slots` both.
        both = template.Template(stored, self.hass).async_render(
            {"day": "tomorrow", "slots": {"day": "tomorrow"}}, parse_result=False
        )
        # Home Assistant's: `slots` only, which is what an intent_script response gets.
        theirs = template.Template(stored, self.hass).async_render(
            {"slots": {"day": "tomorrow"}}, parse_result=False
        )
        self.assertEqual(both, "On tomorrow it rains.")
        self.assertEqual(theirs, "On tomorrow it rains.")

    def test_a_reply_may_ask_whether_a_slot_was_given(self):
        """The binding must leave an absent slot *undefined*, not empty.

        A reply is allowed to branch on the difference - `{% if who is defined %}` picks
        between "Sam's phone was last seen in the Office" and "Sam, your phone was
        last seen in the Office". Writing the binding as `slots.who or ''` answered that
        question wrongly and did not fail loudly: measured on the live house, `PhoneLastSeenAgo`
        could no longer reach its speaker branch, so it looked the *other* person's phone up
        and greeted the asker as "Unknown".
        """
        from homeassistant.helpers import template

        path = pathlib.Path(self.sentences_dir()) / "learned-phone.yaml"
        _write_sentences(
            path,
            "PhoneLastSeenAgo",
            ["where is my phone at"],
            lists={"who": {"values": ["Sam", "Alex"]}},
        )
        write_response(
            path,
            "PhoneLastSeenAgo",
            "{% if who is defined %}{{ who }}'s phone{% else %}{{ speaker }}, your phone{% endif %}",
        )
        stored = load(path)["responses"]["intents"]["PhoneLastSeenAgo"]["default"]
        render = lambda variables: template.Template(stored, self.hass).async_render(  # noqa: E731
            variables, parse_result=False
        )
        # Home Assistant's own renderer, which has `slots` and nothing else.
        self.assertEqual(render({"slots": {}, "speaker": "Sam"}), "Sam, your phone")
        self.assertEqual(render({"slots": {"who": "Alex"}, "speaker": "Sam"}), "Alex's phone")

    def test_a_binding_written_before_this_is_upgraded(self):
        path = pathlib.Path(self.sentences_dir()) / "learned-weather.yaml"
        _write_sentences(
            path, "Weather", ["what's the weather {day}"], lists={"day": {"values": ["today"]}}
        )
        # The form written before the bug above was found.
        path.write_text(
            "language: en\nintents: {Weather: {data: [{sentences: [\"what's the weather {day}\"]}]}}\n"
            "lists: {day: {values: [today]}}\n"
            "responses: {intents: {Weather: {default: \"{% set day = slots.day or '' %}On {{ day }} it rains.\"}}}\n",
            encoding="utf-8",
        )
        # It is read back as the reply that was written, with no binding in it...
        self.assertEqual(file_responses(self.hass, "en")["Weather"], "On {{ day }} it rains.")
        # ... and writing it again replaces it with the form that keeps `is defined` honest.
        write_response(path, "Weather", file_responses(self.hass, "en")["Weather"])
        stored = load(path)["responses"]["intents"]["Weather"]["default"]
        self.assertNotIn("or ''", stored)
        self.assertIn("is defined else day", stored)

    def test_a_slot_no_sentence_supplies_is_not_bound(self):
        path = pathlib.Path(self.sentences_dir()) / "learned-thing.yaml"
        _write_sentences(path, "Thing", ["what is the thing"])
        write_response(path, "Thing", "The room is {{ room }}.")
        stored = load(path)["responses"]["intents"]["Thing"]["default"]
        self.assertNotIn("slots.room", stored, "nothing declares `room`, so there is no value to bind")

    def test_the_binding_is_written_once_and_reads_back_without_it(self):
        path = pathlib.Path(self.sentences_dir()) / "learned-weather.yaml"
        _write_sentences(
            path, "Weather", ["what's the weather {day}"], lists={"day": {"values": ["tomorrow"]}}
        )
        write_response(path, "Weather", "On {{ day }} it rains.")
        first = load(path)["responses"]["intents"]["Weather"]["default"]
        # Writing the reply the reader handed back must not stack a second binding.
        write_response(path, "Weather", file_responses(self.hass, "en")["Weather"])
        self.assertEqual(load(path)["responses"]["intents"]["Weather"]["default"], first)
        self.assertEqual(first.count("{% set day"), 1)
        # And what the reader hands back is the readable text, not the machinery.
        self.assertEqual(file_responses(self.hass, "en")["Weather"], "On {{ day }} it rains.")

    def test_a_slot_list_is_written_with_the_sentences_and_widened_later(self):
        path = pathlib.Path(self.sentences_dir()) / "learned-thing.yaml"
        _write_sentences(
            path, "Thing", ["what's the weather on {day}"], lists={"day": {"values": ["monday"]}}
        )
        _write_lists(path, {"day": {"values": ["monday", "tuesday"]}})
        data = load(path)
        self.assertEqual(data["lists"]["day"]["values"], ["monday", "tuesday"])
        self.assertEqual(
            data["intents"]["Thing"]["data"][0]["sentences"], ["what's the weather on {day}"]
        )

    def test_forgetting_removes_the_sentences_and_deletes_a_file_it_emptied(self):
        directory = pathlib.Path(self.sentences_dir())
        only = directory / "learned-only.yaml"
        shared = directory / "learned-shared.yaml"
        _write_sentences(only, "Only", ["one thing"])
        _write_sentences(shared, "Only", ["another thing"])
        _write_sentences(shared, "Other", ["a different thing"])

        touched = remove_intent_sentences(self.hass, "en", "Only")

        self.assertIn("learned-only.yaml", touched)
        self.assertFalse(only.exists(), "a file the intent alone lived in should be deleted")
        # The other intent's file survives with only its own sentence left in it.
        remaining = load(shared)
        self.assertNotIn("Only", remaining["intents"])
        self.assertEqual(remaining["intents"]["Other"]["data"][0]["sentences"], ["a different thing"])


class Consolidation(HassCase):
    """One file per intent, with the slot lists that keep its sentences loadable."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        directory = pathlib.Path(self.sentences_dir())
        _write_sentences(
            directory / "learned-scattered.yaml",
            "WeatherForecast",
            ["what's the weather on {day}"],
            lists={"day": {"values": ["monday", "tuesday"]}},
        )
        _write_sentences(directory / "learned-elsewhere.yaml", "WeatherForecast", ["and tomorrow?"])
        write_response(directory / "learned-elsewhere.yaml", "WeatherForecast", "It will rain.")
        self.entry.runtime_data.intents = [
            LearnedIntent(
                name="WeatherForecast",
                template="{{ states('weather.home') }}",
                file="custom_sentences/en/learned-elsewhere.yaml",
            )
        ]

    async def test_one_file_per_intent_and_the_lists_come_with_it(self):
        await async_consolidate_intents(self.hass, self.entry)

        directory = pathlib.Path(self.sentences_dir())
        files = sorted(p.name for p in directory.glob("learned-*.yaml"))
        self.assertEqual(files, ["learned-weatherforecast.yaml"])
        data = load(directory / "learned-weatherforecast.yaml")
        sentences = [s for item in data["intents"]["WeatherForecast"]["data"] for s in item["sentences"]]
        # `?` is stripped on write: the stored form is what hassil can actually match.
        self.assertCountEqual(sentences, ["what's the weather on {day}", "and tomorrow"])
        # Without this the sentences above stop matching for every agent.
        self.assertEqual(data["lists"]["day"]["values"], ["monday", "tuesday"])
        self.assertEqual(data["responses"]["intents"]["WeatherForecast"]["default"], "It will rain.")
        self.assertEqual(
            self.entry.runtime_data.intents[0].file,
            "custom_sentences/en/learned-weatherforecast.yaml",
        )


class Rename(HassCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        directory = pathlib.Path(self.sentences_dir())
        path = directory / "learned-weatherweekday.yaml"
        _write_sentences(
            path,
            "WeatherWeekday",
            ["what's the weather on {day}"],
            lists={"day": {"values": ["monday", "tuesday"]}},
        )
        write_response(path, "WeatherWeekday", "On {{ day }} it will rain.")
        self.entry.runtime_data.intents = [
            LearnedIntent(
                name="WeatherWeekday",
                template="On {{ day }} it will rain.",
                slots={"day": {"values": ["monday", "tuesday"]}},
                file="custom_sentences/en/learned-weatherweekday.yaml",
            )
        ]

    async def test_everything_moves_and_the_old_name_is_gone(self):
        result = await async_rename_intent(self.hass, self.entry, "WeatherWeekday", "WeatherForecast")

        directory = pathlib.Path(self.sentences_dir())
        self.assertFalse((directory / "learned-weatherweekday.yaml").exists())
        moved = load(self.hass.config.path(result["file"]))
        self.assertIn("WeatherForecast", moved["intents"])
        self.assertIn("WeatherForecast", moved["responses"]["intents"])
        self.assertEqual(moved["lists"]["day"]["values"], ["monday", "tuesday"])
        self.assertEqual(self.entry.runtime_data.intents[0].name, "WeatherForecast")

    async def test_a_name_that_is_taken_is_refused(self):
        self.entry.runtime_data.intents.append(
            LearnedIntent(name="Other", template="x", file="custom_sentences/en/learned-other.yaml")
        )
        with self.assertRaises(LearnerError):
            await async_rename_intent(self.hass, self.entry, "WeatherWeekday", "Other")


class Merge(HassCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        directory = pathlib.Path(self.sentences_dir())
        _write_sentences(directory / "learned-weatherforecast.yaml", "WeatherForecast", ["what's the weather on {day}"])
        _write_sentences(directory / "learned-weathertomorrow.yaml", "WeatherTomorrow", ["what's the weather tomorrow"])
        self.entry.runtime_data.intents = [
            LearnedIntent(
                name="WeatherForecast",
                template="old",
                file="custom_sentences/en/learned-weatherforecast.yaml",
            ),
            LearnedIntent(
                name="WeatherTomorrow",
                template="tomorrow only",
                file="custom_sentences/en/learned-weathertomorrow.yaml",
            ),
        ]

    async def test_the_dry_run_changes_nothing(self):
        await async_merge_intents(
            self.hass, self.entry, into="WeatherForecast", others=["WeatherTomorrow"], apply=False
        )
        directory = pathlib.Path(self.sentences_dir())
        self.assertTrue((directory / "learned-weathertomorrow.yaml").exists())
        self.assertEqual(len(self.entry.runtime_data.intents), 2)

    async def test_applying_moves_the_wording_and_forgets_the_rest(self):
        await async_merge_intents(
            self.hass,
            self.entry,
            into="WeatherForecast",
            others=["WeatherTomorrow"],
            slots={"day": {"values": ["today", "tomorrow"]}},
            template="{{ day }}",
            reply="{{ day }}",
            apply=True,
        )
        directory = pathlib.Path(self.sentences_dir())
        self.assertFalse((directory / "learned-weathertomorrow.yaml").exists())
        self.assertEqual([s.name for s in self.entry.runtime_data.intents], ["WeatherForecast"])
        # "tomorrow" is a declared value, so the incoming wording folds onto the slot rather
        # than becoming a second sentence with a fixed day in it.
        data = load(directory / "learned-weatherforecast.yaml")
        sentences = [s for item in data["intents"]["WeatherForecast"]["data"] for s in item["sentences"]]
        self.assertIn("what's the weather {day}", sentences)


class LearnIntent(HassCase):
    """The tool a model calls when no existing intent fits."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.hass.states.async_set("weather.home", "rainy", {"temperature": 14.0})

    async def test_the_file_is_named_after_the_intent_not_the_note(self):
        result = await async_learn_intent(
            self.hass,
            self.entry,
            name="WeatherForecast",
            template="It is {{ states('weather.home') }}.",
            utterances=["what's the weather"],
            note="a note that would make a terrible filename",
        )
        self.assertEqual(result["file"], "custom_sentences/en/learned-weatherforecast.yaml")
        self.assertTrue(pathlib.Path(self.hass.config.path(result["file"])).exists())

    async def test_a_slot_folds_the_example_phrasings_onto_one_sentence(self):
        result = await async_learn_intent(
            self.hass,
            self.entry,
            name="WeatherForecast",
            template="On {{ day }} it will be {{ states('weather.home') }}.",
            utterances=["what's the weather on Monday", "what's the weather on Tuesday"],
            slots={"day": ["monday", "tuesday"]},
        )
        self.assertEqual(result["added"], ["what's the weather on {day}"])
        self.assertEqual(result["slots"]["day"]["values"], ["monday", "tuesday"])
        # And the answer the model reads back names the ids it actually reads.
        self.assertEqual(result["reads"], {"weather.home": "rainy"})

    async def test_a_slot_the_sentences_do_not_declare_is_refused(self):
        with self.assertRaises(LearnerError) as caught:
            await async_learn_intent(
                self.hass,
                self.entry,
                name="WeatherForecast",
                template="On {{ day }} it will rain.",
                utterances=["what's the weather on {day}"],
            )
        self.assertIn("day", str(caught.exception))

    async def test_a_duplicate_of_an_installed_intent_is_refused(self):
        # No judge is configured in the test entry, so the check is stubbed to its answer.
        import custom_components.irabelle_home_learning.intents as intents

        original = intents.async_duplicate_intent

        async def duplicate(hass, entry, utterances, template, *, exclude=None):
            return "WeatherForecast"

        intents.async_duplicate_intent = duplicate
        try:
            with self.assertRaises(LearnerError) as caught:
                await async_learn_intent(
                    self.hass,
                    self.entry,
                    name="WeatherTomorrow",
                    template="It is {{ states('weather.home') }}.",
                    utterances=["what's the weather tomorrow"],
                )
            # The refusal has to name what to bind to, or a model has nothing to do next.
            self.assertIn("WeatherForecast", str(caught.exception))
        finally:
            intents.async_duplicate_intent = original


if __name__ == "__main__":
    unittest.main(verbosity=2)


class ReplyTidying(fixtures.WorldCase):
    """A typo the component can fix itself, rather than asking the model again."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.hass.states.async_set("weather.home", "rainy")
        self.entry.runtime_data.intents.append(
            LearnedIntent(
                name="WeatherForecast",
                template="It is {{ states('weather.home') }}.",
                file="custom_sentences/en/learned-weatherforecast.yaml",
            )
        )

    async def test_a_glued_sentence_is_cleaned_up_when_the_reply_is_written(self):
        result = await async_set_response(
            self.hass,
            self.entry,
            "WeatherForecast",
            "It is {{ states('weather.home') }} in the afternoon., the high will be 19.",
        )
        self.assertNotIn(".,", result["response"])
        self.assertEqual(result["says"], "It is rainy in the afternoon, the high will be 19.")
        # And it is not silent about having done it.
        self.assertTrue(any("punctuation" in note for note in result["tidied"]))


class TheReplyItHeard(fixtures.WorldCase):
    """A phrase learned for a built-in intent keeps the reply the person actually heard.

    Measured on the live house: the learner bound "set the office light brightness to warm"
    to HassLightSet with the temperature the answer chose, and then the phrase matched,
    changed the light, and said nothing at all - the built-in intent's reply exists only for
    the sentence groups Home Assistant shipped, and ours had none.

    And the reply belongs to the *group*: teaching "cold" (4000 K) after "warm" (2700 K) made
    cold answer "Office light set to warm white", because one `default` served both.
    """

    def register(self, intent: str) -> None:
        class Stub:
            def __init__(self) -> None:
                self.intent_type = intent
                self.description = "Sets the brightness or color of a light"

            async def async_handle(self, intent_obj):  # pragma: no cover - never called here
                return intent_obj.create_response()

        intent_helper.async_register(self.hass, Stub())

    def written(self) -> dict:
        return yaml_util.parse_yaml(
            pathlib.Path(
                self.hass.config.path("custom_sentences/en/learned-hasslightset.yaml")
            ).read_text()
        )

    async def learn(self, phrase: str, slots: dict, reply: str) -> None:
        await learner.async_learn(
            self.hass,
            self.entry,
            utterances=[phrase],
            intent="HassLightSet",
            slots=slots,
            response=reply,
        )

    async def test_the_heard_reply_is_written_with_the_phrasing(self):
        self.register("HassLightSet")
        await self.learn(
            "set the office light brightness to warm",
            {"temperature": 2700, "name": "Office Light"},
            "Office light set to warm white.",
        )
        written = self.written()
        group = written["intents"]["HassLightSet"]["data"][0]
        self.assertEqual(group["slots"], {"temperature": 2700, "name": "Office Light"})
        self.assertTrue(group["response"], "the group names its own reply")
        self.assertEqual(
            written["responses"]["intents"]["HassLightSet"][group["response"]],
            "Office light set to warm white.",
        )

    async def test_each_value_keeps_its_own_reply(self):
        # The reported bug: cold inherited warm's words.
        self.register("HassLightSet")
        await self.learn(
            "set the office light temperature to warm",
            {"temperature": 2700, "name": "Office Light"},
            "Office light set to warm white.",
        )
        await self.learn(
            "set the office light temperature to cold",
            {"temperature": 4000, "name": "Office Light"},
            "Office light set to cold white.",
        )
        written = self.written()
        groups = {
            tuple(sorted(item["slots"].items())): item["response"]
            for item in written["intents"]["HassLightSet"]["data"]
        }
        replies = written["responses"]["intents"]["HassLightSet"]
        self.assertEqual(replies[groups[(("name", "Office Light"), ("temperature", 2700))]],
                         "Office light set to warm white.")
        self.assertEqual(replies[groups[(("name", "Office Light"), ("temperature", 4000))]],
                         "Office light set to cold white.")
