"""What survives the integration being removed.

The expectation this file pins down: a learned intent is a *file*, and the store is only
provenance. So the file carries the sentences, the slot lists, the reply and the template
behind it; deleting the integration's storage, or the integration itself, loses none of it.

Two failures are being prevented, and only one of them is about losing text:

* an unloaded integration leaves sentences whose intents nothing answers, and recognition
  raises for those phrases - the person gets an error instead of an answer;
* a store that is gone (a fresh install, or an integration removed and added again) leaves
  the files behind with nothing to load them.
"""

from __future__ import annotations

import pathlib
import unittest

import fixtures
import yaml

from homeassistant.helpers import intent as intent_helper

from custom_components.irabelle_home_learning import async_unload_entry
from custom_components.irabelle_home_learning.intents import (
    LearnedIntentHandler,
    async_duplicate_verdicts,
    duplicate_pairs,
    _write_sentences,
    restore_intents,
    async_learn_intent,
    export_intent_script,
    refresh_export,
    async_set_response,
    learner_sentences,
    register_intent,
    write_response,
)
from custom_components.irabelle_home_learning.learner import (
    DISARMED_SUFFIX,
    LearnerError,
    LearnedIntent,
    adopt_disarmed,
    arm_learner_files,
    async_load,
    async_save,
    disarm_learner_files,
    disarmed_files,
    file_responses,
    file_templates,
)


def load(path) -> dict:
    return yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8")) or {}


class TheFileIsTheIntent(fixtures.WorldCase):
    """Reply and template both live in `custom_sentences/`, in one readable file."""

    async def test_a_reply_asked_for_is_written_beside_the_computation(self):
        # A person asks for different words; the computation that builds them is a different
        # text, and both belong in the file or the folder is not the whole intent.
        path = pathlib.Path(self.sentences_dir()) / "learned-weatherforecast.yaml"
        write_response(path, "WeatherForecast", "It is nice today.", "It is {{ states('weather.home') }}.")
        data = load(path)["responses"]["intents"]["WeatherForecast"]
        self.assertEqual(data["default"], "It is nice today.")
        self.assertEqual(data["template"], "It is {{ states('weather.home') }}.")

    async def test_a_reply_that_is_the_template_is_not_duplicated(self):
        path = pathlib.Path(self.sentences_dir()) / "learned-weatherforecast.yaml"
        write_response(path, "WeatherForecast", "It is {{ states('weather.home') }}.", "It is {{ states('weather.home') }}.")
        data = load(path)["responses"]["intents"]["WeatherForecast"]
        self.assertEqual(data, {"default": "It is {{ states('weather.home') }}."})

    async def test_the_file_wins_over_the_store_on_the_next_load(self):
        fixtures.add_entity(self.hass, "weather.home", "rainy")
        await async_learn_intent(
            self.hass,
            self.entry,
            name="WeatherForecast",
            template="It is {{ states('weather.home') }}.",
            utterances=["what's the weather"],
        )
        await async_save(self.hass, self.entry)
        # Edit the text in the file, as a person would, and reload: the file is authoritative.
        path = pathlib.Path(self.sentences_dir()) / "learned-weatherforecast.yaml"
        write_response(path, "WeatherForecast", "I say {{ states('weather.home') }}.", "ignored template")
        reloaded = await async_load(self.hass, self.entry)
        self.assertEqual(reloaded.intents[0].response, "I say {{ states('weather.home') }}.")
        self.assertEqual(reloaded.intents[0].template, "ignored template")

    async def test_a_store_with_nothing_in_it_still_finds_the_text(self):
        fixtures.add_entity(self.hass, "weather.home", "rainy")
        path = pathlib.Path(self.sentences_dir()) / "learned-weatherforecast.yaml"
        _write_sentences(path, "WeatherForecast", ["what's the weather"])
        write_response(path, "WeatherForecast", "It is {{ states('weather.home') }}.")
        # No store rows at all: what the folder says is all there is.
        self.assertEqual(file_responses(self.hass, "en"), {"WeatherForecast": "It is {{ states('weather.home') }}."})
        self.assertEqual(file_templates(self.hass, "en"), {})


class SteppingAside(fixtures.WorldCase):
    """Unload takes the sentences out of the matcher's way; setup puts them back."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        fixtures.add_entity(self.hass, "weather.home", "rainy")
        self.result = await async_learn_intent(
            self.hass,
            self.entry,
            name="WeatherForecast",
            template="It is {{ states('weather.home') }}.",
            utterances=["what's the weather"],
            slots={"day": ["today", "tomorrow"]},
        )
        await async_set_response(
            self.hass, self.entry, "WeatherForecast", "I say {{ states('weather.home') }}."
        )
        await async_save(self.hass, self.entry)

    async def test_unload_renames_the_files_and_keeps_every_word(self):
        armed = pathlib.Path(self.hass.config.path(self.result["file"]))
        self.assertTrue(armed.exists())

        renamed = disarm_learner_files(self.hass, "en")

        self.assertEqual(renamed, [armed.name + DISARMED_SUFFIX])
        self.assertFalse(armed.exists())
        stepped_aside = pathlib.Path(str(armed) + DISARMED_SUFFIX)
        self.assertTrue(stepped_aside.exists())
        # Untouched: still the sentences, the lists, the reply and the computation.
        data = load(stepped_aside)
        self.assertEqual(data["intents"]["WeatherForecast"]["data"][0]["sentences"], ["what's the weather"])
        self.assertEqual(data["lists"]["day"]["values"], ["today", "tomorrow"])
        self.assertEqual(
            data["responses"]["intents"]["WeatherForecast"]["default"],
            "I say {{ states('weather.home') }}.",
        )
        self.assertEqual(
            data["responses"]["intents"]["WeatherForecast"]["template"],
            "It is {{ states('weather.home') }}.",
        )

    async def test_setting_up_again_puts_them_back(self):
        disarm_learner_files(self.hass, "en")
        armed = arm_learner_files(self.hass, "en", {"WeatherForecast"})
        self.assertEqual(armed, [pathlib.Path(self.result["file"]).name])
        self.assertEqual(disarmed_files(self.hass, "en"), [])
        self.assertEqual(learner_sentences(self.hass, "en", {"WeatherForecast"})["WeatherForecast"], ["what's the weather"])


class UnloadTellsTheMatcher(fixtures.WorldCase):
    """Renaming a file is only half of stepping aside: the matcher caches what it last read.

    Measured on the live house before this was fixed: disabling the integration renamed the
    files, released the handlers, and every learned phrase still answered `intent-failed` -
    the sentence was matched from memory and nothing was left to answer it. The phrases only
    fell through to the agent after a restart, which is not something a person removing an
    integration knows to do.
    """

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.reloads = 0
        # Replace the harness's no-op with a counter, so "did it tell the matcher" is visible.
        self.hass.services.async_remove("conversation", "reload")
        self.hass.services.async_register(
            "conversation", "reload", lambda call: setattr(self, "reloads", self.reloads + 1)
        )
        fixtures.add_entity(self.hass, "weather.home", "rainy")
        self.result = await async_learn_intent(
            self.hass,
            self.entry,
            name="WeatherForecast",
            template="It is {{ states('weather.home') }}.",
            utterances=["what's the weather"],
        )

    async def test_unloading_reloads_the_sentences_it_stepped_aside(self):
        self.reloads = 0
        await async_unload_entry(self.hass, self.entry)
        self.assertEqual(self.reloads, 1, "the matcher has to be told the files moved")
        self.assertEqual(
            [path.name for path in disarmed_files(self.hass, "en")],
            [pathlib.Path(self.result["file"]).name + DISARMED_SUFFIX],
        )

    async def test_nothing_stepped_aside_is_not_worth_a_reload(self):
        disarm_learner_files(self.hass, "en")
        self.reloads = 0
        await async_unload_entry(self.hass, self.entry)
        self.assertEqual(self.reloads, 0)


class ReclaimingTheName(fixtures.WorldCase):
    """An exported registration names our intents too, so ours has to be the one holding them.

    `intent_script` sets up *after* this integration, so its stub - an empty action that says
    nothing - ends up in the one slot Home Assistant keeps per intent name. The reply is then
    rendered by Home Assistant instead of by this component, which loses everything the handler
    puts in scope: measured on a restart, "who am I" answered "I don't know who is asking"
    instead of naming the person. Hence the re-registration at the end of startup.
    """

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.spec = LearnedIntent(
            name="WeatherForecast", template="It is rainy.", file="custom_sentences/en/x.yaml"
        )
        self.entry.runtime_data.intents.append(self.spec)

    def test_our_handler_takes_the_name_back_from_a_registered_stub(self):
        class Stub(intent_helper.IntentHandler):
            """What `intent_script` registers for an `action: []` entry."""

            def __init__(self) -> None:
                self.intent_type = "WeatherForecast"

            async def async_handle(self, intent_obj):  # pragma: no cover - never called here
                return intent_obj.create_response()

        intent_helper.async_register(self.hass, Stub())
        self.assertIsInstance(self.holder(), Stub, "the stub is registered after us, as it is at startup")

        restore_intents(self.hass, self.entry.runtime_data)

        self.assertIsInstance(self.holder(), LearnedIntentHandler)

    def holder(self):
        return next(
            handler
            for handler in intent_helper.async_get(self.hass)
            if handler.intent_type == "WeatherForecast"
        )


class Adopting(fixtures.WorldCase):
    """A store that is gone: the files hand the intents back."""

    async def write_aside(self, name: str, *, reply: str | None) -> pathlib.Path:
        body = [
            "language: en",
            "intents:",
            f"  {name}:",
            "    data:",
            "    - sentences:",
            "      - what's the weather on {day}",
            "lists:",
            "  day:",
            "    values: [today, tomorrow]",
        ]
        if reply:
            body += ["responses:", "  intents:", f"    {name}:", f"      default: {reply!r}"]
        path = pathlib.Path(self.sentences_dir()) / f"learned-{name.lower()}.yaml{DISARMED_SUFFIX}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(body) + "\n", encoding="utf-8")
        return path

    async def test_an_intent_is_rebuilt_from_its_own_file(self):
        await self.write_aside("WeatherForecast", reply="It is {{ states('weather.home') }}.")
        adopted = adopt_disarmed(self.hass, "en")
        self.assertEqual([spec.name for spec in adopted], ["WeatherForecast"])
        spec = adopted[0]
        self.assertEqual(spec.response, "It is {{ states('weather.home') }}.")
        # With no separate template key, the reply is the computation.
        self.assertEqual(spec.template, spec.response)
        self.assertEqual(spec.slots["day"]["values"], ["today", "tomorrow"])
        # And it points at the name the file will have once it is armed.
        self.assertFalse(spec.file.endswith(DISARMED_SUFFIX))

    async def test_a_file_with_no_reply_is_left_aside_rather_than_armed_into_a_failure(self):
        await self.write_aside("NoReply", reply=None)
        self.assertEqual(adopt_disarmed(self.hass, "en"), [])
        # Nothing answers it, so arming it would put a sentence in front of the matcher with
        # no handler - the error this mechanism exists to avoid.
        self.assertEqual(arm_learner_files(self.hass, "en", set()), [])
        self.assertEqual(len(disarmed_files(self.hass, "en")), 1)

    async def test_an_adopted_intent_is_registered_and_answered(self):
        from homeassistant.helpers import intent as intent_helper

        fixtures.add_entity(self.hass, "weather.home", "rainy")
        await self.write_aside("WeatherForecast", reply="It is {{ states('weather.home') }}.")
        for spec in adopt_disarmed(self.hass, "en"):
            self.entry.runtime_data.intents.append(spec)
            register_intent(self.hass, spec)
        arm_learner_files(self.hass, "en", {"WeatherForecast"})

        response = await intent_helper.async_handle(
            self.hass, "conversation", "WeatherForecast", {"day": {"value": "monday"}}, text_input="anything"
        )
        self.assertEqual(response.speech["plain"]["speech"], "It is rainy.")


if __name__ == "__main__":
    unittest.main(verbosity=2)


class JudgingDuplicates(fixtures.WorldCase):
    """Two intents that read the same sensor are evidence; whether they are one question is a
    judgement, and it is the part a merge cannot make for itself."""

    def report(self):
        members = [
            {"intent": "PhoneLastSeen", "says": "Your phone was last seen in the Bedroom."},
            {"intent": "PhoneLastSeenAgo", "says": "Your phone was last seen in the Bedroom, 2 minutes ago."},
            {"intent": "PhoneLastSeenTime", "says": "Your phone was last seen at 5:42 PM."},
        ]
        return {"entities": [{"entity_id": "sensor.phone_area", "intents": members}]}

    def test_the_pairs_come_from_the_shared_entity_groups(self):
        pairs = duplicate_pairs(self.report())
        self.assertEqual(
            sorted(tuple(sorted((a["intent"], b["intent"]))) for a, b in pairs),
            [
                ("PhoneLastSeen", "PhoneLastSeenAgo"),
                ("PhoneLastSeen", "PhoneLastSeenTime"),
                ("PhoneLastSeenAgo", "PhoneLastSeenTime"),
            ],
        )

    def test_a_pair_is_only_judged_once(self):
        # The same two intents usually share several entities; asking twice is paying twice.
        report = self.report()
        report["entities"].append(report["entities"][0])
        self.assertEqual(len(duplicate_pairs(report)), 3)

    def test_nothing_shared_is_nothing_to_judge(self):
        self.assertEqual(duplicate_pairs({"entities": []}), [])

    async def test_a_confident_verdict_says_they_are_one_question(self):
        import custom_components.irabelle_home_learning.intents as intents_module

        async def fake_ask(hass, entry, state, questions, **kwargs):
            class Answer:
                noul = 0.9

            return {key: Answer() for key in questions}

        original = intents_module.async_ask
        intents_module.async_ask = fake_ask
        try:
            verdicts = await async_duplicate_verdicts(self.hass, self.entry, self.report())
        finally:
            intents_module.async_ask = original
        self.assertEqual(len(verdicts), 3)
        self.assertTrue(all(row["same"] for row in verdicts))
        self.assertEqual(verdicts[0]["score"], 0.9)

    async def test_an_unsure_verdict_does_not_call_them_the_same(self):
        import custom_components.irabelle_home_learning.intents as intents_module

        async def fake_ask(hass, entry, state, questions, **kwargs):
            class Answer:
                noul = 0.4

            return {key: Answer() for key in questions}

        original = intents_module.async_ask
        intents_module.async_ask = fake_ask
        try:
            verdicts = await async_duplicate_verdicts(self.hass, self.entry, self.report())
        finally:
            intents_module.async_ask = original
        self.assertFalse(any(row["same"] for row in verdicts))


class Exporting(fixtures.WorldCase):
    """Where the registration goes: the house declares it, and this follows.

    A learned intent is a name that needs a handler, and `intent_script` is the only way to
    register one without this component. Which file that lands in is the house's choice, so
    every shape Home Assistant supports is pinned here.
    """

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.entry.runtime_data.intents.append(
            LearnedIntent(name="WeatherForecast", template="x", file="custom_sentences/en/x.yaml")
        )

    def declare(self, text: str) -> pathlib.Path:
        """Write a `configuration.yaml` the way a house would have one."""
        path = pathlib.Path(self.hass.config.path("configuration.yaml"))
        path.write_text(text, encoding="utf-8")
        return path

    def file(self, relative: str) -> pathlib.Path:
        return pathlib.Path(self.hass.config.path(relative))

    async def test_nothing_declared_creates_a_root_file_and_wires_it_up(self):
        result = export_intent_script(self.hass, self.entry)
        self.assertEqual(result["where"], "none")
        self.assertEqual(result["include"], "intent_script: !include configurations/intent_script/irabelle_home_learning.yaml")
        self.assertEqual(result["file"], str(self.file("configurations/intent_script/irabelle_home_learning.yaml")))
        self.assertEqual(load(result["file"]), {"WeatherForecast": {"action": []}})
        # And it is actually used: the include is in configuration.yaml, once.
        config = self.file("configuration.yaml").read_text()
        self.assertEqual(config.count("intent_script: !include configurations/intent_script/irabelle_home_learning.yaml"), 1)
        export_intent_script(self.hass, self.entry)
        self.assertEqual(
            self.file("configuration.yaml").read_text().count("intent_script: !include"), 1
        )

    async def test_an_include_file_gets_our_entries_beside_its_own(self):
        self.declare("intent_script: !include intent_script.yaml\n")
        shared = self.file("intent_script.yaml")
        shared.write_text("PreviousIntent:\n  action: []  # mine, do not touch\n", encoding="utf-8")

        result = export_intent_script(self.hass, self.entry)

        self.assertEqual(result["where"], "file")
        text = shared.read_text()
        self.assertIn("# mine, do not touch", text, "a hand-written entry keeps its comment")
        data = yaml.safe_load(text)
        self.assertEqual(data["PreviousIntent"], {"action": []})
        self.assertEqual(data["WeatherForecast"], {"action": []})

    async def test_a_merge_dir_gets_its_own_file_and_leaves_the_rest(self):
        self.declare("intent_script: !include_dir_merge_named intent_script/\n")
        theirs = self.file("intent_script/other.yaml")
        theirs.parent.mkdir(parents=True, exist_ok=True)
        theirs.write_text("TheirIntent:\n  action: []\n", encoding="utf-8")

        result = export_intent_script(self.hass, self.entry)

        self.assertEqual(result["where"], "merge_dir")
        self.assertEqual(pathlib.Path(result["file"]).name, "irabelle_home_learning.yaml")
        self.assertEqual(load(result["file"]), {"WeatherForecast": {"action": []}})
        self.assertEqual(load(theirs), {"TheirIntent": {"action": []}}, "their file is untouched")

    async def test_a_named_dir_gets_one_file_per_intent(self):
        self.declare("intent_script: !include_dir_named intents/\n")
        result = export_intent_script(self.hass, self.entry)
        self.assertEqual(result["where"], "named_dir")
        # `!include_dir_named` names the intent after the file, so one file each.
        self.assertEqual(load(self.file("intents/WeatherForecast.yaml")), {"action": []})
        # Forgetting it takes its file with it, and only that one.
        keep = self.file("intents/Theirs.yaml")
        keep.write_text("action: []\n", encoding="utf-8")
        self.entry.runtime_data.intents.clear()
        refresh_export(self.hass, self.entry)
        self.assertFalse(self.file("intents/WeatherForecast.yaml").exists())
        self.assertTrue(keep.exists(), "a file without our mark is not ours to delete")

    async def test_an_inline_mapping_is_added_to_in_place(self):
        config = self.declare(
            "homeassistant:\n  name: Home\n"
            "intent_script:\n  PreviousIntent:\n    action: []\n"
            "logger:\n  default: critical\n"
        )
        result = export_intent_script(self.hass, self.entry)
        self.assertEqual(result["where"], "inline")
        data = yaml.safe_load(config.read_text())
        self.assertEqual(data["intent_script"]["PreviousIntent"], {"action": []})
        self.assertEqual(data["intent_script"]["WeatherForecast"], {"action": []})
        self.assertEqual(data["logger"], {"default": "critical"}, "the next key is not swallowed")
        # And again: the entry is updated, never duplicated.
        export_intent_script(self.hass, self.entry)
        self.assertEqual(config.read_text().count("WeatherForecast:"), 1)

    async def test_an_entry_we_wrote_is_removed_when_its_intent_is_forgotten(self):
        self.declare("intent_script: !include intent_script.yaml\n")
        shared = self.file("intent_script.yaml")
        shared.write_text("PreviousIntent:\n  action: []\n", encoding="utf-8")
        export_intent_script(self.hass, self.entry)
        self.entry.runtime_data.intents.clear()

        result = export_intent_script(self.hass, self.entry)

        self.assertEqual(result["removed"], ["WeatherForecast"])
        data = yaml.safe_load(shared.read_text())
        self.assertEqual(data, {"PreviousIntent": {"action": []}})

    async def test_each_entry_says_where_its_phrasings_and_reply_live(self):
        """The registration is also the index: the words and the reply are in the other file."""
        result = export_intent_script(self.hass, self.entry)
        written = pathlib.Path(result["file"]).read_text()
        self.assertIn("# custom_sentences/en/x.yaml", written)
        self.assertIn("WeatherForecast:", written)

    async def test_a_refresh_keeps_the_file_naming_what_exists(self):
        self.declare("intent_script: !include_dir_merge_named intent_script/\n")
        export_intent_script(self.hass, self.entry)
        self.assertFalse(refresh_export(self.hass, self.entry), "nothing changed, no write")
        self.entry.runtime_data.intents.append(
            LearnedIntent(name="WatchLastSeen", template="x", file="custom_sentences/en/watch.yaml")
        )
        self.assertTrue(refresh_export(self.hass, self.entry))
        written = self.file("intent_script/irabelle_home_learning.yaml").read_text()
        self.assertIn("WatchLastSeen:", written)
        self.assertIn("# custom_sentences/en/watch.yaml", written)

    async def test_a_house_that_never_exported_gets_no_registration(self):
        """Opt-in: with nothing declared, a learning must not edit `configuration.yaml`.

        The stubs are insurance, and insurance is asked for. A house that declares
        `intent_script` itself has already said where its registrations go, so that one is kept
        current; one that says nothing keeps one file per intent and gets no writes.
        """
        self.assertFalse(refresh_export(self.hass, self.entry))
        self.assertFalse(self.file("intent_script.yaml").exists())
        self.assertFalse(self.file("configuration.yaml").exists())

    async def test_a_declared_registration_is_kept_current_without_an_export(self):
        self.declare("intent_script: !include intent_script.yaml\n")
        self.entry.runtime_data.intents.append(
            LearnedIntent(name="WatchLastSeen", template="x", file="custom_sentences/en/w.yaml")
        )
        self.assertTrue(refresh_export(self.hass, self.entry))
        self.assertEqual(
            yaml.safe_load(self.file("intent_script.yaml").read_text()),
            {"WatchLastSeen": {"action": []}, "WeatherForecast": {"action": []}},
        )

    async def test_a_path_outside_the_config_folder_is_refused(self):
        # The option is user input: it must not be able to write outside `/config`, and empty
        # means the default.
        for bad in ("/etc/passwd.yaml", "../../outside.yaml", ""):
            with self.subTest(bad=bad):
                self.entry.options["export_path"] = bad
                result = export_intent_script(self.hass, self.entry)
                self.assertEqual(result["file"], str(self.file("configurations/intent_script/irabelle_home_learning.yaml")))
                self.assertTrue(self.file("configurations/intent_script/irabelle_home_learning.yaml").exists())

    async def test_an_export_with_nothing_to_register_is_still_a_document(self):
        # After `forget_all` there is nothing to register. A file of nothing but comments
        # loads as None, and the directory include keeps only what is a dict, so it says `{}`.
        self.entry.runtime_data.intents.clear()
        result = export_intent_script(self.hass, self.entry)
        self.assertEqual(result["count"], 0)
        self.assertEqual(yaml.safe_load(self.file("configurations/intent_script/irabelle_home_learning.yaml").read_text()), {})

    async def test_a_name_that_is_not_a_plain_word_cannot_break_the_file(self):
        # This file is included from configuration.yaml, where a syntax error stops Home
        # Assistant from starting. A name adopted from a hand-edited file is not validated.
        self.entry.runtime_data.intents.append(
            LearnedIntent(name="Odd: Name", template="x", file="custom_sentences/en/y.yaml")
        )
        result = export_intent_script(self.hass, self.entry)
        self.assertEqual(
            yaml.safe_load(
                "\n".join(
                    line
                    for line in self.file("configurations/intent_script/irabelle_home_learning.yaml").read_text().splitlines()
                    if not line.startswith("#")
                )
            ),
            {"WeatherForecast": {"action": []}, "Odd: Name": {"action": []}},
        )

    async def test_a_list_declaration_is_refused_with_a_reason(self):
        # `!include_dir_list` has no names, so there is nowhere for a stub to go.
        self.declare("intent_script: !include_dir_list scripts/\n")
        with self.assertRaises(LearnerError) as caught:
            export_intent_script(self.hass, self.entry)
        self.assertIn("no names to register", str(caught.exception))
