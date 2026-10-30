"""Learned *intents*: whole new question types the house answers by itself.

Sentences alone can only bind wording to intents that already exist. This module
closes that gap: the assistant supplies a name, a Jinja template and the words
people say, and the intent is registered with `intent.async_register`.

Nothing is written to `intent_script.yaml`. The user's own intent file is never
touched, and a learned intent can be removed cleanly with `ForgetIntent`.

Why a template and not code: Home Assistant's template environment already gives
`states()`, `state_attr()`, `is_state()` and friends, sandboxed. A rendered
sentence is exactly what an intent response is, so no arbitrary code is needed -
and the template is rendered once at learn time to prove it works.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from pathlib import Path
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import intent as intent_helper
from homeassistant.helpers import template as template_helper
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.util import yaml as yaml_util
from jinja2 import meta

from .const import (
    SPEAKER_FLOOR,
    SPEAKER_SENSOR,
    INTENT_NAME_RE,
    SPEAKER_VARIABLES,
    MAX_TEMPLATE_CHARS,
    SENTENCES_DIR,
    SIGNAL_UPDATED,
)
from .judge import Choice, JudgeError, Noul, async_ask, has_judge
from .registration import export_intent_script, refresh_export  # noqa: F401 - public here
from .learner import (
    FILE_PREFIX,
    bind_slots,
    async_verify,
    template_entities,
    LearnedIntent,
    LearnerData,
    LearnerError,
    _load,
    _validate,
    async_intent_names,
    async_save,
    async_sentence_index,
    speaker_scope_problem,
    language_of,
    normalize_sentence,
    tidy_punctuation,
    note_learned,
    remove_intent_sentences,
    sentence_for_storage,
    sentences_root,
    slugify,
)

_LOGGER = logging.getLogger(__name__)

#: Jinja renders Python None as the word ``None``, which no natural sentence
#: contains - so seeing it means the template read something that was not there.
_LITERAL_NONE = re.compile(r"\bNone\b")

#: A quoted entity id anywhere in a template: `states('sensor.x')` obviously, but also
#: `{% set e = 'sensor.x' %}{{ states(e) }}` - which is how a template picks between
#: entities per speaker, and which `template_entities()` cannot see. A model that invents
#: `sensor.sam_s_phone` in that position produces "last seen in the unknown", and
#: every literal entity id is checkable whether or not the code passes it to states().
_QUOTED_ENTITY = re.compile(r"""['"]([a-z_]+\.[a-z0-9_]+)['"]""")

#: ``last_changed`` and ``last_updated`` are UTC-aware, so calling ``.strftime``
#: straight on them prints UTC - the right minute with the wrong half of the day.
#: Measured at 17:22 NZ: ``...| as_timestamp | timestamp_custom('%-I:%M %p')``
#: said 5:22 PM and ``....strftime('%-I:%M %p')`` said 5:22 AM. Converting first
#: (``as_local``) or going through ``timestamp_custom`` is fine, because neither
#: leaves ``.strftime`` against the raw attribute.
_UTC_STRFTIME = re.compile(r"\.\s*last_(?:changed|updated)\s*\.\s*strftime\b")

#: A slot as a sentence spells it: `{day}`. The name may be anything hassil allows; it is
#: filtered against `_SLOT_NAME` before it is ever used as a Jinja variable.
_SENTENCE_SLOT = re.compile(r"\{([^{}]+)\}")

#: A slot name is lowercase and word-like, so `{day}` cannot be a typo of some other
#: list's name, and hassil's own list syntax is not accidentally reproduced.
_SLOT_NAME = re.compile(r"^[a-z][a-z0-9_]{0,30}$")

#: How many values one slot may carry. A slot is a closed word-class (the weekdays, the rooms,
#: the people), not a free-form capture - a wildcard is the tool for that - and this is the
#: line between the two, not a performance limit. It used to be 40, on the belief that a long
#: list is how a sentence file gets slow to match; measured against hassil 3.12.1, which
#: indexes text list values in a prefix trie, matching is flat across two orders of magnitude
#: (0.068 ms per utterance at 40 values, 0.070 at 1,000, 0.106 at 5,000) and the cost is at
#: *load*: 0.2 ms for 40 values, 3.8 ms for 1,000, once per reload. So a word-class may be
#: generously complete - every room, every person, every scene - and what this still refuses
#: is a catch-all wearing a list's clothes.
MAX_SLOT_VALUES = 200

#: What a wildcard slot renders as in a sample answer, when a template is proved.
_WILDCARD_SAMPLE = "someone"


def normalise_slots(raw: Any) -> dict[str, dict[str, Any]]:
    """Turn what a model sent as ``slots`` into `lists:` entries hassil can use.

    Accepted shapes, because models write all of them::

        {"day": ["today", "tomorrow", "monday"]}
        {"day": "today, tomorrow, monday"}
        {"day": {"values": ["today", "tomorrow"]}}
        {"speaker": {"wildcard": True}}      /  {"speaker": "wildcard"}

    A one-value list is refused rather than installed: the first attempt at this wrote
    ``{"day": "day"}`` while meaning "the day word", and a slot that silently matches
    only the literal word "day" is worse than being told the list is too short.
    """
    if not raw:
        return {}
    if not isinstance(raw, dict):
        raise LearnerError(
            "slots must be an object of slot name to values, for example "
            '{"day": ["today", "tomorrow", "monday"]}'
        )
    slots: dict[str, dict[str, Any]] = {}
    for name, spec in raw.items():
        key = str(name).strip().lower()
        if not _SLOT_NAME.match(key):
            raise LearnerError(
                f"{name!r} is not a usable slot name: use lowercase letters, digits and "
                "underscores, starting with a letter (for example day)"
            )
        if isinstance(spec, dict):
            wildcard = bool(spec.get("wildcard"))
            values = spec.get("values")
        elif isinstance(spec, str) and spec.strip().lower() == "wildcard":
            wildcard, values = True, None
        else:
            wildcard, values = False, spec
        if values is None and not wildcard:
            raise LearnerError(
                f"the slot {key!r} has no values: give it a list, for example "
                f'"{key}": ["today", "tomorrow"]'
            )
        if wildcard:
            if values:
                raise LearnerError(
                    f"the slot {key!r} is both a wildcard and a list of values; pick one"
                )
            slots[key] = {"wildcard": True}
            continue
        if isinstance(values, str):
            values = [part.strip() for part in values.split(",") if part.strip()]
        if not isinstance(values, (list, tuple)):
            raise LearnerError(
                f"the values of slot {key!r} must be a list of words, for example "
                f'"{key}": ["today", "tomorrow"]'
            )
        cleaned: list[str] = []
        for value in values:
            text = str(value).strip()
            if not text or any(char in text for char in "{}[]|"):
                raise LearnerError(
                    f"{text!r} cannot be a value of slot {key!r}: slot values are plain "
                    "words the person says out loud"
                )
            if text.lower() not in {item.lower() for item in cleaned}:
                cleaned.append(text)
        if len(cleaned) < 2:
            raise LearnerError(
                f"the slot {key!r} has {len(cleaned)} value(s). List every value the slot "
                "can take - for a weekday that is all seven days - so a value nobody has "
                "said yet still matches. If the wording really is fixed, leave slots out."
            )
        if len(cleaned) > MAX_SLOT_VALUES:
            raise LearnerError(
                f"the slot {key!r} has {len(cleaned)} values (> {MAX_SLOT_VALUES}); a slot "
                "is a closed word-class, not a catch-all - use a wildcard for free text"
            )
        slots[key] = {"values": cleaned}
    return slots


def slot_sample(slots: dict[str, dict[str, Any]] | None) -> dict[str, str]:
    """One value per slot, so a template can be rendered before anyone asks it.

    A template that reads ``day`` cannot be proved without a day: Jinja renders the
    undefined name as empty text, so ``On {{ day }} it will be rainy`` would be read back
    as ``On  it will be rainy`` and the reviewer asked to judge that.
    """
    sample: dict[str, str] = {}
    for name, spec in (slots or {}).items():
        values = (spec or {}).get("values") or []
        if values:
            sample[name] = str(values[0])
        elif (spec or {}).get("wildcard"):
            sample[name] = _WILDCARD_SAMPLE
    return sample


def parameterise(utterance: str, slots: dict[str, dict[str, Any]] | None) -> str:
    """Replace the values of a declared slot in one example phrasing with ``{slot}``.

    This is what makes one phrasing cover a whole word-class: the model supplies example
    requests in the words a person said ("... on Wednesday") and the list of values the
    slot takes, and every weekday example collapses onto the single sentence
    ``... on {day}``. Longest value first, so "after tomorrow" is not eaten by "tomorrow".
    """
    text = utterance
    for name, spec in (slots or {}).items():
        for value in sorted((spec or {}).get("values") or [], key=len, reverse=True):
            text = re.sub(
                rf"(?<![a-z0-9]){re.escape(value)}(?![a-z0-9])",
                "{" + name + "}",
                text,
                flags=re.IGNORECASE,
            )
    return " ".join(text.split())


def concrete(sentence: str, slots: dict[str, dict[str, Any]] | None) -> str:
    """Fill every slot with one value, so the local matcher can be asked about it."""
    text = sentence
    for name, value in slot_sample(slots).items():
        text = text.replace("{" + name + "}", value)
    return text


def _duplicate_question(candidates: dict[str, str]) -> Choice:
    """A choice whose options are the installed intents that could already answer this."""
    options: dict[str, str] = {
        "none": "No installed intent answers this request; a new one is warranted"
    }
    options.update(candidates)
    return Choice(
        "A new question is about to be taught to this house as its own intent. Does one of "
        "the installed intents below already answer `request` - the SAME question, even in "
        "different words, even with a shorter or longer answer? Judge the question, not the "
        "wording. Pick the installed intent, or none.",
        options,
    )


async def async_duplicate_intent(
    hass: HomeAssistant,
    entry: ConfigEntry,
    utterances: list[str],
    template: str,
    *,
    exclude: str | None = None,
) -> str | None:
    """An installed intent that already answers this request, if there is one.

    Two intents for one question is how a house starts answering the same thing two
    different ways: four weather intents were learned from four wordings and the answer a
    person heard depended on which words they happened to use. The wording is what should
    vary, never the answer, so a create is refused when an installed intent already answers
    the question and the refusal names it to bind to.

    Only intents whose template touches the same entity are offered - a request about the
    weather cannot be the same question as one about the washing machine - and the question
    itself is asked of the judge. No judge, or a judge that cannot be reached, means no
    refusal: this is a guard against waste, not a reason to fail a learning.
    """
    if not has_judge(entry):
        return None
    mine = template_entities(template) | set(_QUOTED_ENTITY.findall(template))
    candidates: dict[str, str] = {}
    for spec in entry.runtime_data.intents:
        if spec.name == exclude:
            continue
        theirs = template_entities(spec.template) | set(
            _QUOTED_ENTITY.findall(spec.template)
        )
        if not mine & theirs:
            continue
        sample = slot_sample(spec.slots)
        says = _render(hass, spec.response or spec.template, sample)[:160]
        candidates[spec.name] = f"{spec.name} - it already answers: {says}"
    if not candidates:
        return None
    state = {
        "request": utterances[0] if utterances else "",
        "also_said_as": utterances[1:6],
        "the_template_being_proposed": template[:600],
    }
    try:
        answers = await async_ask(
            hass, entry, state, {"already_answers": _duplicate_question(candidates)}
        )
    except JudgeError as err:
        _LOGGER.debug("Irabelle Home Learning could not check for a duplicate answer: %s", err)
        return None
    answer = answers.get("already_answers")
    chosen = answer.choice if answer is not None else None
    if not chosen or chosen == "none" or chosen not in candidates:
        return None
    _LOGGER.info(
        "Irabelle Home Learning refused a second answer to %r: %s already answers it",
        state["request"],
        chosen,
    )
    return chosen



class LearnedIntentHandler(intent_helper.IntentHandler):
    """Answer one learned intent by rendering its template."""

    def __init__(self, spec: LearnedIntent) -> None:
        self.intent_type = spec.name
        self.description = spec.description or f"Learned intent {spec.name}"
        # slot_schema is a property on the base class that already returns None,
        # so it is deliberately not set here (assigning would fail).
        self._spec = spec

    @property
    def template_source(self) -> str:
        return self._spec.response or self._spec.template

    async def async_handle(self, intent_obj: intent_helper.Intent) -> intent_helper.IntentResponse:
        """Render the template with the slots and the asking user in scope."""
        hass = intent_obj.hass
        slots = {
            key: (value or {}).get("value")
            for key, value in (intent_obj.slots or {}).items()
        }
        variables = await _speaker_variables(hass, intent_obj)
        reply = self._spec.response or self._spec.template
        rendered = template_helper.Template(reply, hass).async_render(
            {**variables, **slots, "slots": slots}, parse_result=False
        )
        response = intent_obj.create_response()
        response.async_set_speech(str(rendered).strip())
        return response


async def _speaker_variables(
    hass: HomeAssistant, intent_obj: intent_helper.Intent
) -> dict[str, str]:
    """Resolve who is asking, for the template to use.

    This is the thing a hand-written `intent_script` cannot do: its templates are
    rendered with slots only, so it has no way to know whether the request came
    from a signed-in user in the Home Assistant chat (a real `user_id`) or from a
    satellite (no user; VoiceBM's `sensor.voicebm_current_speaker` is the only
    clue). Both are exposed here, always defined, so a template can branch:

        {% set who = user_name or (states('sensor.voicebm_current_speaker') if speaker_certain else '') %}
        {% if who %}...{% else %}I am not sure who is asking.{% endif %}

    `speaker_certain` is "yes" when the name came from a signed-in user (certain by
    definition) or from a voice match confident enough to act on, and empty otherwise, so
    a template can decline rather than guess. `speaker_confidence` is the voice match's own
    number, so a template can pick its own floor.
    """
    variables: dict[str, str] = dict.fromkeys(sorted(SPEAKER_VARIABLES), "")

    context = getattr(intent_obj, "context", None)
    user_id = getattr(context, "user_id", None)
    if user_id:
        variables["user_id"] = user_id
        user = await hass.auth.async_get_user(user_id)
        if user is not None:
            variables["user_name"] = user.name or ""
        for state in hass.states.async_all("person"):
            if state.attributes.get("user_id") == user_id:
                variables["person_entity_id"] = state.entity_id
                variables["person_name"] = state.name
                variables["person_state"] = state.state
                break
        # A signed-in user is not a guess, whatever a voice match would have said.
        variables["speaker_certain"] = "yes"

    if not variables["speaker_certain"]:
        heard = hass.states.get(SPEAKER_SENSOR)
        if heard is not None:
            confidence = heard.attributes.get("confidence")
            try:
                variables["speaker_confidence"] = f"{float(confidence):.2f}"
            except (TypeError, ValueError):
                variables["speaker_confidence"] = ""
            if heard.state not in ("", "unknown", "unavailable", "none") and (
                variables["speaker_confidence"]
                and float(variables["speaker_confidence"]) >= SPEAKER_FLOOR
            ):
                variables["speaker_certain"] = "yes"

    devices = dr.async_get(hass)
    areas = ar.async_get(hass)
    for key in ("device_id", "satellite_id"):
        value = getattr(intent_obj, key, None)
        if not value:
            continue
        variables[key] = value
        device = devices.async_get(value)
        if device is None or not device.area_id:
            continue
        area = areas.async_get_area(device.area_id)
        if area is None:
            continue
        if not variables["user_area"]:
            variables["user_area"] = area.name

    return variables


def register_intent(hass: HomeAssistant, spec: LearnedIntent) -> None:
    """Register (or replace) the handler for a learned intent."""
    unregister_intent(hass, spec.name)
    intent_helper.async_register(hass, LearnedIntentHandler(spec))


def unregister_intent(hass: HomeAssistant, name: str) -> None:
    """Remove a handler, but only one this integration owns."""
    for handler in list(intent_helper.async_get(hass)):
        if handler.intent_type == name and isinstance(handler, LearnedIntentHandler):
            intent_helper.async_remove(hass, name)
            return


def restore_intents(hass: HomeAssistant, data: LearnerData) -> None:
    """Re-register every stored learned intent (called at setup)."""
    for spec in data.intents:
        register_intent(hass, spec)


def release_intents(hass: HomeAssistant, data: LearnerData) -> None:
    """Remove every handler this integration registered (called at unload)."""
    for spec in data.intents:
        unregister_intent(hass, spec.name)


def undeclared_variables(
    hass: HomeAssistant, source: str, defined: set[str] | None = None
) -> set[str]:
    """Return the names a template reads that nothing will define.

    This is the check that catches the tempting mistake: writing a template
    around a slot (``name``, ``room``) that the sentences never supply, so it
    renders a polite "I could not find that" forever. ``defined`` is the set of slot
    names the intent declares under `lists:` - those are supplied by the matcher when the
    sentence that uses them matches, and are the one legitimate exception.
    """
    environment = template_helper.TemplateEnvironment(hass)
    try:
        parsed = environment.parse(source)
    except Exception:  # noqa: BLE001 - a parse failure is reported by the render check
        return set()
    return (
        meta.find_undeclared_variables(parsed)
        - set(environment.globals)
        - SPEAKER_VARIABLES
        - {"slots"}
        - set(defined or ())
    )


def _render(hass: HomeAssistant, source: str, sample: dict[str, str] | None = None) -> str:
    """What a template says right now, for a caller to show back to whoever wrote it."""
    try:
        return str(
            template_helper.Template(source, hass).async_render(
                {**(sample or {}), "slots": sample or {}}, parse_result=False
            )
        ).strip()
    except Exception:  # noqa: BLE001 - a render failure is the caller's business
        return "(it failed to render)"


def _check_template(
    hass: HomeAssistant,
    source: str,
    defined: set[str] | None = None,
    sample: dict[str, str] | None = None,
) -> None:
    """Prove the template parses and renders before it is ever used.

    Rendering is not enough on its own: a template that reads something which is
    not there does not raise, it renders the word ``None`` and the house says it
    out loud. ``state_attr(sensor, 'last_updated')`` is the usual cause - the
    timestamp lives on the state object, not in its attributes - and it produced
    "your phone was last seen in the Bedroom at None" before this check existed.

    A declared slot is rendered with the first of its values, so a template written around
    a slot is judged on one real answer rather than on undefined names left blank.
    """
    undefined = undeclared_variables(hass, source, defined)
    if undefined:
        raise LearnerError(
            "the template reads "
            + ", ".join(sorted(undefined))
            + " - nothing defines that when a learned sentence is answered. If the answer "
            "varies by one word people say - a day, a room, a person - declare that word "
            "as a slot (its values in `slots`, twice over: the list, and {name} where the "
            "word goes in the phrasing) and the template may read it. Otherwise write the "
            "template for the exact request."
        )
    # The entity check comes before the render on purpose: a template that indexes an
    # attribute of an entity that is not there (`state_attr('weather.home_home_forecast',
    # 'forecast')[0]`) raises "None has no element 0", which names nothing a model can act
    # on. The missing entity is the actual mistake, so it is reported first.
    wanted = set(template_entities(source))
    wanted.update(_QUOTED_ENTITY.findall(source))
    missing = sorted(
        entity_id for entity_id in wanted if hass.states.get(entity_id) is None
    )
    if missing:
        raise LearnerError(
            "the template reads "
            + ", ".join(missing)
            + ", which does not exist. "
            + _did_you_mean(hass, missing)
            + "Read an entity from the exposed list: the state of an entity that is not "
            "there renders as the word unknown, and the house would say it out loud."
        )

    try:
        rendered = template_helper.Template(source, hass).async_render(
            {**(sample or {}), "slots": sample or {}}, parse_result=False
        )
    except Exception as err:  # noqa: BLE001 - any failure means "do not install"
        raise LearnerError(f"the template failed to render: {err}") from err

    if _UTC_STRFTIME.search(source):
        raise LearnerError(
            "the template calls strftime on last_changed or last_updated, which are "
            "UTC, so the time comes out 12 hours wrong (AM for PM). Convert first: "
            "{{ value | as_timestamp | timestamp_custom('%-I:%M %p') }} or "
            "{{ (value | as_local).strftime('%-I:%M %p') }}"
        )

    text = str(rendered)
    if _LITERAL_NONE.search(text):
        raise LearnerError(
            f"the template renders {text.strip()[:120]!r} - a None in the answer "
            "means it reads something that is not there. last_updated and "
            "last_changed are on the state, not in its attributes: use "
            "states.<entity_id>.last_updated, not "
            "state_attr(<entity_id>, 'last_updated')."
        )


async def async_learn_intent(
    hass: HomeAssistant,
    entry: ConfigEntry,
    *,
    name: str,
    template: str,
    utterances: list[str],
    description: str | None = None,
    note: str | None = None,
    source: str | None = None,
    response: str | None = None,
    slots: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a new intent (template + sentences) and make it live.

    ``slots`` declares the word-classes the wording varies by (``{"day": ["today",
    "tomorrow", ...]}``). Every example phrasing that contains one of a slot's values is
    rewritten to use ``{day}``, so the list of weekdays the model wrote becomes one
    sentence that answers any weekday - including one never said before - and the
    template is allowed to read ``day``.
    """
    data: LearnerData = entry.runtime_data
    language = language_of(entry)

    name = str(name).strip()
    template = tidy_punctuation(str(template).strip())
    if not re.match(INTENT_NAME_RE, name):
        raise LearnerError(
            "the intent name must start with a letter and contain only letters, "
            "digits and underscores (for example WashingMachineStatus)"
        )
    if len(template) > MAX_TEMPLATE_CHARS:
        raise LearnerError(f"the template is too long (> {MAX_TEMPLATE_CHARS} chars)")

    owned = {spec.name for spec in data.intents}
    existing = await async_intent_names(hass)
    if name in existing and name not in owned:
        raise LearnerError(
            f"an intent called {name!r} already exists - pick another name"
        )

    previous = next((spec for spec in data.intents if spec.name == name), None)
    declared = normalise_slots(slots)
    if not declared and previous is not None and previous.slots:
        # A correction that says nothing about slots keeps the ones already installed.
        declared = previous.slots

    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in utterances:
        text = str(raw).strip()
        if text and text.lower() not in seen:
            seen.add(text.lower())
            cleaned.append(text)

    if declared:
        # One phrasing per value collapses onto one phrasing with the slot in it, which
        # is the whole point: "how is gonna be the weather on Wednesday" and "... on
        # Monday" become "... on {day}".
        folded: list[str] = []
        for text in cleaned:
            written = parameterise(text, declared)
            if written.lower() not in {item.lower() for item in folded}:
                folded.append(written)
        cleaned = folded

    index = async_sentence_index(hass, language)
    # A proposed variant may be wording the local matcher already answers (a
    # canonical phrasing like "who am I" usually is). Dropping it keeps the
    # never-shadow guarantee without throwing away the words the person
    # actually used, which are the point of the exercise.
    kept = [u for u in cleaned if index.get(normalize_sentence(u), name) == name]
    if not kept:
        raise LearnerError(
            "every proposed phrasing is already answered by another intent: "
            + ", ".join(repr(u) for u in cleaned[:5])
        )
    _validate(kept, name, index, existing | {name}, declared=set(declared))
    if problem := speaker_scope_problem(hass, template, kept):
        raise LearnerError(problem)
    sample = slot_sample(declared)
    _check_template(hass, template, defined=set(declared), sample=sample)

    if previous is None and (
        duplicate := await async_duplicate_intent(hass, entry, kept, template, exclude=name)
    ):
        raise LearnerError(
            f"{duplicate!r} already answers this request, and a second intent would make the "
            "house answer the same question two different ways depending on the words used. "
            f"Bind these words to {duplicate!r} instead (LearnUtterance) - and if the answer "
            f"itself should change, change {duplicate!r}'s reply rather than adding another "
            "intent that answers the same thing."
        )

    # `normalize_sentence`, not `lower()`: the index is keyed by the normalised
    # form, so a sentence carrying punctuation would look new every time.
    fresh = [u for u in kept if normalize_sentence(u) not in index]
    if not fresh and previous is None:
        result = {
            "intent": name,
            "added": [],
            "already_present": kept,
            "note": "already answered locally; nothing to install",
        }
        data.last_result = result
        async_dispatcher_send(hass, SIGNAL_UPDATED)
        return result

    # Reusing the existing file matters when only the template is being
    # corrected: re-deriving the name would leave the sentences behind in the old
    # file and register an intent whose file is empty.
    #
    # The file is named after the *intent*, not after the note that happened to come with
    # the write. A note is a sentence ("weekday slot picks the forecast entry dated the
    # next..."), and a file named after one says nothing about what is inside it - the
    # notes that produced `learned-fixed-weekday-picks-the-forecast-entry-dated-the-next-
    # wednes.yaml` and `learned-user-asked-for-tomorrow-but-customweatherforecast-
    # answered-t.yaml` were accurate and useless.
    slug = slugify(name)
    relative = (
        previous.file
        if previous is not None and not fresh
        else f"{SENTENCES_DIR}/{language}/{FILE_PREFIX}{slug}.yaml"
    )
    path = Path(hass.config.path(relative))
    if fresh:
        _write_sentences(path, name, fresh, lists=declared or None)

    # An explicit response survives a template correction; without one the reply is the
    # template and follows it.
    reply = (response or "").strip() or (
        previous.response if previous is not None and previous.response else None
    )
    write_response(path, name, reply or template, template)
    spec = LearnedIntent(
        name=name,
        template=template,
        response=reply,
        description=description,
        note=note,
        source=source,
        slots=declared or None,
        file=relative,
    )
    data.intents = [s for s in data.intents if s.name != name] + [spec]
    register_intent(hass, spec)
    # Before the save, or the record of what was learned is only in memory.
    note_learned(
        data,
        fresh[0] if fresh else f"the reply for {name}",
        name,
        "created a new intent" if previous is None else "corrected an intent",
        file=relative,
    )
    await async_save(hass, entry)
    await hass.services.async_call("conversation", "reload", {}, blocking=True)
    result = {
        "intent": name,
        "template": template,
        "response": reply or template,
        "says": _render(hass, reply or template, sample)[:200],
        "file": relative,
        "added": fresh,
        "already_present": [u for u in kept if u not in fresh],
        "warnings": missing_attributes(hass, reply or template),
        "reads": _reads(hass, reply or template),
        "verified": await async_verify(
            hass, language, [concrete(u, declared) for u in fresh], name
        ),
    }
    if declared:
        result["slots"] = declared
    data.last_result = result
    async_dispatcher_send(hass, SIGNAL_UPDATED)
    return result


async def async_update_template(
    hass: HomeAssistant,
    entry: ConfigEntry,
    name: str,
    template: str,
    description: str | None = None,
) -> dict[str, Any]:
    """Replace a learned intent's template, keeping its sentences.

    Needed because a template cannot be corrected by re-learning: its sentences
    already exist and match, so the write would be treated as "nothing to
    install" and the same wrong answer would stay.
    """
    data: LearnerData = entry.runtime_data
    spec = next((item for item in data.intents if item.name == name), None)
    if spec is None:
        raise LearnerError(f"{name!r} is not a learned intent")

    template = tidy_punctuation(str(template).strip())
    if not template:
        raise LearnerError("the template is empty")
    if len(template) > MAX_TEMPLATE_CHARS:
        raise LearnerError(f"the template is too long (> {MAX_TEMPLATE_CHARS} chars)")
    _check_template(
        hass, template, defined=set(spec.slots or {}), sample=slot_sample(spec.slots)
    )

    updated = LearnedIntent(
        name=name,
        template=template,
        description=description or spec.description,
        note=spec.note,
        source=f"review:{spec.source or 'unknown'}",
        slots=spec.slots,
        file=spec.file,
    )
    data.intents = [s for s in data.intents if s.name != name] + [updated]
    register_intent(hass, updated)
    # The file keeps both halves: the reply as `default` and the computation as `template`
    # when the two differ, so a correction never loses the wording a person asked for.
    write_response(
        Path(hass.config.path(updated.file)), name, updated.response or template, template
    )
    await async_save(hass, entry)
    await hass.services.async_call("conversation", "reload", {}, blocking=True)
    warnings = missing_attributes(hass, template)
    data.last_result = {
        "intent": name,
        "template": template,
        "corrected": True,
        "warnings": warnings,
        "reads": _reads(hass, template),
    }
    if warnings:
        _LOGGER.warning(
            "Irabelle Home Learning: the template for %s %s", name, "; ".join(warnings)
        )
    async_dispatcher_send(hass, SIGNAL_UPDATED)
    _LOGGER.warning("Irabelle Home Learning corrected the template for %s", name)
    return data.last_result


def _intent_slots(data: dict[str, Any], intent: str) -> set[str]:
    """Every slot name the intent's own file supplies to its reply.

    The file is the authority, so this needs no store: a name is a slot when the sentences
    use it (`{day}`), when a sentence pins it (`slots: {day: tomorrow}`), or when `lists:`
    defines it.
    """
    names = {str(name).strip().lower() for name in (data.get("lists") or {})}
    entry = (data.get("intents") or {}).get(intent) or {}
    for datum in entry.get("data") or []:
        names.update(str(name).strip().lower() for name in (datum.get("slots") or {}))
        for sentence in datum.get("sentences") or []:
            names.update(_SENTENCE_SLOT.findall(str(sentence)))
    return {name for name in names if _SLOT_NAME.match(name)}


def write_response(path: Path, intent: str, reply: str, template: str | None = None) -> None:
    """Write one intent's reply into its learner-owned sentence file.

    This is where Home Assistant itself keeps the reply for a built-in intent -
    `responses: intents: <name>: default` - so putting a learned intent's reply there
    makes the house's wording a thing you can read in the file next to the sentences, and
    a thing this component can rewrite when asked to answer differently.

    ``template`` is written beside it as an extra key when it differs from the reply. Home
    Assistant ignores a response key it never asks for, so the file can hold both halves of
    a learned intent - what it says, and the computation behind it - and the folder is then
    the whole intent: something a person can read, edit, back up, keep after removing this
    component, or copy to another house. The store is provenance, not storage.

    The reply is stored with its slot bindings (see `bind_slots`), and ``template`` without
    them, so the file shows both the text as written and the text as rendered.
    """
    data: dict[str, Any] = {"language": "en", "intents": {}}
    if path.exists():
        try:
            data = _load(path) or data
        except LearnerError:
            data = {"language": "en", "intents": {}}
    data.setdefault("language", "en")
    responses = data.setdefault("responses", {})
    entry: dict[str, Any] = {"default": bind_slots(reply, _intent_slots(data, intent))}
    if template and template.strip() and template.strip() != reply.strip():
        entry["template"] = template
    responses.setdefault("intents", {})[intent] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml_util.dump(data), encoding="utf-8")


#: ``state_attr('sensor.x', 'attr')`` — the one way a template reads a named attribute
#: literally. A read whose name comes from a variable cannot be checked and is not matched
#: here, which is the honest thing: this is a warning about the readable case.
_STATE_ATTR = re.compile(
    r"""state_attr\(\s*['"]([a-z_]+\.[a-z0-9_]+)['"]\s*,\s*['"]([^'"]+)['"]\s*\)"""
)

#: Names that belong to the state object itself rather than to `attributes`.
_STATE_FIELDS = frozenset(
    {
        "state",
        "entity_id",
        "name",
        "domain",
        "attributes",
        "context",
        "last_changed",
        "last_updated",
        "last_reported",
    }
)


def _did_you_mean(hass: HomeAssistant, unknown: list[str], limit: int = 4) -> str:
    """Name the entity an invented id is probably a misspelling of.

    A model that has just been shown an entity's *friendly name* writes the id it would
    have had: "Home Home Forecast" becomes `weather.home_home_forecast`, which does not
    exist because the id is `weather.home`. The refusal said that, correctly, fifteen times
    in one turn while the model sent the same call again - what was missing is what to use
    instead. The id it should have written almost always shares the domain and most of its
    words, so a cheap near-match is the way out of the loop.
    """
    candidates = [state.entity_id for state in hass.states.async_all()]
    suggestions: list[str] = []
    for wanted in unknown[:limit]:
        domain = wanted.partition(".")[0]
        words = {word for word in re.split(r"[^a-z0-9]+", wanted.lower()) if word}
        scored: list[tuple[float, str]] = []
        for candidate in candidates:
            if not candidate.startswith(f"{domain}."):
                continue
            other = {word for word in re.split(r"[^a-z0-9]+", candidate.lower()) if word}
            shared = len(words & other)
            if shared:
                scored.append((shared - 0.1 * abs(len(other) - len(words)), candidate))
        scored.sort(reverse=True)
        if scored:
            suggestions.append(f"{wanted} -> {scored[0][1]}")
    return ("Did you mean: " + ", ".join(suggestions) + "? ") if suggestions else ""


def missing_attributes(hass: HomeAssistant, source: str) -> list[str]:
    """Attribute names a template reads that the entity does not have right now.

    ``state_attr`` on a name that is not there returns None without raising, so a template
    written around it degrades silently. A model asked to make the weather answer more
    detailed read ``weather.home.forecast`` - the attribute is ``forecast_daily`` - and the
    house quietly answered with the current conditions instead: the reply looked plausible,
    ``verified`` said ok, and every mechanical check passed.

    Reported, not refused: reading an optional attribute is legitimate. An integration may
    or may not provide ``forecast_hourly``, and a template that guards it with ``or []`` is
    correct. What is not legitimate is not being told.
    """
    warnings: list[str] = []
    seen: set[tuple[str, str]] = set()
    for entity_id, attribute in _STATE_ATTR.findall(source):
        if (entity_id, attribute) in seen:
            continue
        seen.add((entity_id, attribute))
        state = hass.states.get(entity_id)
        if state is None:
            continue  # a missing entity is its own, louder complaint
        if attribute in state.attributes or attribute in _STATE_FIELDS:
            continue
        available = ", ".join(sorted(state.attributes)[:12]) or "no attributes at all"
        warnings.append(
            f"it reads {entity_id}.{attribute}, which that entity does not have (it has: "
            f"{available}); a missing attribute renders empty, so the answer may quietly "
            "fall back to something else"
        )
    return warnings


def _tidy_note(original: str, tidied: str) -> list[str]:
    """Which punctuation was cleaned up, in the words of the check that would have flagged it."""
    from .autolearn import reply_problems  # noqa: PLC0415 - autolearn imports this module

    return [
        problem
        for problem in reply_problems(original)
        if "two punctuation marks" in problem
    ] or ["a doubled punctuation mark was cleaned up before it was installed"]


def _reads(hass: HomeAssistant, source: str) -> dict[str, str]:
    """The entities a template reads, with the state it will read from each.

    Returned with every write so the model that wrote the template sees the real ids it
    ended up using - not the names it was imagining. A model that has just written
    `weather.home_home_forecast` reads back `weather.home = rainy` and can see the
    difference in the same tool result, instead of learning it from a refusal.
    """
    found: dict[str, str] = {}
    for entity_id in sorted(set(template_entities(source)) | set(_QUOTED_ENTITY.findall(source))):
        state = hass.states.get(entity_id)
        found[entity_id] = str(state.state) if state is not None else "(not in this house)"
    return found


def _write_lists(path: Path, lists: dict[str, Any] | None) -> None:
    """Set the `lists:` block of a learner file, keeping its sentences and replies.

    A slot list belongs to the file, so widening one - adding today and tomorrow to the
    weekdays - is a change to the file rather than to the sentences that use it.
    """
    if not lists:
        return
    data: dict[str, Any] = {"language": "en", "intents": {}}
    if path.exists():
        try:
            data = _load(path) or data
        except LearnerError:
            data = {"language": "en", "intents": {}}
    data.setdefault("language", "en")
    declared = data.setdefault("lists", {})
    for name, spec in lists.items():
        declared[name] = spec
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml_util.dump(data), encoding="utf-8")


def learner_sentences(
    hass: HomeAssistant, language: str, names: set[str] | None = None
) -> dict[str, list[str]]:
    """Every sentence in this integration's own files, grouped by intent.

    Read from the files rather than from the store, because the files are what the matcher
    loads: a sentence written by an older version, or edited by hand afterwards, is still a
    sentence - and a merge that moved only what the store remembered would leave the rest
    behind as wording whose intent no longer exists, which stops matching for every agent.
    """
    found: dict[str, list[str]] = {}
    root = sentences_root(hass, language)
    if not root.is_dir():
        return found
    for path in sorted(root.glob(f"{FILE_PREFIX}*.yaml")):
        try:
            content = _load(path)
        except LearnerError:
            continue
        for name, body in (content.get("intents") or {}).items():
            if names is not None and name not in names:
                continue
            for item in (body or {}).get("data") or []:
                for sentence in (item or {}).get("sentences") or []:
                    if isinstance(sentence, str) and sentence.strip():
                        known = found.setdefault(name, [])
                        if sentence not in known:
                            known.append(sentence)
    return found


def duplicate_report(hass: HomeAssistant, entry: ConfigEntry) -> dict[str, Any]:
    """Learned intents grouped by the entity they read, with what each says.

    Two intents reading the same entity are candidates for being the same question, and the
    rendered answer is what makes that readable at a glance: two intents that both read
    `weather.home` and both say "it will be cloudy, high 16" are one question answered
    twice, while a weather intent and a rain intent are not. Grouping by one shared entity
    rather than by an identical entity set matters - the same question is usually asked by
    one template that also reads a sensor and another that does not, and those two would
    never land in the same set.

    No model is involved: this is the evidence for a merge, not the decision.
    """
    by_entity: dict[str, list[dict[str, Any]]] = {}
    for spec in entry.runtime_data.intents:
        source = spec.response or spec.template
        entries = sorted(template_entities(source) | set(_QUOTED_ENTITY.findall(source)))
        sample = slot_sample(spec.slots)
        member = {
            "intent": spec.name,
            "file": spec.file,
            "slots": spec.slots,
            "says": _render(hass, source, sample)[:200],
        }
        for entity_id in entries:
            by_entity.setdefault(entity_id, []).append(member)
    shared = [
        {"entity_id": entity_id, "intents": members}
        for entity_id, members in by_entity.items()
        if len(members) > 1
    ]
    shared.sort(key=lambda group: (-len(group["intents"]), group["entity_id"]))
    return {
        "entities": shared,
        "note": (
            "Each entry is one entity read by more than one learned intent. Intents under "
            "the same entity may answer the same question in different words - compare "
            "what they say above, and merge the ones that do with irabelle_home_learning.merge so the "
            "wording varies and the answer does not."
        ),
    }


#: How sure the judge has to be that two intents answer one question. Like every threshold
#: here: a starting point to be replaced by measurement. Being wrong means a merge that throws
#: away a distinction nobody can recover from the file, so it is cautiously high.
SAME_CLEAN = 0.80

#: The most pairs to ask about in one report. Groups are capped rather than sampled, so the
#: pairs a person sees are the whole story for the intents they are looking at.
MAX_DUPLICATE_PAIRS = 40


def duplicate_pairs(report: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """The pairs worth judging, from the entity groups `duplicate_report` already found."""
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    seen: set[tuple[str, str]] = set()
    for group in report.get("entities") or []:
        members = group.get("intents") or []
        for index, a in enumerate(members):
            for b in members[index + 1 :]:
                key = tuple(sorted((a.get("intent", ""), b.get("intent", ""))))
                if key in seen:
                    continue
                seen.add(key)
                pairs.append((a, b))
                if len(pairs) >= MAX_DUPLICATE_PAIRS:
                    return pairs
    return pairs


async def async_duplicate_verdicts(
    hass: HomeAssistant, entry: ConfigEntry, report: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Ask the judge which of the intents sharing an entity really are one question.

    `duplicate_report` is the evidence - two intents that read the same sensor and answer with
    the same shape - and this is the decision, which is the part a model has to make: "your
    phone was last seen in the Bedroom" and "your phone was last seen in the Bedroom, two
    minutes ago" are one question, while "where is it" and "when was it" are two, and the prose
    looks equally similar either way. One request covers every pair.
    """
    pairs = duplicate_pairs(report if report is not None else duplicate_report(hass, entry))
    if not pairs:
        return []
    spoken = learner_sentences(
        hass, language_of(entry), {item["intent"] for pair in pairs for item in pair}
    )
    state = {
        "pairs": [
            {
                "id": f"p{index}",
                "a": {"intent": a["intent"], "asked_as": (spoken.get(a["intent"]) or [""])[0], "says": a.get("says")},
                "b": {"intent": b["intent"], "asked_as": (spoken.get(b["intent"]) or [""])[0], "says": b.get("says")},
            }
            for index, (a, b) in enumerate(pairs)
        ]
    }
    answers = await async_ask(
        hass,
        entry,
        state,
        {
            f"p{index}": Noul(
                f"Two intents in a smart home. `a` is {a['intent']}, asked as "
                f"\"{(spoken.get(a['intent']) or [''])[0]}\" and it says \"{a.get('says')}\". "
                f"`b` is {b['intent']}, asked as \"{(spoken.get(b['intent']) or [''])[0]}\" "
                f"and it says \"{b.get('says')}\". Could ONE reply - the richer of the two - "
                "answer both? They are the same question when the difference is only how much "
                "is said (the area, and how long ago, and the time are all parts of one answer "
                "about the same thing). They are different questions when one asks where and "
                "the other asks when, which are answered differently on purpose.",
                true="One reply could answer both without losing anything real",
                false="They are different questions, or merging them would lose an answer",
            )
            for index, (a, b) in enumerate(pairs)
        },
        rate_limited=False,
    )
    verdicts: list[dict[str, Any]] = []
    for index, (a, b) in enumerate(pairs):
        answer = answers.get(f"p{index}")
        score = (answer.noul or 0.0) if answer is not None else 0.0
        verdicts.append(
            {
                "a": a["intent"],
                "b": b["intent"],
                "score": round(score, 3),
                "same": score >= SAME_CLEAN,
            }
        )
    verdicts.sort(key=lambda row: -row["score"])
    return verdicts


async def async_rename_intent(
    hass: HomeAssistant, entry: ConfigEntry, name: str, to: str
) -> dict[str, Any]:
    """Give a learned intent another name, and its file the matching one.

    A name that no longer says what the intent does is a trap: `WeatherWeekday` that answers
    "what's the weather tomorrow" is deleted by the next person tidying up, and a file named
    after the note that happened to create it says nothing at all. Renaming moves the spec,
    the file, its sentences (pinned slot values and all), its reply, its `lists:` block and
    every phrase record together. The wording and the answer are untouched.
    """
    data: LearnerData = entry.runtime_data
    language = language_of(entry)
    spec = next((item for item in data.intents if item.name == name), None)
    if spec is None:
        raise LearnerError(f"{name!r} is not a learned intent")
    to = str(to).strip()
    if not re.match(INTENT_NAME_RE, to):
        raise LearnerError(
            "the new name must start with a letter and contain only letters, digits and "
            "underscores (for example WeatherForecast)"
        )
    if to == name:
        raise LearnerError("the new name is the same as the old one")
    if any(item.name == to for item in data.intents) or to in await async_intent_names(hass):
        raise LearnerError(f"an intent called {to!r} already exists")

    relative = f"{SENTENCES_DIR}/{language}/{FILE_PREFIX}{slugify(to)}.yaml"
    target_path = Path(hass.config.path(relative))
    target: dict[str, Any] = {"language": language, "intents": {}}
    if target_path.exists():
        try:
            target = _load(target_path) or target
        except LearnerError:
            target = {"language": language, "intents": {}}
    target.setdefault("language", language)

    root = sentences_root(hass, language)
    files: list[str] = []
    for path in sorted(root.glob(f"{FILE_PREFIX}*.yaml")):
        try:
            content = _load(path)
        except LearnerError:
            continue
        if path.resolve() == target_path.resolve():
            # The file already carries the new name's slug: rename the key in place.
            if (block := (content.get("intents") or {}).pop(name, None)) is not None:
                content.setdefault("intents", {})[to] = block
                files.append(path.name)
            responses = (content.get("responses") or {}).get("intents") or {}
            if (reply := responses.pop(name, None)) is not None:
                responses[to] = reply
                files.append(path.name)
                content.setdefault("responses", {})["intents"] = responses
            target = content
            continue
        intents = content.get("intents") or {}
        responses = (content.get("responses") or {}).get("intents") or {}
        block = intents.pop(name, None)
        reply = responses.pop(name, None)
        if block is None and reply is None:
            continue
        files.append(path.name)
        if block is not None:
            target.setdefault("intents", {})[to] = block
        if reply is not None:
            target.setdefault("responses", {}).setdefault("intents", {})[to] = reply
        if declared := content.get("lists"):
            # A `lists:` block belongs to the file, and a slot sentence without it stops
            # matching for every agent.
            target.setdefault("lists", {}).update(declared)
        if intents:
            path.write_text(yaml_util.dump(content), encoding="utf-8")
        else:
            path.unlink(missing_ok=True)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(yaml_util.dump(target), encoding="utf-8")

    renamed = LearnedIntent(
        name=to,
        template=spec.template,
        response=spec.response,
        description=spec.description,
        note=spec.note,
        source=spec.source,
        slots=spec.slots,
        learned_at=spec.learned_at,
        file=relative,
    )
    # The phrase records follow the intent. They are the provenance of wording that is still
    # installed, so they move rather than being removed with the old name.
    for phrase in data.phrases:
        if phrase.intent == name:
            phrase.intent = to
            phrase.file = relative
    unregister_intent(hass, name)
    data.intents = [item for item in data.intents if item.name != name] + [renamed]
    register_intent(hass, renamed)
    await async_save(hass, entry)
    await hass.services.async_call("conversation", "reload", {}, blocking=True)
    result = {
        "renamed": name,
        "to": to,
        "file": relative,
        "moved_from": files,
        "verified": await async_verify(
            hass,
            language,
            [
                concrete(text, spec.slots)
                for text in learner_sentences(hass, language, {to}).get(to, [])[:3]
            ],
            to,
        ),
    }
    data.last_result = result
    async_dispatcher_send(hass, SIGNAL_UPDATED)
    _LOGGER.warning("Irabelle Home Learning renamed %s to %s (%s)", name, to, relative)
    return result


async def async_merge_intents(
    hass: HomeAssistant,
    entry: ConfigEntry,
    *,
    into: str,
    others: list[str],
    template: str | None = None,
    reply: str | None = None,
    phrasings: list[str] | None = None,
    slots: dict[str, Any] | None = None,
    apply: bool = True,
) -> dict[str, Any]:
    """Fold other intents into one, so one question has one answer.

    This is the other half of refusing a duplicate: four intents can already answer the same
    weather question in four different lengths, and which one a person hears then depends on
    the words they happened to use. Every phrasing moves onto one intent, parameterised
    where that intent has a slot for the word, and the others are forgotten - so the wording
    still varies and the answer does not.

    ``phrasings`` carries wording that is not an intent of its own: a phrase *bound* to a
    built-in intent has no learned intent to move, and it is exactly where the inconsistency
    hides, because the built-in's reply may not even take the slot the words imply.
    ``slots`` widens the target's own word-class first, so a merge can fold "today" and
    "tomorrow" into a weekday list and have all of them become one `{day}` sentence.

    ``apply=False`` reports the plan and changes nothing, which is what a person asking
    "what would you merge" wants; the model-facing caller applies.
    """
    data: LearnerData = entry.runtime_data
    language = language_of(entry)
    target = next((spec for spec in data.intents if spec.name == into), None)
    if target is None:
        raise LearnerError(f"{into!r} is not a learned intent")
    doomed = [name for name in others if name and name != into]
    for name in doomed:
        if not any(spec.name == name for spec in data.intents):
            raise LearnerError(f"{name!r} is not a learned intent")
    if slots is not None:
        target.slots = normalise_slots(slots) or None

    harvested = learner_sentences(hass, language, set(doomed))
    index = async_sentence_index(hass, language)
    moved: list[str] = []
    skipped: list[dict[str, str]] = []
    for name in doomed:
        for sentence in harvested.get(name, []):
            written = parameterise(sentence, target.slots)
            if written in moved:
                continue
            owner = index.get(normalize_sentence(written))
            if owner is not None and owner not in (into, *doomed):
                skipped.append({"sentence": written, "already_answered_by": owner})
                continue
            moved.append(written)
    for sentence in phrasings or []:
        written = parameterise(str(sentence).strip(), target.slots)
        if written and written not in moved:
            moved.append(written)

    plan = {
        "into": into,
        "merge": doomed,
        "moved": moved,
        "skipped": skipped,
        "template": "replaced" if template else "kept",
        "reply": "replaced" if reply else "kept",
        "applied": apply,
    }
    if not apply:
        return plan

    if template:
        await async_update_template(hass, entry, into, template)
    if reply:
        await async_set_response(hass, entry, into, reply)
    target = next((spec for spec in data.intents if spec.name == into), target)
    if target.slots:
        _write_lists(Path(hass.config.path(target.file)), target.slots)
    if moved:
        _write_sentences(
            Path(hass.config.path(target.file)), into, moved, lists=target.slots
        )
        await hass.services.async_call("conversation", "reload", {}, blocking=True)
    for name in doomed:
        await async_forget_intent(hass, entry, name)
    plan["verified"] = await async_verify(
        hass, language, [concrete(text, target.slots) for text in moved[:3]], into
    )
    _LOGGER.warning(
        "Irabelle Home Learning merged %s into %s (%d phrasing(s) moved)",
        ", ".join(doomed) or "nothing",
        into,
        len(moved),
    )
    return plan


async def async_consolidate_intents(
    hass: HomeAssistant, entry: ConfigEntry
) -> dict[str, list[str]]:
    """Give every intent one file: its own.

    A learned intent's sentences end up wherever the write that added them put them - a
    binding in one file, a generalisation in another, a correction in a third, the reply
    in whichever of them happened to be the intent's file. Reading what the house will
    say therefore means opening several files and guessing which is authoritative, and the
    same intent can be found in three.

    Each intent is moved to `learned-<intent name>.yaml`, with its reply beside it, and any
    learner file left with nothing in it is removed. Hand-written files are read for their
    intents (so a binding to one is not treated as unknown) and never written.

    Returns what it wrote and removed, so a caller can report it rather than guess.
    """
    data: LearnerData = entry.runtime_data
    language = language_of(entry)
    root = sentences_root(hass, language)
    written: list[str] = []
    removed: list[str] = []
    if not root.is_dir():
        return {"written": written, "removed": removed}

    # Harvest every sentence and reply this integration wrote, grouped by intent.
    items: dict[str, list[Any]] = {}
    replies: dict[str, str] = {}
    lists_by_intent: dict[str, dict[str, Any]] = {}
    sources: list[Path] = []
    seen: dict[str, set[str]] = {}
    for path in sorted(root.glob(f"{FILE_PREFIX}*.yaml")):
        try:
            content = _load(path)
        except LearnerError:
            continue
        sources.append(path)
        for name, body in (content.get("intents") or {}).items():
            bucket = items.setdefault(name, [])
            known = seen.setdefault(name, set())
            for item in (body or {}).get("data") or []:
                block = dict(item or {})
                fresh = [
                    sentence
                    for sentence in block.get("sentences") or []
                    if isinstance(sentence, str)
                    and normalize_sentence(sentence) not in known
                    and not known.add(normalize_sentence(sentence))
                ]
                if not fresh:
                    continue
                block["sentences"] = fresh
                bucket.append(block)
                # A `lists:` block belongs to the file, and moving the sentences to a new
                # file without it leaves `{day}` undefined - which stops sentence matching
                # for every agent, not just this intent.
                for slot in re.findall(r"\{([^{}]*)\}", " ".join(fresh)):
                    if (spec := (content.get("lists") or {}).get(slot.strip())) is not None:
                        lists_by_intent.setdefault(name, {})[slot.strip()] = spec
        for name, body in ((content.get("responses") or {}).get("intents") or {}).items():
            reply = (body or {}).get("default")
            if isinstance(reply, str) and reply.strip():
                replies.setdefault(str(name), reply)

    # Anything that ended up with no sentences at all is not an intent any more.
    planned: dict[Path, dict[str, Any]] = {}
    files_by_intent: dict[str, str] = {}
    for name in sorted(items):
        if not items[name]:
            continue
        slug = slugify(name)
        relative = f"{SENTENCES_DIR}/{language}/{FILE_PREFIX}{slug}.yaml"
        path = Path(hass.config.path(relative))
        content: dict[str, Any] = {
            "language": language,
            "intents": {name: {"data": items[name]}},
        }
        if declared := lists_by_intent.get(name):
            content["lists"] = declared
        if reply := replies.get(name):
            content["responses"] = {"intents": {name: {"default": reply}}}
        planned[path] = content
        files_by_intent[name] = relative

    # Write the tidy files, then remove the old ones. Same content means no write, so a
    # second run costs a read and nothing else.
    for path, content in planned.items():
        new_text = yaml_util.dump(content)
        if path.exists():
            try:
                if path.read_text(encoding="utf-8") == new_text:
                    continue
            except OSError:
                pass
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(new_text, encoding="utf-8")
        written.append(path.name)
    keep = {path.resolve() for path in planned}
    for path in sources:
        if path.resolve() in keep:
            continue
        try:
            path.unlink()
            removed.append(path.name)
        except OSError as err:  # pragma: no cover - filesystem trouble
            _LOGGER.error("Could not remove %s: %s", path, err)

    # The store's idea of where an intent lives has to follow the file.
    moved = 0
    for spec in data.intents:
        if (relative := files_by_intent.get(spec.name)) and spec.file != relative:
            spec.file = relative
            moved += 1
        # And what its sentences refer to has to match what the file declares, or the
        # template checker would allow a `{day}` the matcher no longer knows.
        if (declared := lists_by_intent.get(spec.name)) and spec.slots != declared:
            spec.slots = declared
            moved += 1
    for phrase in data.phrases:
        if (relative := files_by_intent.get(phrase.intent)) and phrase.file != relative:
            phrase.file = relative
    if moved:
        await async_save(hass, entry)

    if written or removed:
        _LOGGER.warning(
            "Irabelle Home Learning consolidated %d intent file(s) (%d written, %d removed): %s",
            len(written) + len(removed),
            len(written),
            len(removed),
            ", ".join(sorted(removed)) or "no files removed",
        )
    return {"written": sorted(written), "removed": sorted(removed)}


def backfill_responses(hass: HomeAssistant, entry: ConfigEntry) -> int:
    """Write every reply that is not yet in its sentence file, or not yet bound. Returns how many.

    One-off housekeeping: an intent learned before replies lived in the file has none, and
    the point of putting them there is that the file shows what the house will say. It is
    also the migration for the slot bindings (see `bind_slots`): a reply installed before
    them is compared as it would be written now, so an unbound one is rewritten once and an
    already-bound one is left alone.
    """
    written = 0
    for spec in entry.runtime_data.intents:
        path = Path(hass.config.path(spec.file))
        reply = spec.response or spec.template
        try:
            data = _load(path) if path.exists() else {}
        except Exception:  # noqa: BLE001 - a broken file is replaced below anyway
            data = {}
        stored = (
            ((data.get("responses") or {}).get("intents") or {}).get(spec.name) or {}
        ).get("default") or ""
        if stored == bind_slots(reply, _intent_slots(data, spec.name)):
            continue
        write_response(path, spec.name, reply, spec.template)
        written += 1
    return written


async def async_set_response(
    hass: HomeAssistant, entry: ConfigEntry, name: str, response: str
) -> dict[str, Any]:
    """Change how a learned intent answers, in the file and in the running handler.

    The reply is the part of a learned intent a person is most likely to want different -
    shorter, in their own words, with a time in it - and it is separate from the template
    that computes the answer, so changing it does not mean re-learning the intent.
    """
    data: LearnerData = entry.runtime_data
    spec = next((s for s in data.intents if s.name == name), None)
    if spec is None:
        raise LearnerError(f"{name!r} is not a learned intent")
    reply = " ".join((response or "").split())
    if not reply:
        raise LearnerError("a response cannot be empty")
    original, reply = reply, tidy_punctuation(reply)
    sample = slot_sample(spec.slots)
    _check_template(hass, reply, defined=set(spec.slots or {}), sample=sample)

    path = Path(hass.config.path(spec.file))
    write_response(path, name, reply, spec.template)
    spec.response = reply
    register_intent(hass, spec)
    await async_save(hass, entry)
    # What it renders now, not just what it says: a reply that reads an entity that is not
    # there renders "in the unknown", and the model that wrote it is the one that can fix
    # it - but only if the tool result shows it.
    warnings = missing_attributes(hass, reply)
    data.last_result = {
        "intent": name,
        "response": reply,
        "says": _render(hass, reply, sample)[:200],
        "warnings": warnings,
        "reads": _reads(hass, reply),
    }
    if reply != original:
        # Not silent: whoever wrote the text should know the house will not say a doubled
        # punctuation mark, and can see which one went.
        data.last_result["tidied"] = _tidy_note(original, reply)
    if warnings:
        _LOGGER.warning(
            "Irabelle Home Learning: the reply for %s %s", name, "; ".join(warnings)
        )
    async_dispatcher_send(hass, SIGNAL_UPDATED)
    _LOGGER.warning("Irabelle Home Learning changed the response for %s", name)
    return data.last_result


def replace_sentences(
    path: Path,
    intent: str,
    remove: set[str],
    add: list[str],
    lists: dict[str, Any] | None = None,
    pin: dict[str, Any] | None = None,
) -> list[str]:
    """Take named sentences out of an intent's file and put new ones in. Returns what remains.

    `_write_sentences` only ever adds, which is what learning needs and what makes calling it
    twice safe. Folding two phrasings into one has to *remove* the two, and this is the only
    place that does it. Matching is by the normalised form - the same comparison forgetting
    uses - so the sentence that goes is the sentence the matcher was using, question mark and
    all, and a group left with no sentences goes with them.

    ``pin`` is the `slots:` a group carries, and the added sentences go into a group with
    exactly those. It matters more than it looks: "what's the weather tomorrow" is pinned
    `day: tomorrow`, and a folded sentence that lands in an unpinned group still *matches* -
    and then answers about today, because nothing filled the slot.
    """
    data: dict[str, Any] = {"language": "en", "intents": {}}
    if path.exists():
        try:
            data = _load(path) or data
        except LearnerError:
            data = {"language": "en", "intents": {}}
    data.setdefault("language", "en")
    if lists:
        declared = data.setdefault("lists", {})
        for name, spec in lists.items():
            declared[name] = spec
    body = (data.get("intents") or {}).get(intent)
    if body is None:
        return []
    wanted = {normalize_sentence(sentence) for sentence in remove}
    blocks: list[dict[str, Any]] = []
    for item in body.get("data") or []:
        kept = [
            sentence
            for sentence in (item or {}).get("sentences") or []
            if normalize_sentence(str(sentence)) not in wanted
        ]
        if kept:
            blocks.append({**item, "sentences": kept})
    fresh: list[str] = []
    present = {
        normalize_sentence(str(sentence))
        for item in blocks
        for sentence in item.get("sentences") or []
    }
    for sentence in add:
        stored = sentence_for_storage(sentence)
        key = normalize_sentence(stored)
        if stored and key not in present:
            present.add(key)
            fresh.append(stored)
    if fresh:
        wanted = dict(pin or {})
        target = next(
            (item for item in blocks if dict((item or {}).get("slots") or {}) == wanted), None
        )
        if target is None:
            target = {"sentences": []}
            if wanted:
                target["slots"] = wanted
            blocks.append(target)
        target["sentences"] = list(target.get("sentences") or []) + fresh
    body["data"] = blocks
    data.setdefault("intents", {})[intent] = body
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml_util.dump(data), encoding="utf-8")
    return sorted(present)


def _write_sentences(
    path: Path,
    intent: str,
    utterances: list[str],
    lists: dict[str, Any] | None = None,
) -> None:
    """Merge sentences for a learned intent into its learner-owned file.

    ``lists`` is written beside them because that is how hassil resolves `{day}`: a `lists:`
    block belongs to the file it is in, so a sentence referring to a slot nothing defines
    stops matching for every agent. Keeping the two together is what makes a learned
    sentence self-contained.
    """
    data: dict[str, Any] = {"language": "en", "intents": {}}
    if path.exists():
        try:
            data = _load(path) or data
        except LearnerError:
            data = {"language": "en", "intents": {}}
    data.setdefault("language", "en")
    data.setdefault("intents", {})
    if lists:
        declared = data.setdefault("lists", {})
        for name, spec in lists.items():
            declared[name] = spec
    body = data["intents"].setdefault(intent, {"data": []})
    body.setdefault("data", [])
    present = {
        normalize_sentence(s)
        for item in body["data"]
        for s in (item or {}).get("sentences") or []
        if isinstance(s, str)
    }
    # `present` is updated as the list is built, not only read from the file: stripping
    # punctuation collapses two utterances onto one sentence ("is the thing on?" and "is the
    # thing on"), and comparing them all against the *file* would store the same sentence
    # twice in one write. The bind path has always done this; this one did not.
    fresh: list[str] = []
    for utterance in utterances:
        stored = sentence_for_storage(utterance)
        key = normalize_sentence(stored)
        if stored and key not in present:
            present.add(key)
            fresh.append(stored)
    if not fresh:
        return
    target = None
    for item in body["data"]:
        if not (item or {}).get("slots"):
            target = item
            break
    if target is None:
        target = {"sentences": []}
        body["data"].append(target)
    target.setdefault("sentences", []).extend(fresh)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml_util.dump(data), encoding="utf-8")


async def async_forget_intent(
    hass: HomeAssistant, entry: ConfigEntry, name: str
) -> dict[str, Any]:
    """Remove a learned intent: its handler, its sentences and its record."""
    data: LearnerData = entry.runtime_data
    spec = next((s for s in data.intents if s.name == name), None)
    if spec is None:
        raise LearnerError(f"{name!r} is not a learned intent")

    unregister_intent(hass, name)

    # Every file, not only the intent's own: a phrasing bound to it was written to
    # whichever file the binding went to, and a sentence whose intent no longer exists
    # fails the whole pipeline with "Unexpected error during intent recognition".
    removed = remove_intent_sentences(hass, language_of(entry), name)

    data.intents = [s for s in data.intents if s.name != name]
    data.phrases = [p for p in data.phrases if p.intent != name]
    await async_save(hass, entry)
    await hass.services.async_call("conversation", "reload", {}, blocking=True)
    data.last_result = {"forgotten_intent": name, "sentences_removed_from": removed}
    async_dispatcher_send(hass, SIGNAL_UPDATED)
    return data.last_result
