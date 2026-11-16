"""Turn observations into local answers, with the model off the hot path.

Two mechanisms, in order of confidence.

**Binding** - the turn ended by running a Home Assistant intent. That intent
already exists and already produced the right answer; the local matcher simply
did not recognise the words, so the words are bound to it immediately. This is a
fact about what just happened, not a judgement, so no model is involved at all.
Two kinds of turn arrive this way: a Jev-routed one, whose response carries the
``Intent`` that ran, and a model one that called an intent tool
(``assist__intent__HassTurnOn``).

**Generalisation** - the agent answered some other way: a state read, or plain
text. Whether that can be made local *is* a judgement, so a second model call
makes it - but only after the request has been heard ``min_occurrences`` times,
because a phrasing said twice is one that will be said again, and a one-off is
not worth a sentence file.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from types import SimpleNamespace
import re
import time
from dataclasses import dataclass, field
from string import Template
from typing import Any

from homeassistant.components.homeassistant import async_should_expose
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import intent as intent_helper
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.util import dt as dt_util

from .const import SIGNAL_UPDATED
from .intents import (
    async_set_response,
    async_forget_intent,
    async_learn_intent,
    async_update_template,
    missing_attributes,
    slot_sample,
)
from .learner import (
    LearnerData,
    LearnerError,
    Observation,
    ai_task_entity,
    async_intent_names,
    async_learn,
    async_save,
    sentence_index,
    async_forget,
    entity_aliases,
    entity_label,
    entity_owners,
    language_of,
    learned_phrases,
    min_occurrences,
    names_nothing,
    normalize_sentence,
    record_observation,
    template_entities,
)
from .models import async_generate_data
from .fileio import async_run
from .judge import Choice, JudgeError, Noul, async_ask, has_judge
from .upstream import reference
from .observer import Turn

_LOGGER = logging.getLogger(__name__)

#: Tool names the Assist API hands the model are namespaced: the intent handlers
#: arrive as ``assist__<owner>__<IntentName>``.
_INTENT_TOOL_PREFIX = "assist__"
_SLUG = re.compile(r"[^A-Za-z0-9_]")

#: Intents an assist pipeline refuses to answer locally whenever the agent
#: advertises CONTROL (Home Assistant's own `_async_local_fallback_intent_filter`).
#: Binding words to them produces a sentence the matcher is then told to ignore,
#: so the phrase comes back to the agent anyway - and the sentence file claims a
#: local answer that does not exist. A question like "how long has the tv been
#: on" resolves to HassGetState; it needs a *learned* intent with a template, not
#: a binding.
NOT_LOCAL_INTENTS: frozenset[str] = frozenset(
    {"HassGetState", "HassMediaSearchAndPlay"}
)

#: The assistant is asked to generalise; keep a lid on how often, so a chatty
#: afternoon cannot turn into a bill.
MAX_GENERALISATIONS_PER_HOUR = 20
_GENERALISED: dict[str, list[float]] = {}

GENERALIZE_INSTRUCTIONS = """\
A request reached a language model because this smart home's local sentence matcher
could not answer it. Decide whether the house can answer the same request locally, in
milliseconds and with no model involved, from now on.

THE REQUEST (the person's own words)
${utterance}

WHAT THE ASSISTANT DID
${tools}

WHAT IT ANSWERED
${answer}

WHAT THAT ANSWER ACTED ON
${targets}

INTENTS THAT ALREADY EXIST (bind to one of these, spelled exactly as written)
${intents}

ENTITIES EXPOSED TO ASSIST (what a *request* may name; one marked "X's" belongs to that
person, so a phrasing about "my" something must branch on who is asking before reading it.
A template may read any entity in this house, exposed or not - exposure decides what a
person can ask for, not what an answer can look at)
${entities}

THE ATTRIBUTES OF THE ENTITIES THIS REQUEST IS ABOUT
${attributes}

An attribute is read with state_attr('<entity_id>', '<attribute>'), and a nested value reads
straight from the shape above: a list is indexed and a dictionary's keys take a dot, so
`state_attr('weather.home', 'forecast_daily')[0].temperature` is one day's high. **Never
invent an attribute name** - read one from the list, and if the entity you want is not
there, answer from its state or choose "ignore" - and do not refuse a request merely because the
attribute it needs is not on the exposed list: check the attributes block first.

Choose one action.

- "bind": an existing intent above already does exactly this - the matcher simply did not
  recognise the words. Set "intent" to that intent's name. Leave "name" and "template"
  empty. Prefer this whenever it is even close: a phrasing that means the same thing in
  different words is a bind, not a new intent. Read the examples beside each intent.
  Never bind to HassGetState or MediaSearchAndPlay: an assist pipeline refuses to answer
  those locally, so the words would come straight back. A question that resolves to
  HassGetState ("how long has the tv been on") needs "create" instead.
- "create": no existing intent does this at all, but the answer is a fact about exposed
  entities that a Jinja template can render. Set "name" (CamelCase, letters and digits),
  "template" and "description". Leave "intent" empty. Never create an intent that
  duplicates one above.

ONE REQUEST, ONE ANSWER. If an installed intent answers the same question - in different
words, at a different length, on another day of the week - the answer is "bind". Two
intents for one question is how the house starts answering the same thing two different
ways depending on the words used, and the person asking cannot tell which they will get.
Create only when no installed intent answers the question at all. If the answer to an
installed intent is wrong or too short, that is a change to *that intent*, not a reason to
add a second one beside it; a duplicate create is refused and names the intent to bind to.

A phrasing that only swaps one word for another value in the same class - a weekday, a
room, a person, a colour - is ONE intent with a slot, never one intent per value and never
one hard-coded value. Declare the slot in "slots" as `day = today, tomorrow, monday,
tuesday, wednesday, thursday, friday, saturday, sunday` (list EVERY value the class can
take, including ones nobody has said yet - that is what makes the first mention of
Saturday work), write `{day}` where that word goes in the wording, and the template may
read `day`. A request that varies by weekday without a slot will only ever match the one
weekday it was taught with.

The wording may use Home Assistant's template syntax, which is how one line covers the ways
of asking that nobody has said yet: `[optional]` for a word that may be left out (`what's the
weather [like]`), `(this | that)` for a choice of words (`turn (on | off)`), `{slot}` for a
value class, and a bracketed ending for a part of a word (`light[s]`). Never write `,` `;` `?`
or `!` anywhere in a phrasing: the matcher treats them as separators inside the stored
sentence while Home Assistant strips them from what a person says, so a sentence containing
one matches nothing at all.

Everything else is fixed wording, so the template must name the entities it reads
directly - for "when was the office light last switched on", read light.office_light, do
not try to look up a light from a variable. A template that reads a variable nothing
defines (and no slot declares) renders a polite failure forever and will be refused.

- "ignore": everything else - a one-off, something that depends on a live value no
  template can read, something needing judgement or conversation, or anything you are not
  certain would be answered identically every time.

For a template: one short spoken sentence of Jinja over the exposed entities above,
using states(), state_attr() and is_state(). It is rendered every time the request is
made, so it must not depend on the moment it was learned. Format times, do not print
raw timestamps: {{ states.sensor.x.last_updated | as_timestamp |
timestamp_custom('%-I:%M %p') }}. Every entity also carries last_changed and
last_updated, so "when did X last switch on" IS answerable: use
{{ states.switch.x.last_changed | as_timestamp | timestamp_custom('%-I:%M %p') }}.
NEVER write state_attr(x, 'last_updated') or state_attr(x, 'last_changed'): those are on
the state, not in its attributes, so they render the literal word None and the house says
"at None" out loud. NEVER call strftime on them directly either: they are UTC, so a
5:22 PM event comes out as 5:22 AM. Both of those mistakes are refused. The variables user_name,
user_id, person_name, person_state, user_area, user_floor, device_id and satellite_id
are always defined, as are speaker_certain and speaker_confidence. From a voice satellite
user_name is empty, so read the recognised speaker with
states('sensor.voicebm_current_speaker') - but only when speaker_certain is set, and say you
are not sure who is asking otherwise. A voice match is a guess, and a guess answers the
wrong person just as confidently as the right one.

For utterances: the person's own words first, then at most three close variants including
likely speech-to-text slips. Every variant must be another way of asking the SAME request
with the SAME scope. Never add a variant that names one thing when the request was about
several, or the other way round: a question about all the TVs must not pick up "is the
bedroom TV on", because that phrasing asks about one TV and would then be answered with
all of them. No commas - a comma makes a stored sentence unmatchable. Phrasings that the
examples above show are already handled will be dropped.

For reason: one short sentence explaining the choice.${correction}"""

GENERALIZE_STRUCTURE = {
    "action": {
        "description": "bind, create or ignore",
        "required": True,
        "selector": {
            "select": {"options": ["bind", "create", "ignore"], "mode": "dropdown"}
        },
    },
    "intent": {
        "description": "for bind: the exact name of an existing intent",
        "selector": {"text": {}},
    },
    "name": {
        "description": "for create: a CamelCase intent name such as PhoneLastSeen",
        "selector": {"text": {}},
    },
    "description": {
        "description": "for create: what the intent answers",
        "selector": {"text": {}},
    },
    "template": {
        "description": "for create: the Jinja template that renders the answer",
        "selector": {"text": {"multiline": True}},
    },
    "utterances": {
        "description": "one phrasing per line, the person's own words first",
        "selector": {"text": {"multiline": True}},
    },
    "slots": {
        "description": (
            "for create, when the same request is asked with one word swapped: "
            "'slot = value, value, value' on its own line, e.g. 'day = today, tomorrow, "
            "monday, tuesday, wednesday, thursday, friday, saturday, sunday'. List every "
            "value the class can take, including ones nobody has said yet, then write "
            "{slot} where that word goes in each phrasing. Leave empty when the wording is "
            "fixed."
        ),
        "selector": {"text": {"multiline": True}},
    },
    "reason": {"description": "why this choice", "selector": {"text": {}}},
}


# --------------------------------------------------------------------------
# Mechanism 1: bind to the intent the assistant itself used
# --------------------------------------------------------------------------


def _platform_of_agent(hass: HomeAssistant, agent_id: str) -> str | None:
    """The integration domain behind a conversation agent id, for provenance."""
    if not agent_id:
        return None
    if (row := er.async_get(hass).async_get(agent_id)) is not None and row.platform:
        return row.platform
    return None


def _named_exposed_entities(hass: HomeAssistant, utterance: str) -> set[str]:
    """Exposed entities whose whole name the words appear to say.

    Multi-word names only, so a one-word friendly name like "TV" cannot claim
    every sentence that happens to contain the word.
    """
    words = set(normalize_sentence(utterance).split())
    found: set[str] = set()
    for state in hass.states.async_all():
        name_words = normalize_sentence(state.name or "").split()
        if len(name_words) < 2:
            continue
        if set(name_words) <= words and async_should_expose(
            hass, "conversation", state.entity_id
        ):
            found.add(state.entity_id)
    return found


def _binding_scope_problem(
    hass: HomeAssistant, entry: ConfigEntry, utterance: str, intent: str
) -> str | None:
    """Why binding these words to this learned intent would answer the wrong thing.

    Measured failure: asked "is the living room TV on", the generaliser argued in
    its own reason that binding to a bedroom-only intent "would answer about the
    wrong TV" - and then returned "bind" anyway. The words name an exposed entity;
    if the target's template reads only other entities, the answer cannot be about
    what was asked, whatever the model concluded.
    """
    spec = next((item for item in entry.runtime_data.intents if item.name == intent), None)
    if spec is None:
        # Not a learned template (a built-in or an intent_script): nothing to
        # compare the words against, so the model's judgement stands.
        return None
    named = _named_exposed_entities(hass, utterance)
    if not named:
        return None
    read = template_entities(spec.response or spec.template)
    if named & read:
        return None
    return (
        f"the words name {', '.join(sorted(named))} but {intent} reads only "
        f"{', '.join(sorted(read)) or 'no entity'}"
    )


async def intent_from_turn(
    hass: HomeAssistant, turn: Turn
) -> tuple[str, dict[str, Any]] | None:
    """Return the single intent this turn resolved to, if there is one.

    An agent that routed straight to an intent says so (Jev does); otherwise the
    intent is read out of the tool the model called.
    """
    known = await async_intent_names(hass)

    if turn.intent and turn.intent.get("name"):
        name = str(turn.intent["name"])
        if name in NOT_LOCAL_INTENTS:
            _LOGGER.debug(
                "Irabelle Home Learning will not bind to %s: the pipeline never answers it "
                "locally, so the words would come straight back",
                name,
            )
            return None
        if name in known:
            return name, dict(turn.intent.get("slots") or {})
        _LOGGER.debug("Irabelle Home Learning ignoring an unknown reported intent %r", name)
        return None

    tails: dict[str, str] = {}
    for name in known:
        tails[name] = name
        tails[_SLUG.sub("_", name)] = name

    matches: list[tuple[str, dict[str, Any]]] = []
    for call in turn.tool_calls:
        tool_name = call.get("name") or ""
        if not tool_name.startswith(_INTENT_TOOL_PREFIX):
            continue
        intent = tails.get(tool_name.rsplit("__", 1)[-1])
        if intent is not None and intent not in NOT_LOCAL_INTENTS:
            matches.append((intent, call.get("args") or {}))

    distinct = {intent for intent, _ in matches}
    if len(distinct) != 1:
        return None
    return distinct.pop(), matches[0][1]


#: Words too generic to count as evidence that a person named the thing they
#: meant: "the office lamp" and "the office light" share only "office", and that
#: is the word that makes the phrase mean something on its own.
_GENERIC_WORDS = frozenset(
    {
        "light", "lights", "lamp", "lamps", "switch", "switches", "power",
        "turn", "turned", "turning", "toggle", "please", "everything", "all",
    }
)
_WORD = re.compile(r"[a-z0-9]+")

#: Slots that only say what *kind* of thing is meant, or where the request came
#: from. They are safe to leave out of a learned sentence when something the
#: person actually named is pinned beside them ("kill the bedroom telly" needs
#: Bedroom TV, not "switch"); pinning the satellite's own area would even be
#: wrong, since the same words may later be spoken somewhere else.
_DROPPABLE_SLOTS = frozenset(
    {"domain", "device_class", "preferred_area_id", "preferred_floor_id"}
)


def _names_the_target(value: Any, utterance: str) -> bool:
    """Whether the person's own words name this slot value.

    A slot whose value the person never referred to was filled in by the model
    from context: "turn it off" becomes HassTurnOff on the office light because
    that is where the satellite is. Pinning that would make "turn it off" mean
    the office light forever. "Switch the office lamp on" is different - it says
    which thing it means, in words of its own.
    """
    text = str(value).lower()
    if text and text in utterance.lower():
        return True
    spoken = set(_WORD.findall(utterance.lower()))
    words = {
        word
        for word in _WORD.findall(text)
        if len(word) >= 4 and word not in _GENERIC_WORDS
    }
    return bool(words & spoken)


#: Built-in intents that act on or read a target. A sentence bound to one of these with no
#: target pinned matches and then has nothing to act on - an error read aloud, every time.
TARGETED_INTENTS = frozenset(
    {
        "HassTurnOn",
        "HassTurnOff",
        "HassToggle",
        "HassLightSet",
        "HassGetState",
        "HassSetPosition",
        "HassClimateSetTemperature",
        "HassFanSetSpeed",
        "HassVacuumStart",
        "HassVacuumReturnToBase",
        "HassLawnMowerStartMowing",
        "HassLawnMowerDock",
    }
)

#: How sure the judge must be that misheard words still name the target before it is pinned.
#: Wrong costs a phrasing that operates the wrong device, so this sits high.
HEARD_TARGET_CLEAN = 0.85


async def async_heard_target(
    hass: HomeAssistant, entry: ConfigEntry, utterance: str, args: dict[str, Any]
) -> dict[str, Any] | None:
    """The slots to pin when the words name the target only through a mishearing.

    `_names_the_target` looks for the target's own words, which is right for "turn it off"
    (it names nothing) and wrong for "toggle the D2 keyword" (it names the D2 keyboard,
    misheard). Telling those apart is a judgement, so the judge is asked. None when there is
    no judge, no target in the call, or the judge is not sure.
    """
    slots = {
        key: _unwrapped(raw)
        for key, raw in args.items()
        if key not in _DROPPABLE_SLOTS and _unwrapped(raw) not in (None, "", [])
    }
    if not slots or not has_judge(entry):
        return None
    target = ", ".join(
        ", ".join(map(str, value)) if isinstance(value, (list, tuple)) else str(value)
        for value in slots.values()
    )
    try:
        answers = await async_ask(
            hass,
            entry,
            {"request": utterance, "target": target},
            {
                "names_it": Noul(
                    "`request` is what speech-to-text heard a person say, and it may contain "
                    "mistakes. Does `request` itself name `target` - by its name, a part of its "
                    "name, or a misheard version of either? Answer false when the request "
                    "names nothing and the target could only have come from context, such as "
                    "'turn it off' or 'toggle that'.",
                    true="The request names the target, possibly misheard",
                    false="The request does not name the target",
                )
            },
        )
    except JudgeError as err:
        _LOGGER.debug("Irabelle Home Learning could not ask about %r: %s", utterance, err)
        return None
    score = answers["names_it"].noul or 0.0
    if score < HEARD_TARGET_CLEAN:
        return None
    _LOGGER.info(
        "Irabelle Home Learning pins %s for %r: the judge says the words name it (%.2f)",
        target,
        utterance,
        score,
    )
    return slots


def _unwrapped(value: Any) -> Any:
    """Strip the ``{"value": ...}`` envelope Home Assistant intent slots use.

    Jev hands us the slots exactly as ``intent.async_handle`` expects them, so
    they arrive wrapped; a model's tool arguments do not.
    """
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def _reproducible_slots(args: dict[str, Any], utterance: str) -> dict[str, Any] | None:
    """Return the slots to pin on this phrasing, or None to refuse.

    Two kinds of slot come out of the answering call, and both are worth keeping.

    **Named** - the person said it, so the sentence would not mean anything on its own
    without it: "the office light", "40 percent".

    **Read** - a word the model turned into a value it chose: *"to warm"* became
    `temperature: 2700`. The words do not contain the number, but they do contain the
    meaning, and this is the difference between learning the words and learning what they
    meant. Refusing these is what taught the house "set the office colour temperature to
    warm" with no temperature at all - the phrase then matched, changed nothing, and said
    nothing.

    What is still refused is a phrasing that *names* nothing: "turn it off", "make it
    nicer". A read value on its own would pin mere context to forever, and the learner's
    own value check refuses a bind that ends up with no value anyway.
    """
    named: dict[str, Any] = {}
    read: dict[str, Any] = {}
    for key, raw in args.items():
        value = _unwrapped(raw)
        if value is None or value == "" or value == []:
            continue
        values = list(value) if isinstance(value, (list, tuple)) else [value]
        if key in _DROPPABLE_SLOTS:
            # Context, not meaning: the area the sentence was spoken in, the domain a helper
            # narrowed to. It belongs to the moment, not to the words.
            _LOGGER.debug(
                "Irabelle Home Learning leaving %s=%r out of %r: the words do not name it",
                key,
                value,
                utterance,
            )
        elif any(_names_the_target(item, utterance) for item in values):
            named[key] = value
        else:
            read[key] = value

    if not named:
        # Nothing the person said identifies what they meant, so there is nothing to
        # anchor the reading to: "turn it off", "everything", a bare verb.
        _LOGGER.debug("Irabelle Home Learning refusing %r: nothing in it names a target", utterance)
        return None
    for key, value in read.items():
        _LOGGER.info(
            "Irabelle Home Learning pins %s=%r for %r: the answer read the words that way",
            key,
            value,
            utterance,
        )
    return named | read


# --------------------------------------------------------------------------
# Mechanism 2: generalise with a model, off the hot path
# --------------------------------------------------------------------------


def _ai_task_for(hass: HomeAssistant, entry: ConfigEntry) -> str | None:
    """Return the processing model: an `ai_task` entity or a conversation agent.

    Nothing configured falls back to the first `ai_task` entity, the one kind of model
    that exists for exactly this.
    """
    configured = ai_task_entity(entry)
    if configured:
        return configured
    states = hass.states.async_all("ai_task")
    return states[0].entity_id if states else None


def _exposed_entities(hass: HomeAssistant, limit: int = 150) -> str:
    """List what a template is allowed to read, whose it is, and what else it is called.

    The owner comes from the person's own trackers, and it is on the line because
    nothing else says it: the entities are named after hardware, so
    "sensor.bermuda_aa_bb_cc_dd_ee_ff_area_last_seen" is Alex's watch only if you
    already know that.

    Entity aliases are on the line for the same reason, and they are the better answer:
    an alias is how an entity keeps the name it came with and is still findable by the
    words a person says. A model that can see `Area Last Seen aka "Alex's watch"` can
    build a template for "where is Alex's watch" without anything being renamed.
    """
    owners = entity_owners(hass)
    lines = sorted(
        f"{state.entity_id} ({entity_label(hass, state.entity_id, state.name)}"
        + (f", {owners[state.entity_id]}'s)" if state.entity_id in owners else ")")
        for state in hass.states.async_all()
        if async_should_expose(hass, "conversation", state.entity_id)
    )
    return "\n".join(lines[:limit]) or "(none)"


#: Words that say nothing about which entity a request is about.
_ATTRIBUTE_STOP = frozenset(
    """the a an is are was were be been being what whats how when where who why do does
    did my your our their his her its this that these those it on off in of for and or
    to can could would should will shall tell show give get about with from me please
    now today tomorrow yesterday time state status""".split()
)

#: Attributes that are noise on every entity rather than information about it.
_BORING_ATTRIBUTES = frozenset(
    """friendly_name icon device_class unit_of_measurement supported_features
    attribution supported_color_modes effect_list preset_modes assumed_state
    restored entity_picture""".split()
)


def _short(value: Any, width: int = 26) -> str:
    """A value as a model should see it: one line, quoted, and cut rather than wrapped."""
    text = " ".join(str(value).split())
    return f"'{text}'" if len(text) <= width else f"'{text[:width]}...'"


def _attribute_shape(value: Any, max_items: int = 6) -> str:
    """The *shape* of an attribute value, with the first item's real values in it.

    A nested value is what a model gets wrong. `state_attr('weather.home', 'forecast_daily')`
    is a list of dictionaries, and knowing only the key names is not enough to write a
    template that finds tomorrow evening: the *format* of `datetime` decides whether you can
    match on it, and the difference between `templow` and `temperature` decides which one
    answers the question.

    So the first item is shown with its values - `[{datetime='2026-09-21T00:00:00+12:00',
    temperature=16.2, templow=11.8, condition='rainy'} x7]` - and the rest are counted. That
    is the useful middle between names alone and a dump: the whole thing, for the weather
    entity, is 48 hourly entries and several thousand tokens of state budget.
    """
    if isinstance(value, dict):
        shown = ", ".join(
            f"{key}={_short(value[key])}" for key in list(value)[:max_items]
        )
        return "{" + shown + ("..." if len(value) > max_items else "") + "}"
    if isinstance(value, (list, tuple)):
        if not value:
            return "[]"
        head = value[0]
        inner = (
            _attribute_shape(head, max_items)
            if isinstance(head, (dict, list, tuple))
            else _short(head)
        )
        return f"[{inner} x{len(value)}]"
    return ""


def _entity_attributes(hass: HomeAssistant, entity_id: str) -> str:
    """One entity's state and attributes, with nested values shown as shapes."""
    state = hass.states.get(entity_id)
    if state is None:
        return f"{entity_id}: no state"
    parts = [f"state={str(state.state)[:30]!r}"]
    for name in sorted(state.attributes):
        if name in _BORING_ATTRIBUTES:
            continue
        value = state.attributes[name]
        shape = _attribute_shape(value)
        parts.append(f"{name}={shape}" if shape else f"{name}={str(value)[:40]!r}")
        if len(parts) > 14:
            parts.append("...")
            break
    return f"{entity_id} ({state.name}): " + ", ".join(parts)


def _prompt_attributes(
    hass: HomeAssistant,
    utterance: str,
    template: str | None = None,
    limit: int = 8,
) -> str:
    """The attributes of the entities a request is about.

    Everything the house exposes would be thousands of lines, and the questions are about
    one or two things: the entities the request names, plus - when an existing template is
    being corrected - the ones that template already reads, since the entity a correction
    is about is usually named by the template rather than by the request.
    """
    words = [
        word
        for word in re.findall(r"[a-z0-9_]+", (utterance or "").lower())
        if len(word) > 2 and word not in _ATTRIBUTE_STOP
    ]
    wanted: list[str] = list(template_entities(template or ""))
    for state in hass.states.async_all():
        if len(wanted) >= limit:
            break
        if state.entity_id in wanted:
            continue
        # Attribute names count as a match: a request about "temperature" names no entity
        # in this house, but `weather.home` has a temperature attribute and that is the
        # entity the request is about.
        haystack = " ".join(
            [state.entity_id, state.name or "", *state.attributes.keys()]
        ).lower()
        if any(word in haystack for word in words):
            wanted.append(state.entity_id)
    block = "\n".join(
        _entity_attributes(hass, entity_id) for entity_id in wanted[:limit]
    ) or "(nothing in the request names an entity in this house)"
    # Logged because this is what the model is told, and a template that reads the wrong
    # attribute is almost always a prompt that did not show the right one.
    _LOGGER.debug("Irabelle Home Learning attributes for %r:\n%s", utterance, block)
    return block


def _intent_examples(hass: HomeAssistant, language: str) -> dict[str, list[str]]:
    """Every intent a phrase could bind to, with the wordings that reach it.

    The examples are the point. They are how a model recognises that "tell me
    which user I am" is the same request as the hand-written "who am I", so it
    binds instead of inventing a second intent that does the same job.
    """
    examples: dict[str, list[str]] = {}
    for sentence, intent in sentence_index(hass, language).items():
        examples.setdefault(intent, []).append(sentence)
    return examples


def _intent_descriptions(hass: HomeAssistant) -> dict[str, str]:
    """What each registered intent says it does, in Home Assistant's own words."""
    described: dict[str, str] = {}
    for handler in intent_helper.async_get(hass):
        described[handler.intent_type] = (
            getattr(handler, "description", None) or ""
        ).strip()
    return described


def _intent_catalogue(
    hass: HomeAssistant,
    language: str,
    intents: list[Any] | None = None,
    examples: dict[str, list[str]] | None = None,
) -> str:
    """Describe every intent a phrase could bind to, with example wordings.

    A *learned* intent also shows what it answers, because two intents that both answer
    the weather read identically on paper and completely differently out loud: the model
    choosing between "bind" and "create" can see that the house already says
    "Tomorrow's weather will be cloudy, high 19" and decide to bind rather than write a
    second answer to the same question.
    """
    if examples is None:
        examples = _intent_examples(hass, language)
    described = _intent_descriptions(hass)
    learned = {spec.name: spec for spec in (intents or [])}
    lines: list[str] = []
    for name in sorted(set(described) | set(examples) | set(learned)):
        line = f"- {name}"
        if described.get(name):
            line += f": {described[name]}"
        if samples := examples.get(name):
            shown = ", ".join(repr(text) for text in sorted(samples)[:4])
            line += f" (said as: {shown})"
        # The slots the handler reads, from Home Assistant's own sentence files for that
        # intent: a phrasing that says {level} where the handler reads {brightness} matches
        # and then hands it nothing, and no amount of prose in the prompt fixes that as
        # cheaply as showing the names.
        if takes := reference(language, name):
            line += f" (reads: {takes})"
        if (spec := learned.get(name)) is not None:
            line += f" (it answers: {_render(hass, spec.response or spec.template, slot_sample(spec.slots))[:160]!r})"
        lines.append(line)
    return "\n".join(lines) or "(none)"


def _describe_tools(turn: Turn, limit: int = 1200) -> str:
    """Say what the agent did, in a form the next model can reuse."""
    if not turn.tool_calls:
        return "(it answered without calling a tool)"
    parts: list[str] = []
    for call in turn.tool_calls:
        args = call.get("args") or {}
        rendered = ", ".join(f"{k}={v!r}" for k, v in args.items()) or "no arguments"
        parts.append(f"- {call.get('name')}({rendered})")
    text = "\n".join(parts)
    return text[:limit]


def _rate_limited(entry_id: str) -> bool:
    now = time.monotonic()
    recent = [stamp for stamp in _GENERALISED.get(entry_id, []) if now - stamp < 3600]
    _GENERALISED[entry_id] = recent
    if len(recent) >= MAX_GENERALISATIONS_PER_HOUR:
        return True
    recent.append(now)
    return False


async def _generalize(
    hass: HomeAssistant,
    entry: ConfigEntry,
    row: Observation,
    *,
    force: bool = False,
    correction: str | None = None,
) -> dict[str, Any] | None:
    """Ask a model whether this request can be made local, and act on the answer."""
    entity_id = _ai_task_for(hass, entry)
    if entity_id is None:
        row.note = "no ai_task entity to generalise with"
        await async_save(hass, entry)
        return None
    if not force and _rate_limited(entry.entry_id):
        row.note = "generalisation rate limit reached"
        await async_save(hass, entry)
        return None

    known = sorted(await async_intent_names(hass))
    instructions = Template(GENERALIZE_INSTRUCTIONS).substitute(
        utterance=row.utterance,
        tools=_describe_tools(
            Turn(
                utterance=row.utterance,
                agent_id=row.agent_id,
                tool_calls=row.tool_calls
                or [{"name": name, "args": {}} for name in row.tools],
                answer=row.answer,
                targets=list(row.targets),
            )
        ),
        answer=(row.answer or "(nothing recorded)")[:1200],
        targets=", ".join(row.targets[:6]) or "(nothing recorded)",
        intents=_intent_catalogue(
            hass,
            language_of(entry),
            entry.runtime_data.intents,
            examples=await async_run(hass, _intent_examples, hass, language_of(entry)),
        ),
        entities=_exposed_entities(hass),
        attributes=_prompt_attributes(hass, row.utterance),
        correction=f"\n\nCORRECTION: {correction}" if correction else "",
    )

    try:
        response = await async_generate_data(
            hass,
            {
                "task_name": "learn a local intent",
                "entity_id": entity_id,
                "instructions": instructions,
                "structure": GENERALIZE_STRUCTURE,
            },
        )
    except Exception as err:  # noqa: BLE001 - a failed pass must not break anything
        row.note = f"generalisation failed: {err}"
        await async_save(hass, entry)
        _LOGGER.warning("Irabelle Home Learning generalisation failed: %s", err)
        return None

    payload = (response or {}).get("data")
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except ValueError:
            payload = None
    if not isinstance(payload, dict):
        row.note = "generalisation returned nothing usable"
        await async_save(hass, entry)
        return None

    action = str(payload.get("action") or "ignore").strip().lower()
    reason = str(payload.get("reason") or "").strip()
    _LOGGER.debug("Irabelle Home Learning generalised %r -> %s (%s)", row.utterance, action, reason)

    if action == "bind":
        intent = str(payload.get("intent") or "").strip()
        if intent in NOT_LOCAL_INTENTS:
            refusal = (
                f"{intent} is never answered locally, so the words would come "
                "straight back"
            )
        elif intent not in set(known):
            refusal = f"there is no intent called {intent!r}"
        else:
            refusal = _binding_scope_problem(hass, entry, row.utterance, intent)

        if refusal is None:
            return await _bind(hass, entry, row, intent, {})

        if correction is None:
            # The model contradicted itself: it argued in "reason" that binding
            # here would answer about the wrong thing, then returned "bind". Give
            # it its own argument back once rather than accepting a wrong answer.
            _LOGGER.info(
                "Irabelle Home Learning rejected binding %r to %s (%s); asking once more",
                row.utterance,
                intent,
                refusal,
            )
            return await _generalize(
                hass,
                entry,
                row,
                force=True,
                correction=(
                    f"You chose action \"bind\" to {intent}, but {refusal}. Choose "
                    "\"create\" with a template over the entity the words actually "
                    'name, or "ignore".'
                ),
            )
        row.state = "ignored"
        row.note = f"would have bound to {intent}, but {refusal}"
        await async_save(hass, entry)
        return {"action": "bind", "ok": False, "error": refusal}

    if action == "create":
        name = str(payload.get("name") or "").strip()
        template = str(payload.get("template") or "").strip()
        utterances = _utterance_lines(payload.get("utterances"), row.utterance)
        if not name or not template:
            row.note = "asked to create an intent but gave no name or template"
            await async_save(hass, entry)
            return None
        try:
            result = await async_learn_intent(
                hass,
                entry,
                name=name,
                template=template,
                utterances=utterances,
                description=str(payload.get("description") or "").strip() or None,
                note="generalised from conversation",
                source=row.agent_id or "conversation",
                slots=_slot_lines(payload.get("slots")) or None,
            )
        except LearnerError as err:
            row.state = "ignored"
            row.note = f"create refused: {err}"
            await async_save(hass, entry)
            if correction is None:
                # A refusal is the one piece of feedback this model cannot see, and it is
                # the useful kind: "the slot day has 1 value" is exactly what makes the
                # next attempt list all seven weekdays. Ask once, with its own mistake.
                _LOGGER.info(
                    "Irabelle Home Learning refused creating %s for %r (%s); asking once more",
                    name,
                    row.utterance,
                    err,
                )
                return await _generalize(
                    hass,
                    entry,
                    row,
                    force=True,
                    correction=f'The intent you proposed was refused: {err}.',
                )
            return {"action": "create", "ok": False, "error": str(err)}
        row.state = "learned"
        row.intent = name
        row.note = reason or "generalised into a new intent"
        await async_save(hass, entry)

        answer: dict[str, Any] = {"action": "create", "ok": True, "intent": name, "result": result}
        # A new intent arrives with its own list of phrasings, and a phrasing that
        # asks something narrower or wider than the template answers is the way
        # this goes wrong - the template can be perfectly faithful to the request
        # that produced it and still be claimed for a different one.
        if (review := await async_review_learning(hass, entry, name, utterance=row.utterance, answer=row.answer, row=row, replace=correction is None)) is not None:
            answer["review"] = review
        return answer

    row.state = "ignored"
    row.note = reason or "not worth answering locally"
    await async_save(hass, entry)
    return {"action": "ignore", "reason": row.note}


def _utterance_lines(value: Any, fallback: str) -> list[str]:
    """Split the model's phrasings into lines, always keeping the real one."""
    lines = [line.strip(" -•\t") for line in str(value or "").splitlines()]
    fresh = [line for line in lines if line]
    if fallback not in fresh:
        fresh.insert(0, fallback)
    return fresh


def _slot_lines(value: Any) -> dict[str, list[str]]:
    """Parse the model's `slot = value, value` lines into what `normalise_slots` wants.

    A plain text field rather than a structured one: the ai_task schema hands the model
    one flat object, and a nested map came back as prose the first time it was asked for.
    One line per slot is also what a person would write, and an unparseable line simply
    produces no slot - `normalise_slots` then refuses the template that reads it, which is
    the same refusal as never having declared it.
    """
    slots: dict[str, list[str]] = {}
    for line in str(value or "").splitlines():
        line = line.strip(" -•\t")
        if not line or ("=" not in line and ":" not in line):
            continue
        separator = "=" if "=" in line else ":"
        name, _, rest = line.partition(separator)
        name = name.strip().strip("\"'").lower().replace(" ", "_")
        if not name or any(char in rest for char in "{}[]"):
            continue
        values = [part.strip(" \"'") for part in rest.split(",")]
        values = [part for part in values if part]
        if values:
            known = slots.setdefault(name, [])
            for value in values:
                if value.lower() not in {item.lower() for item in known}:
                    known.append(value)
    return slots


REVIEW_INSTRUCTIONS = """\
The house just learned to answer something locally, with no model. A wrong local answer
is worse than no local answer: it is instant, confident and permanent. Check this
learning before it stands.

${context}WHAT THE HOUSE WILL NOW SAY, WITH NO MODEL
intent: ${intent}
template: ${template}
which renders, right now, as:
${rendered}
Once per person and once with nobody speaking. A template that branches on who is asking
shows its fallback in the "nobody speaking" line, because `user_name` is unset and
`sensor.voicebm_current_speaker` reads `unknown` - that is the fallback working, not a
fault, as long as the line for a person is right.
THE ENTITIES THAT TEMPLATE READS, RIGHT NOW
${reads}
Every one of those is a real entity with the state shown. They are not on the exposed
list below, and that is not a fault - exposure only decides what a *request* may name, not
what a template may read. Never report one of them as missing, nonexistent or not real.
It is now ${now} where the house is, so a time in an answer should be plausible against
that - a time of day on the wrong side of it is a formatting mistake, not a stale fact.

EVERY PHRASING THAT INTENT NOW ANSWERS
${phrasings}

ENTITIES EXPOSED TO ASSIST
${entities}

THE ATTRIBUTES OF THE ENTITIES THIS REQUEST IS ABOUT
${attributes}

An attribute is read with state_attr('<entity_id>', '<attribute>'). A nested value reads
straight from the shape above: a list is indexed and a dictionary's keys take a dot, so
`state_attr('weather.home', 'forecast_daily')[0].temperature` is one day's high. The shapes
are what the values look like right now - a list of dictionaries shows the keys of its
first item, which is what a template needs and what a guess gets wrong. **Never invent an
attribute name**: read one from the list above, and if the entity you want is not there,
answer from its state or say you cannot.
A template may read any entity in the house, whether or not it is on that list.

Check each phrasing separately. It is right only if that wording is answered correctly by
that template - same subject, same scope. These are the mistakes to look for:

- a phrasing naming ONE device answered with several, or with a DIFFERENT device
  ("is the bedroom TV on" answered with the living room's, or with both);
- a phrasing naming SEVERAL devices answered with only one;
- a phrasing about one property answered with another (asked WHEN, answered WHERE);
- a template that would answer a different question if the phrasing changed.

A phrase that asks about "the TVs" with no room named, answered by a template that reports
every TV, is CORRECT - keep it. Naming each room in the answer is not over-answering when
the question did not name a room. Do not remove a phrasing for being broader than the
template; remove one only for being NARROWER, or for naming a different thing.
- an answer that names a value as unknown, unavailable or None ("last seen in the
  unknown"), which means the template reads an entity that is missing or has no usable
  state - that is always a fix, and the corrected template must read a real entity from
  the exposed list;
- an answer that is the same for everyone when the phrasing is about the speaker ("where
  is my watch", "when was my phone last seen" answered identically for every person). If
  the template reads an entity whose name belongs to one person ("Alex's Watch") and
  never looks at who is asking, then it answers the wrong person half the time: that is a
  fix, and the corrected template must branch on the speaker, e.g. {% set who =
  (person_name or user_name or states('sensor.voicebm_current_speaker')) | lower %}. But
  when the template *does* branch on the speaker and a phrasing names a specific device
  instead ("where is my one plus" - Sam's phone, not whoever asks), the phrasing is
  the thing that is wrong: put it in remove_phrasings, and do not rewrite the template to
  ignore who is asking.

Reply "ok" only if every phrasing above is answered correctly. Otherwise "fix", and give:
- "remove_phrasings": exactly the phrasings, copied from the list above, that this
  template answers wrongly. One per line. Leave empty if they are all fine and only the
  template is wrong.
- "template": a corrected Jinja template, if and only if the template itself is what is
  wrong. Same rules as always: one short spoken sentence over exposed entities; never
  state_attr(x, 'last_updated') or state_attr(x, 'last_changed'); never read a variable
  nothing defines. Leave empty when the phrasings are the problem.
- "problem": one sentence saying what is wrong.

A phrasing that names a device or a room the template does not answer is answered wrongly,
and there is usually no correcting the template to fix it: a device that does not exist
cannot be read. Put that phrasing in remove_phrasings.

"fix" with an empty remove_phrasings AND an empty template changes nothing, and the wrong
answer stands. If you cannot name one of the two, answer "ok" - a fault you cannot act on
is worse than no verdict, because it hides the answer that is actually being given.
Removing a phrasing is not destructive: the request simply goes to the agent again and can
be learned properly another time. Keeping a wrong answer is destructive."""

REVIEW_STRUCTURE = {
    "verdict": {
        "description": "ok, or fix",
        "required": True,
        "selector": {"select": {"options": ["ok", "fix"], "mode": "dropdown"}},
    },
    "remove_phrasings": {
        "description": "phrasings this template answers wrongly, one per line",
        "selector": {"text": {"multiline": True}},
    },
    "template": {
        "description": "a corrected template, only if the template itself is wrong",
        "selector": {"text": {"multiline": True}},
    },
    "problem": {"description": "what is wrong, in one sentence", "selector": {"text": {}}},
}

#: Reviews cost a model call per learned intent, so they have their own budget.
MAX_REVIEWS_PER_HOUR = 30
_REVIEWED: dict[str, list[float]] = {}


def _review_limited(entry_id: str) -> bool:
    now = time.monotonic()
    recent = [stamp for stamp in _REVIEWED.get(entry_id, []) if now - stamp < 3600]
    _REVIEWED[entry_id] = recent
    if len(recent) >= MAX_REVIEWS_PER_HOUR:
        return True
    recent.append(now)
    return False


def _render(
    hass: HomeAssistant, template: str, variables: dict[str, Any] | None = None
) -> str:
    """What a template says right now, so the reviewer can read the real answer."""
    try:
        from homeassistant.helpers import template as template_helper  # noqa: PLC0415

        return str(
            template_helper.Template(template, hass).async_render(
                {"slots": {}, **(variables or {})}, parse_result=False
            )
        ).strip()
    except Exception:  # noqa: BLE001 - a render failure is the reviewer's business
        return "(it failed to render)"


def _render_for_people(
    hass: HomeAssistant, template: str, sample: dict[str, str] | None = None
) -> dict[str, str]:
    """What a template says for each person the house knows, and for nobody speaking.

    A template that branches on who is asking shows its *fallback* when nobody is:
    `user_name` is unset and `sensor.voicebm_current_speaker` reads `unknown`. Judging
    that render is how a speaker-aware intent gets flagged for answering the wrong
    person while it is in fact correct - `PhoneLastSeen` came back with
    `answers_the_request=0.04` and `what_is_wrong=not_for_the_speaker` for a template
    that names the right phone for Sam and for Alex. So it is shown once per
    person, and the questions say that the nobody-speaking render is allowed to be the
    fallback.

    ``sample`` fills each declared slot with one of its values, so a template that reads a
    slot is judged on a real answer instead of on the blanks an undefined name renders as.
    """
    base = dict(sample or {})
    rendered: dict[str, str] = {
        "(nobody speaking)": _render(hass, template, {**base, "slots": base})
    }
    for state in sorted(hass.states.async_all("person"), key=lambda item: item.entity_id):
        name = state.attributes.get("friendly_name") or state.name
        rendered[str(name)] = _render(
            hass,
            template,
            {
                **base,
                "user_name": name,
                "person_name": name,
                "person_entity_id": state.entity_id,
                "person_state": state.state,
            },
        )
    return rendered


#: What the screen asks, and the probabilities that make it clean. These are starting
#: points to be replaced by measured accuracy, not truths: the docs say calibration is
#: group-level and that thresholds belong to the domain.
#:
#: 0.60, not the docs' 0.80. Measured over the installed intents, correct learnings land
#: between 0.68 and 0.98 on the individual Nouls once a real template is involved - long
#: Jinja with per-person branches is harder to be certain about than the toy cases the
#: band was written for - while the wrong ones that mattered came in at 0.26, 0.52 and
#: below. A veto at 0.80 escalated `TvStatus`, `PhoneLastSeen` and `WatchLastSeen` to the
#: text reviewer on every audit, and the reviewer said "ok" every time. The Choice still
#: vetoes on its own, so "something is wrong" does not need a low Noul to be noticed.
SCREEN_CLEAN = 0.60
SCREEN_MARGIN = 0.50

SCREEN_QUESTIONS_DOC = """One request, seven snap judgements, in front of the expensive reviewer."""


@dataclass(slots=True)
class Screen:
    """What the cheap judge made of a learning."""

    answers: dict[str, Any] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    checked: bool = False

    @property
    def clean(self) -> bool:
        return self.checked and not self.reasons

    def as_dict(self) -> dict[str, Any]:
        return {
            "checked": self.checked,
            "clean": self.clean,
            "reasons": self.reasons,
            "answers": self.answers,
        }


def _rendered_variants(hass: HomeAssistant, spec: Any) -> dict[str, str]:
    """What a template says per person, once with its slots filled and once without.

    A slotted intent runs two ways: the phrasings that carry the slot ("where is Alex's
    phone") and the ones that do not ("where is my phone", which branches on the speaker).
    Rendering only the filled version made every person's answer the sample person's -
    "Sam's iPhone was last seen..." for Alex too - and a screen shown only that
    correctly reported the intent as answering the wrong person, for a template that is
    right. Both paths are shown, labelled, so the judgement is made on what actually
    happens.
    """
    sample = slot_sample(getattr(spec, "slots", None))
    # What the house says: a learned intent's reply is the text it renders, and the field named
    # `template` can be an older wording the store kept from before someone asked for another.
    seen = spec.response or spec.template
    if not sample:
        return _render_for_people(hass, seen)
    out: dict[str, str] = {}
    without = _render_for_people(hass, seen)
    out.update({f"with no slot given - {who}": says for who, says in without.items()})
    shown = ", ".join(f"{name}={value}" for name, value in sample.items())
    out.update(
        {
            f"with {shown} - {who}": says
            for who, says in _render_for_people(hass, seen, sample).items()
        }
    )
    return out


def _screen_state(
    hass: HomeAssistant,
    entry: ConfigEntry,
    row: Observation | None,
    spec: Any,
    *,
    utterance: str | None = None,
    answer: str = "",
    phrasings: list[str] | None = None,
) -> dict[str, Any]:
    """The state the screen reads: what was asked, what was learned, what it now says.

    Audit has no observation behind it, so the request falls back to the phrasings the
    intent answers - asking an audit "does this answer '(auditing an installed intent)'"
    is how a correct learning got flagged too_wide.

    Only the entities the template actually reads are included. The whole exposed
    catalogue used to go in here, and a hundred and fifty unrelated entity ids around one
    small judgement is noise: Jev answered 0.43 for "does this answer the request" on a
    template that answered it exactly.
    """
    if phrasings is None:
        phrasings = learned_phrases(hass, language_of(entry)).get(spec.name, [])
    if row is not None:
        request, replied = row.utterance, row.answer
    else:
        request = utterance or "; ".join(phrasings) or "(no phrasing recorded for this intent)"
        replied = answer
    seen = spec.response or spec.template
    return {
        "request": request,
        "what_the_agent_answered_at_the_time": (replied or "(nothing recorded)")[:600],
        "the_template_that_was_learned": seen,
        "the_slots_the_phrasings_use": getattr(spec, "slots", None) or "(none: fixed wording)",
        "what_the_template_says": _rendered_variants(hass, spec),
        "the_phrasings_this_intent_now_answers": phrasings,
        "the_entities_this_template_reads": _screen_entities(hass, seen),
        "the_real_entities_that_could_answer_this": _candidate_entities(hass, request, seen)
        or "(no other entity in this house matches the request)",
        "the_time_where_the_house_is": dt_util.now().strftime("%A %-I:%M %p %Z"),
    }


def _candidate_entities(
    hass: HomeAssistant, request: str, template: str, limit: int = 12
) -> dict[str, str]:
    """Other real entities whose name matches the request, and what they read right now.

    The screen could already see the entities a template reads; what it could not see is the
    alternative. A model that read `sensor.home_uv_index` when
    `sensor.openuv_current_uv_index` is the one that reports looks perfect in isolation -
    the entity exists, it renders a value - and the only way to notice the wrong one is to
    put the *other* entities that fit the request beside it. Matched on the request's words
    against the id, the friendly name and the aliases, which is the search a person would do.
    """
    words = {
        word
        for word in re.split(r"[^a-z0-9]+", normalize_sentence(request))
        if len(word) > 2
    }
    reads = set(template_entities(template))
    scored: list[tuple[int, str]] = []
    for state in hass.states.async_all():
        if state.entity_id in reads:
            continue
        haystack = " ".join(
            [state.entity_id, state.name or "", *entity_aliases(hass, state.entity_id)]
        ).lower()
        score = sum(1 for word in words if word in haystack)
        if score:
            scored.append((score, state.entity_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    found: dict[str, str] = {}
    for _, entity_id in scored[:limit]:
        state = hass.states.get(entity_id)
        if state is None:
            continue
        found[entity_id] = f"{entity_label(hass, entity_id, state.name)} = {state.state}"
    return found


def _screen_entities(hass: HomeAssistant, template: str) -> dict[str, str]:
    """Every entity a template reads, with the state it reads right now.

    Absent and "not reporting right now" are kept apart, because only one of them is the
    template's fault. A tracker that is out of range, a light that is offline, a sensor
    that has not reported since boot - all read `unknown`, and in a real house that is
    most of the time for an area sensor. Treating that as a missing entity flagged every
    watch and phone intent in the house on every audit, so the question only asks
    whether the entity is *there*.
    """
    found: dict[str, str] = {}
    for entity_id in template_entities(template):
        state = hass.states.get(entity_id)
        if state is None:
            found[entity_id] = "(not in this house)"
            continue
        label = entity_label(hass, entity_id, state.name)
        if state.state in ("unknown", "unavailable"):
            found[entity_id] = f"{label} - {state.state} (present, not reporting right now)"
            continue
        # The attribute names are on the line because a template often reads one of them
        # rather than the state: `state_attr('weather.home', 'temperature')` answers a
        # question about temperature, and a screen shown only `weather.home = rainy` judges
        # it a rain answer and prefers whichever entity is *named* temperature.
        attributes = ", ".join(sorted(state.attributes)[:10])
        found[entity_id] = (
            f"{label} = {state.state}"
            + (f" (attributes: {attributes})" if attributes else "")
        )
    return found or {"(none)": "the template reads nothing"}


async def async_screen_learning(
    hass: HomeAssistant,
    entry: ConfigEntry,
    row: Observation | None,
    spec: Any,
    *,
    utterance: str | None = None,
    answer: str = "",
) -> Screen | None:
    """Ask the classifier whether a learning is sound, before spending a text model on it.

    The reviewer that follows can only afford to run on learnings involving a learned
    intent, so bindings to built-in intents have never been checked at all - and that
    is where several of this component's worst answers came from. This screen is a
    single request costing about a hundredth of a cent, so it can run on everything.

    Returns None when no classification model is configured or it cannot be reached; the caller
    then falls through to the text reviewer exactly as before. A screen must never be
    the reason a learning goes unchecked.
    """
    if not has_judge(entry):
        return None
    learned = await async_run(hass, learned_phrases, hass, language_of(entry))
    state = _screen_state(
        hass,
        entry,
        row,
        spec,
        utterance=utterance,
        answer=answer,
        phrasings=learned.get(spec.name, []),
    )
    _LOGGER.debug("Irabelle Home Learning screen state for %s: %s", spec.name, json.dumps(state))
    try:
        answers = await async_ask(
            hass,
            entry,
            state,
            {
                "answers_the_request": Noul(
                    "`what_the_template_says` is what `the_template_that_was_learned` "
                    "answers, once for each person who lives here and once with nobody "
                    "speaking. A template that branches on who is asking shows its "
                    "fallback with nobody speaking, which is not by itself wrong. Does "
                    "the answer it gives to a person answer `request`?",
                    true="It answers what was asked, for whoever is asking",
                    false="It answers a different question, or a different thing",
                ),
                "same_scope": Noul(
                    "`the_phrasings_this_intent_now_answers` are all matched to this one "
                    "intent and answered by `the_template_that_was_learned`. Is every one "
                    "of them answered correctly by that template? A phrasing that names a "
                    "different device, a different room, or a different property than the "
                    "template gives is NOT answered correctly, however similar it sounds.",
                    true="Every phrasing is answered correctly by that template",
                    false=(
                        "At least one phrasing would be answered with the wrong device, "
                        "room or property"
                    ),
                ),
                "reads_a_real_entity": Noul(
                    "`the_entities_this_template_reads` is every entity "
                    "`the_template_that_was_learned` reads, with the state it reads right "
                    "now. Does every one of them exist in this house? `(not in this "
                    "house)` marks one that does not. A state of unknown or unavailable "
                    "means the entity is there and simply is not reporting at this "
                    "moment - that is normal and is not a fault.",
                    true="Every entity it reads exists in this house",
                    false="At least one is not in this house, so the answer is invented",
                ),
                "reads_the_right_entity": Noul(
                    "`the_template_that_was_learned` reads `the_entities_this_template_reads`, "
                    "each with its attributes. `the_real_entities_that_could_answer_this` "
                    "lists other entities in this house whose name matches `request`. Is "
                    "every entity the template reads a sensible source for `request` - the "
                    "same KIND of thing that was asked about? Two entities that hold the "
                    "same quantity are BOTH fine (a weather entity's temperature attribute "
                    "and a temperature sensor are both the temperature), so reading one while "
                    "the other is listed is not a fault. Reading something about a different "
                    "thing entirely - a light, a lock, another person's device - is.",
                    true="Every entity it reads is about the kind of thing that was asked",
                    false="It reads an entity that is about something else entirely",
                ),
                "answers_the_speaker": Noul(
                    "`what_the_template_says` is the same template answered once per "
                    "person who lives here. If a phrasing in "
                    "`the_phrasings_this_intent_now_answers` asks about \"my\" OBJECT - a "
                    "phone, a watch, keys, headphones, something that belongs to one person - "
                    "every person should get their own answer, and the lines should not be "
                    "identical. A living thing is not an object: \"my dog\" is the house's "
                    "dog and one answer for everyone is correct. Answer true when no phrasing "
                    "is about the speaker, or when the ones that are are about a living thing, "
                    "or when the people get different answers.",
                    true="No phrasing is about the speaker, or the people get different answers",
                    false="A phrasing is about the speaker's own object and every person gets the same answer",
                ),
                "time_looks_right": Noul(
                    "Is every time of day in `what_the_template_says` plausible for "
                    "`request`, given `the_time_where_the_house_is`? A time twelve hours "
                    "out - AM where it should be PM - is not.",
                    true="No time is present, or every time is plausible",
                    false="A time is wrong, missing, or reads as a placeholder",
                ),
                "reads_like_a_sentence": Noul(
                    "`what_the_template_says` is what the house will say out loud. Does "
                    "it read like one sentence a person would say? Every word in place, "
                    "nothing missing or repeated, nothing that names a thing instead of "
                    "answering about it (\"your Amazfit GTS Area Last Seen was last seen\" "
                    "names the sensor twice over), and properly a sentence: it starts "
                    "with a capital and ends with punctuation.",
                    true="It reads like something a person would say out loud",
                    false="A word is missing, repeated or out of place, or it reads as two half-sentences",
                ),
                "what_is_wrong": Choice(
                    "What, if anything, is wrong with this learning?",
                    {
                        "correct": "Every phrasing above is answered correctly by that template",
                        "too_narrow": "A phrasing is answered too broadly for what it asks",
                        "too_wide": "A phrasing is answered with less than it asks for",
                        "different_device": "A phrasing names one thing and is answered with another",
                        "wrong_property": "A phrasing asks one thing and is answered with another (asked WHEN, answered WHERE)",
                        "missing_state": "The template reads something that is not there",
                        "not_for_the_speaker": "The answer is the same for everyone when it is about the speaker",
                        "none_of_these": "Something else is wrong",
                    },
                ),
            },
        )
    except JudgeError as err:
        _LOGGER.debug("Irabelle Home Learning screen unavailable for %s: %s", spec.name, err)
        return None
    except Exception as err:  # noqa: BLE001 - a screen must never block a learning
        _LOGGER.warning("Irabelle Home Learning screen failed for %s: %s", spec.name, err)
        return None

    screen = Screen(
        answers={
            name: {
                "noul": answer.noul,
                "choice": answer.choice,
                "confidence": answer.confidence,
                "margin": answer.margin(),
            }
            for name, answer in answers.items()
        },
        checked=True,
    )

    for name, threshold in (
        ("answers_the_request", SCREEN_CLEAN),
        ("same_scope", SCREEN_CLEAN),
        ("reads_a_real_entity", SCREEN_CLEAN),
        ("reads_the_right_entity", SCREEN_CLEAN),
        ("answers_the_speaker", SCREEN_CLEAN),
        ("time_looks_right", SCREEN_CLEAN),
        ("reads_like_a_sentence", SCREEN_CLEAN),
    ):
        value = answers[name].noul
        if value is None or value < threshold:
            screen.reasons.append(f"{name}={value}")

    what = answers["what_is_wrong"]
    if what.choice != "correct":
        screen.reasons.append(f"what_is_wrong={what.choice}")
    elif (what.margin() or 0.0) < SCREEN_MARGIN:
        # A near-tie on "correct" is not a verdict. It is the signal that sent the
        # learner the wrong way twice: a phrasing that fitted two intents, and a
        # plural question that fitted a singular template.
        screen.reasons.append(f"what_is_wrong is a near-tie (margin {what.margin()})")

    if screen.clean:
        _LOGGER.debug("Irabelle Home Learning screen passed %s", spec.name)
    else:
        _LOGGER.info(
            "Irabelle Home Learning screen flagged %s: %s", spec.name, ", ".join(screen.reasons)
        )
    return screen


# --------------------------------------------------------------------------
# Routing: which intent, if any, does this request already have?
# --------------------------------------------------------------------------


#: What a routing answer has to reach before it is acted on. As with the screen, these
#: are starting points to be replaced by measured accuracy.
ROUTE_CONFIDENCE = 0.80
ROUTE_MARGIN = 0.50
ROUTE_CLEAN = 0.80

#: How sure the judge has to be that a template could answer something before the text
#: model is asked to write one. **Lower than the other bars on purpose**, because the two
#: mistakes are not symmetrical: a wrong *create* costs one model call and a template that
#: the screen then judges, while a wrong *ignore* sends the request to a model every time
#: it is said, for ever. "Is the living room light off?" came back computable=0.77 and was
#: refused at 0.80 - four times, after being ignored once already.
COMPUTABLE_ENOUGH = 0.50

#: A compound sentence is a *statement*, not a shrug: "whats the current state of the
#: living room lights" came back with one_request=0.66 and was read as two requests at the
#: 0.80 bar, which ignored it eleven times. Only a confident "this is two things" counts.
COMPOUND_ABOVE = 0.50

#: What a *bind* needs to be acted on the first time a phrase is heard. Stricter than the
#: usual bar, because the usual bar is backed by a second sighting and this is not.
ROUTE_CERTAIN = 0.95
ROUTE_CERTAIN_MARGIN = 0.90

#: How many times a phrase the house already answers locally must come back to the agent
#: before the intent behind it is looked at again.
REPEAT_REVIEW_AFTER = 2

#: How often one installed intent is re-judged. Screening only: an audit that changed
#: things on a timer would be a component that edits the house by itself at 3am.
AUDIT_EVERY = dt.timedelta(hours=6)

ROUTE_QUESTIONS_DOC = """One request, one intent to choose from a catalogue, two go/no-go Nouls."""


@dataclass(slots=True)
class Route:
    """What the cheap judge made of a request the house could not answer."""

    decision: str = "ignore"
    intent: str | None = None
    reason: str = ""
    answers: dict[str, Any] = field(default_factory=dict)
    checked: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "checked": self.checked,
            "decision": self.decision,
            "intent": self.intent,
            "reason": self.reason,
            "answers": self.answers,
        }


def _route_state(hass: HomeAssistant, entry: ConfigEntry, row: Observation) -> dict[str, Any]:
    """The state the router reads. Nothing about the house is included on purpose.

    Choosing between intents is a question about the request and the catalogue; the
    catalogue's own examples carry everything the decision needs, and the entities are
    the generaliser's problem, not this one.
    """
    return {
        "request": row.utterance,
        "what_the_agent_answered_at_the_time": (row.answer or "(nothing recorded)")[:600],
        "what_the_agent_did": _describe_tools(
            Turn(
                utterance=row.utterance,
                agent_id=row.agent_id,
                tool_calls=row.tool_calls
                or [{"name": name, "args": {}} for name in row.tools],
                answer=row.answer,
                targets=list(row.targets),
            )
        ),
        "what_it_acted_on": ", ".join(row.targets[:6]) or "(nothing recorded)",
        "the_time_where_the_house_is": dt_util.now().strftime("%A %-I:%M %p %Z"),
    }


def _intent_choice(
    hass: HomeAssistant, language: str, examples: dict[str, list[str]] | None = None
) -> Choice:
    """One option per intent the house can answer, described by its own wordings."""
    if examples is None:
        examples = _intent_examples(hass, language)
    described = _intent_descriptions(hass)
    criteria: dict[str, str] = {}
    for name in sorted(set(described) | set(examples)):
        if name in NOT_LOCAL_INTENTS:
            # Answered by writing words rather than by state, so the words would come
            # straight back. Offering it as an option is offering a mistake.
            continue
        parts: list[str] = []
        if described.get(name):
            parts.append(described[name])
        if samples := examples.get(name):
            shown = ", ".join(repr(text) for text in sorted(samples)[:3])
            parts.append(f"said as {shown}")
        criteria[name] = "; ".join(parts) or "an intent the house answers locally"
    criteria["none_of_these"] = (
        "The house does not answer this kind of request, or two of the above fit "
        "equally well, or it depends on what was said before"
    )
    return Choice(
        "`request` is something the house could not answer locally. Each option below is "
        "an intent it *can* answer, with example wordings that reach it. Which one, if "
        "any, is `request` an example of? Choose none_of_these unless one intent is "
        "clearly the same request as the one that was made.",
        criteria,
    )


def _one_request_question() -> Noul:
    """The question both paths ask before anything is learned from a sentence."""
    return Noul(
        "Does `request` ask for exactly one thing? A sentence asking for two at once - an "
        "action and a question, two actions, two rooms - cannot be answered by one local "
        "sentence, and learning it would answer half of it for ever.",
        true="It asks for one thing",
        false="It asks for two or more things at once",
    )


#: Words that join two clauses. A sentence containing one *may* be two requests, so it is
#: worth one question before binding it. A conjunction is not evidence on its own - "turn
#: on the kitchen and living room lights" is one request with two targets - which is why
#: the judge is asked rather than a word list applied.
COMPOUND_HINTS = (" and ", " then ", " also ", " plus ")


async def async_is_compound(
    hass: HomeAssistant, entry: ConfigEntry, utterance: str
) -> bool | None:
    """Whether a sentence asks for two things. None when there is no judge to ask."""
    if not has_judge(entry):
        return None
    try:
        answers = await async_ask(
            hass, entry, {"request": utterance}, {"one_request": _one_request_question()}
        )
    except JudgeError as err:
        _LOGGER.debug("Irabelle Home Learning could not ask about %r: %s", utterance, err)
        return None
    return (answers["one_request"].noul or 0.0) <= COMPOUND_ABOVE


def _about_the_assistant_question() -> Noul:
    """The question asked before a request is recorded at all."""
    return Noul(
        "`request` is something a person said to this house's voice assistant. Are they "
        "asking the house to do something or answer something, or are they talking *about* "
        "the assistant - asking it to change how it answers, to remember something, or to "
        "check what it just said? Judge only that, not whether the request is answerable.",
        true="They are asking the house to do or answer something",
        false=(
            "They are talking about the assistant itself: change how you answer, update "
            "your template, use a different attribute, remember this, check what you just "
            "said, or do the same for the other one. A sentence that mentions a template, "
            "a reply or a sentence is this, however much it sounds like an instruction - "
            "they are instructing the assistant, not the house"
        ),
    )


#: Words that make a sentence *worth asking about* - nothing more. A phrase about the
#: assistant is usually an instruction about the assistant ("next time add the device
#: name"), and asking costs a judge call on every turn that reaches the agent otherwise.
#: A word list can be wrong here in a way that costs a fraction of a cent; it cannot be
#: wrong in a way that loses a request, because a sentence without any of these is never
#: asked about and is treated as addressed to the house.
ABOUT_THE_ASSISTANT_HINTS = (
    "next time",
    "from now on",
    "reply",
    "response",
    "sentence",
    "phrasing",
    "grammar",
    "grammatically",
    "your answer",
    "the answer",
    "change the",
    "add the",
    "apply",
    "same change",
    "remember this",
    "remember that",
    "when i say",
    "teach you",
    "learn this",
    "don't say",
    "check previous",
    "you just said",
    # Instructions about a reply go through the template as often as the wording:
    # "you need to update the template as I asked for tomorrow and replied for today" had
    # no hint from the list above, so it was recorded as though it were a request to the
    # house - which is the class this gate exists for.
    "template",
    "update the",
    "as i asked",
    "you replied",
    "not what i asked",
)


async def async_is_about_the_assistant(
    hass: HomeAssistant, entry: ConfigEntry, utterance: str
) -> bool | None:
    """Whether a sentence is about the assistant rather than for the house.

    True means it should not be recorded at all: it is a person talking to their
    assistant about how it answers, and those words will never be said again. Recording
    one leaves a permanent "heard once" row and buys a judge call on a second sighting
    that will never come - and in one evening, more than half of the phrases recorded were
    of exactly this kind.

    None means the question could not be asked, and the request is kept: a gate must never
    be the reason something real is dropped.
    """
    lowered = f" {utterance.lower()} "
    if not any(hint in lowered for hint in ABOUT_THE_ASSISTANT_HINTS):
        return None
    if not has_judge(entry):
        return None
    try:
        answers = await async_ask(
            hass,
            entry,
            {"request": utterance},
            {"for_the_house": _about_the_assistant_question()},
        )
    except JudgeError as err:
        _LOGGER.debug("Irabelle Home Learning could not ask about %r: %s", utterance, err)
        return None
    return (answers["for_the_house"].noul or 0.0) < ROUTE_CLEAN


async def async_route_learning(
    hass: HomeAssistant,
    entry: ConfigEntry,
    row: Observation,
) -> Route | None:
    """Ask which intent a request is, before a text model is asked to write anything.

    The generaliser is one 4,000-token text call that decides *and* writes a template.
    Most of what it decides is "this is the intent we already have" - a judgement over a
    catalogue, which is what a Choice is for. Bound here, that judgement costs about a
    hundredth of a cent and no text model at all; the text model is left for the part
    only it can do, which is writing the template.

    Returns None when no classification model is configured or it cannot be reached; the caller
    then falls through to the generaliser exactly as before.
    """
    if not has_judge(entry):
        return None
    state = _route_state(hass, entry, row)
    try:
        answers = await async_ask(
            hass,
            entry,
            state,
            {
                "which_intent": _intent_choice(
                    hass,
                    language_of(entry),
                    await async_run(hass, _intent_examples, hass, language_of(entry)),
                ),
                "one_request": _one_request_question(),
                "repeatable": Noul(
                    "Will `request` be said again, in these words or words close to "
                    "them? A one-off about a particular thing right now will not be.",
                    true="It is the kind of thing that gets said again",
                    false="It is a one-off that is unlikely to be said again",
                ),
                "computable": Noul(
                    "Could one template over the entities and areas of this house "
                    "answer `request` every time, from state alone? Answer false when "
                    "the answer needs words written or repeated back, or depends on "
                    "what was said before.",
                    true="A template over the house's own state could answer it",
                    false="It needs words written, or it depends on an earlier turn",
                ),
            },
        )
    except JudgeError as err:
        _LOGGER.debug("Irabelle Home Learning router unavailable for %r: %s", row.utterance, err)
        return None
    except Exception as err:  # noqa: BLE001 - a router must never block a learning
        _LOGGER.warning("Irabelle Home Learning router failed for %r: %s", row.utterance, err)
        return None

    route = Route(
        answers={
            name: {
                "noul": answer.noul,
                "choice": answer.choice,
                "confidence": answer.confidence,
                "margin": answer.margin(),
            }
            for name, answer in answers.items()
        },
        checked=True,
    )

    if (answers["one_request"].noul or 0.0) <= COMPOUND_ABOVE:
        # One local sentence cannot do two things, so there is nothing here to learn - and
        # binding it anyway would answer half the request and swallow the other half. The
        # agent keeps it, which is what happens today.
        route.decision = "ignore"
        route.reason = (
            f"two requests in one sentence (one_request={answers['one_request'].noul})"
        )
        _LOGGER.info("Irabelle Home Learning routed %r -> ignore (%s)", row.utterance, route.reason)
        return route

    chosen = answers["which_intent"]
    confident = (chosen.confidence or 0.0) >= ROUTE_CONFIDENCE
    decisive = (chosen.margin() or 0.0) >= ROUTE_MARGIN
    fits = chosen.choice not in (None, "none_of_these")
    judged = confident and decisive
    # `repeatable` is asked and recorded, but it does not decide: by the time this runs the
    # phrase has been heard `min_occurrences` times, which *is* the evidence that it gets
    # said again. A model's 0.32 against five repetitions - which is what happened to "is
    # the living room light off?" - is not a reason to refuse.
    computable = (answers["computable"].noul or 0.0) >= COMPUTABLE_ENOUGH

    if fits and judged:
        # One intent clearly is this request. Nothing left for a text model to decide,
        # and nothing for it to write, so the generaliser is not reached at all.
        route.decision = "bind"
        route.intent = str(chosen.choice)
        route.reason = (
            f"{route.intent} fits (confidence {chosen.confidence}, margin {chosen.margin()})"
        )
    elif fits:
        # Two intents both fit, or none does convincingly. That is precisely the
        # judgement the text model is still better at, and the near-tie is what
        # probabilities are for: unknown, not decided.
        route.decision = "generalise"
        route.reason = (
            f"{chosen.choice} was chosen but not decisively "
            f"(confidence {chosen.confidence}, margin {chosen.margin()})"
        )
    elif judged and not computable:
        # Confidently nothing here fits, and it is not worth answering locally: a
        # one-off, or something that needs words written. This is where the generaliser
        # is skipped rather than consulted, and it is most of what gets observed.
        route.decision = "ignore"
        route.reason = (
            f"nothing fits (confidence {chosen.confidence}, margin {chosen.margin()}), "
            f"computable={answers['computable'].noul}, "
            f"repeatable={answers['repeatable'].noul} (asked, not decisive)"
        )
    else:
        # Nothing fits, but it looks repeatable and answerable from state - so a new
        # template is worth having, and only the text model can write one.
        route.decision = "generalise"
        route.reason = (
            f"nothing fits but a template could answer it "
            f"(computable={answers['computable'].noul}, "
            f"repeatable={answers['repeatable'].noul})"
        )

    _LOGGER.info(
        "Irabelle Home Learning routed %r -> %s (%s)", row.utterance, route.decision, route.reason
    )
    return route


#: Phrases that say the same thing twice when they turn up twice in one short answer.
_REPEATED_PHRASES = ("last seen", "last changed", "switched on", "turned on")

#: Two punctuation marks in a row, which is what gluing two finished fragments together
#: produces: a rewrite of an answer keeps the sentence the description already ended with and
#: joins its own clause to it, and the house says "…in the afternoon., the high will be 19".
#: An ellipsis is a deliberate pause, so a run of full stops alone is allowed.
_DOUBLED_PUNCTUATION = re.compile(r"[.!?,;:]{2,}")

#: A value that did not resolve, spoken as the answer: "The office light is unknown." or
#: "The living room light is unknown and the Hue Iris is unknown." Both are `{{ states(...) }}`
#: printed raw, which tells the room nothing; the good version says what it *cannot* do
#: ("its state is unknown so I cannot tell when it was switched on") - so the word only counts
#: here when it ends the clause.
_UNRESOLVED = re.compile(
    r"\b(?:is|are|was|were)\s+(?:unknown|unavailable)\s*(?=[.,;!?]|$|\band\b)", re.IGNORECASE
)

#: The same word standing where a name should be: "Unknown, your iPhone Alex was last seen in
#: the Bedroom" - measured on the live house, on a turn where the speaker sensor had no reading
#: yet, so the reply greeted the asker as "Unknown" and looked up somebody else's phone.
_UNRESOLVED_LEADING = re.compile(r"^\s*(?:unknown|unavailable)\b", re.IGNORECASE)

#: A number and its unit that do not agree. A reply that builds a duration by hand writes
#: `(secs // 3600) ~ ' hours ago'`, which is right until the value is exactly one - measured:
#: "your iPhone Alex was last seen in the Bedroom, 1 hours ago".
_AGREEMENT = re.compile(r"\b1\s+(hours|minutes|seconds|days|weeks|months|years)\b", re.IGNORECASE)

#: A capital in the middle of a sentence, after the comma that joined two of them. Only
#: reported when the *same answer* also writes that word in lowercase: "Showers" beside
#: "Isolated showers" is a join the model forgot to lowercase, while "Sam" appearing once is a
#: proper noun and must be left alone. The evidence is in the text, so no name list is needed.
_MID_SENTENCE_CAPITAL = re.compile(r"(?<=[,;:] )([A-Z][a-z]*)\b")


def reply_problems(text: str) -> list[str]:
    """Why a rendered answer does not read like a sentence, if it does not.

    This is the mechanical half of "is it grammatical": a spoken answer is one sentence,
    so it starts with a capital, ends with punctuation, has no doubled space - which is
    where a value failed to render - carries no unrendered template, and does not say the
    same thing twice. The case that prompted it said

        Sam, your Amazfit GTS Area Last Seen was last seen in the Living Room

    which names the sensor and then answers about it, so "last seen" arrives twice.

    The punctuation rule was added when a model rewrote a weather answer on request and
    produced "…in the afternoon., the high will be 19" and "A little rain is expected., and
    the UV index is 1.7" - both halves individually fine, the join wrong, and nothing else
    here looking at it.
    """
    problems: list[str] = []
    stripped = " ".join(text.split())
    if not stripped:
        return ["it is empty"]
    if not stripped[0].isupper() and not stripped[0].isdigit():
        problems.append("it does not start with a capital letter")
    if stripped[-1] not in ".!?":
        problems.append("it does not end with a full stop")
    if "  " in text.strip() or not stripped:
        problems.append("it has a doubled space, so a value did not render")
    if any(marker in stripped for marker in ("{{", "}}", "{%")):
        problems.append("a template expression was not rendered")
    if disagreement := _AGREEMENT.search(stripped):
        problems.append(
            f"it says {disagreement.group(0)!r}, so the number and the unit do not agree"
        )
    unresolved = _UNRESOLVED.search(stripped) or _UNRESOLVED_LEADING.search(stripped)
    if unresolved is not None:
        problems.append(
            f"it says {unresolved.group(0).strip()!r}, so a value it read did not resolve and "
            "the answer is about nothing"
        )
    for match in _DOUBLED_PUNCTUATION.finditer(stripped):
        marks = match.group()
        if set(marks) == {"."}:
            continue  # an ellipsis is deliberate
        problems.append(
            f"it has two punctuation marks in a row ({marks!r}), so two half-sentences were "
            "glued together"
        )
        break
    for match in _MID_SENTENCE_CAPITAL.finditer(stripped):
        word = match.group(1)
        if re.search(rf"\b{re.escape(word.lower())}\b", stripped):
            problems.append(
                f"it starts the word {word!r} with a capital in the middle of a sentence, "
                "which is how a joined sentence keeps the capital it had"
            )
            break
    lowered = stripped.lower()
    for phrase in _REPEATED_PHRASES:
        if lowered.count(phrase) > 1:
            problems.append(f"it says {phrase!r} twice")
    words = re.findall(r"[a-z']+", lowered)
    repeated = next(
        (word for word, following in zip(words, words[1:]) if word == following and len(word) > 2),
        None,
    )
    if repeated:
        problems.append(f"it repeats the word {repeated!r}")
    return problems


def reply_problems_for(
    hass: HomeAssistant, source: str, sample: dict[str, str] | None = None
) -> dict[str, list[str]]:
    """Which of the people this answer is for hear something that does not read well."""
    heard = _render_for_people(hass, source, sample)
    people = {who: text for who, text in heard.items() if who != "(nobody speaking)"}
    if not people:
        people = heard
    # The render with nobody speaking is checked as well, and only here: it is where a reply
    # that reads the speaker sensor unguarded gives itself away ("unknown, your watch was last
    # seen..."). The *judge* is deliberately not shown it for the "wrong person" question - a
    # correct speaker-aware answer has to fall back to something when nobody is speaking - but
    # "does this read like a sentence" has no such excuse.
    for who, text in heard.items():
        if who == "(nobody speaking)" and reply_problems(text):
            people[who] = text
    return {who: reply_problems(text) for who, text in people.items() if reply_problems(text)}


async def async_change_response(
    hass: HomeAssistant, entry: ConfigEntry, name: str, response: str
) -> dict[str, Any]:
    """Install a reply, judge how it reads, and fix the wording once if it reads badly.

    The model that writes a reply is the model that can rewrite one, and asking it to
    polish its own sentence is the difference between a change that works and a change the
    person has to ask for again:

        "your Amazfit GTS Area Last Seen was last seen in the Living Room, 4 minutes ago"

    says the same thing twice. The facts are pinned - every Jinja expression has to come
    back character for character - the rewrite is validated exactly like the original, and
    one attempt is made, so the worst case is the sentence that was asked for.
    """
    result = await async_set_response(hass, entry, name, response)
    spec = next((item for item in entry.runtime_data.intents if item.name == name), None)
    if spec is None:
        return {**result, "says": response}
    source = spec.response or spec.template
    sample = slot_sample(getattr(spec, "slots", None))
    heard = _render_for_people(hass, source, sample)
    complaints = reply_problems_for(hass, source, sample)
    says = heard.get("Sam") or next(iter(heard.values()), "")

    screen = await async_screen_learning(hass, entry, None, spec)
    flagged = [reason for reason in (screen.reasons if screen else []) if "reads_like" in reason]
    if not complaints and not flagged:
        return {**result, "says": says, "screen": screen.as_dict() if screen else None}

    problem = "; ".join(
        f"for {who}: " + ", ".join(issues) for who, issues in complaints.items()
    ) or "a cheaper judge says it does not read like a sentence"
    fixed = await async_polish_sentence(hass, entry, source, says, problem)
    if not fixed or fixed == source:
        return {
            **result,
            "says": says,
            "screen": screen.as_dict() if screen else None,
            "wording": problem,
            "polish": "no rewrite offered",
        }
    try:
        await async_set_response(hass, entry, name, fixed)
    except LearnerError as err:
        return {**result, "says": says, "wording": problem, "polish": f"refused: {err}"}
    polished_hears = _render_for_people(hass, fixed, sample)
    return {
        **result,
        "wording": problem,
        "polish": "rewritten",
        "was": says,
        "says": polished_hears.get("Sam") or next(iter(polished_hears.values()), ""),
        "still_awkward": reply_problems_for(hass, fixed, sample),
    }


POLISH_INSTRUCTIONS = """\
The house answers a request with one spoken sentence, and the wording is wrong. Rewrite
the sentence; do not change what it says.

THE TEMPLATE
${template}

WHAT IT SAYS RIGHT NOW
${says}

WHAT IS WRONG WITH THE WORDING
${problem}

Return the whole template again, as ONE sentence a person would say out loud, saying
exactly the same thing from exactly the same facts. Keep every {{ ... }} expression that
computes a value, and every {% ... %} statement that sets one, character for character -
change only the words around them. Keep the length about the same. Do not add information that is not already there and
do not leave any out. Start with a capital letter and end with a full stop, and do not
repeat a phrase.

Reply with a short "problem" only if the sentence cannot be fixed this way (for example
because it states something that is not true), and leave "sentence" empty then. Otherwise
leave "problem" empty and put the rewritten sentence in "sentence".
"""

POLISH_STRUCTURE = {
    "sentence": {
        "description": "the rewritten sentence, in full, or empty if it cannot be fixed",
        "selector": {"text": {"multiline": True}},
    },
    "problem": {
        "description": "why it cannot be fixed, or empty",
        "selector": {"text": {}},
    },
}


async def async_polish_sentence(
    hass: HomeAssistant, entry: ConfigEntry, template: str, says: str, problem: str
) -> str | None:
    """Ask the text model to fix the wording of one sentence, keeping what it says.

    A model wrote the sentence; a model can fix the sentence. The facts are pinned in
    place - every Jinja expression has to come back character for character - and the
    result is validated like any other reply, so the worst case is the original.
    """
    entity_id = _ai_task_for(hass, entry)
    if entity_id is None:
        return None
    instructions = Template(POLISH_INSTRUCTIONS).substitute(
        template=template, says=says, problem=problem
    )
    try:
        response = await async_generate_data(
            hass,
            {
                "task_name": "fix the wording of an answer",
                "entity_id": entity_id,
                "instructions": instructions,
                "structure": POLISH_STRUCTURE,
            },
        )
    except Exception as err:  # noqa: BLE001 - a failed polish must change nothing
        _LOGGER.warning("Irabelle Home Learning could not fix the wording: %s", err)
        return None

    payload = (response or {}).get("data")
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except ValueError:
            payload = None
    if not isinstance(payload, dict):
        return None
    sentence = " ".join(str(payload.get("sentence") or "").split())
    return sentence or None


async def async_check_installed(
    hass: HomeAssistant, entry: ConfigEntry
) -> list[dict[str, Any]]:
    """Run the checks a new learning gets over the ones already installed.

    A learning is checked when it is made and never again, so a rename, a removed entity
    or a botched reply edit leaves it answering wrongly and silently - the template
    renders the word "unknown" and the house says it out loud. This is the same two checks
    (mechanical, then how it reads), applied to what is already there.

    Reported, not repaired: repairing means a model call per intent, and a component that
    rewrites the house on every restart is worse than one that names what is wrong.
    """
    from .intents import _check_template  # noqa: PLC0415 - intents imports this module

    drift: list[dict[str, Any]] = []
    for spec in entry.runtime_data.intents:
        source = spec.response or spec.template
        declared = getattr(spec, "slots", None)
        sample = slot_sample(declared)
        problems: list[str] = []
        try:
            _check_template(hass, source, set(declared or {}), sample)
        except Exception as err:  # noqa: BLE001 - any refusal is a finding
            problems.append(str(err))
        # An attribute that is not there renders empty instead of failing, which is how a
        # reply that used to be detailed quietly became current conditions only.
        problems.extend(missing_attributes(hass, source))
        for who, why in reply_problems_for(hass, source, sample).items():
            problems.append(f"reads badly for {who}: {', '.join(why)}")
        if problems:
            drift.append({"intent": spec.name, "file": spec.file, "problems": problems})
    if drift:
        _LOGGER.critical(
            "Irabelle Home Learning: %d installed intent(s) no longer pass their own checks: %s",
            len(drift),
            "; ".join(f"{item['intent']}: {item['problems'][0]}" for item in drift),
        )
    return drift


async def async_audit_one(hass: HomeAssistant, entry: ConfigEntry) -> dict[str, Any] | None:
    """Re-judge one installed intent, the next in line, and report what it finds.

    Rotation is by position in the store, so the whole set is covered over time without
    tracking anything extra. Screen only - see `AUDIT_EVERY`.
    """
    data: LearnerData = entry.runtime_data
    if not data.intents or not has_judge(entry):
        return None
    position = int(data.spend.get("audit_position", 0)) % len(data.intents)
    spec = data.intents[position]
    data.spend["audit_position"] = position + 1
    screen = await async_screen_learning(hass, entry, None, spec)
    if screen is None:
        return None
    finding = {
        "intent": spec.name,
        "clean": screen.clean,
        "reasons": screen.reasons,
    }
    data.spend["last_audit"] = finding
    if not screen.clean:
        _LOGGER.warning(
            "Irabelle Home Learning audit of %s found: %s. Run irabelle_home_learning.review with apply to fix it.",
            spec.name,
            ", ".join(screen.reasons),
        )
    await async_save(hass, entry)
    return finding


async def async_sweep_meta(hass: HomeAssistant, entry: ConfigEntry) -> dict[str, Any]:
    """Mark the requests already recorded that are about the assistant, not for the house.

    The gate stops new ones; this deals with what was recorded before it existed. Nothing
    is deleted - an ignored row is still the record of what was heard, and why it was left
    alone - but it leaves the count of what is waiting for a second sighting.
    """
    data: LearnerData = entry.runtime_data
    examined = 0
    marked: list[str] = []
    for row in data.observations:
        if row.state != "new":
            continue
        examined += 1
        if await async_is_about_the_assistant(hass, entry, row.utterance):
            row.state = "ignored"
            row.note = "about the assistant rather than for the house"
            marked.append(row.utterance)
    if marked:
        await async_save(hass, entry)
        _LOGGER.info(
            "Irabelle Home Learning swept %d of %d waiting request(s): %s",
            len(marked),
            examined,
            ", ".join(repr(text) for text in marked[:5]),
        )
    return {"examined": examined, "marked": marked}


def certain_bind(route: Route) -> bool:
    """Whether this is a bind worth acting on the first time the phrase is heard.

    Binding costs nothing and is undone by one command, but the second sighting is what
    usually justifies it - so a first sighting is only acted on when the choice is as close
    to certain as the judge gets, and the sentence asks for one thing.
    """
    if route.decision != "bind":
        return False
    chosen = (route.answers or {}).get("which_intent") or {}
    one = (route.answers or {}).get("one_request") or {}
    return (
        (chosen.get("confidence") or 0.0) >= ROUTE_CERTAIN
        and (chosen.get("margin") or 0.0) >= ROUTE_CERTAIN_MARGIN
        and (one.get("noul") or 0.0) >= ROUTE_CLEAN
    )


async def async_review_learning(
    hass: HomeAssistant,
    entry: ConfigEntry,
    intent: str,
    *,
    utterance: str | None = None,
    answer: str = "",
    row: Observation | None = None,
    screen: Screen | None = None,
    apply: bool = True,
    replace: bool = True,
) -> dict[str, Any] | None:
    """Have a model check a learning, and undo it when the answer is wrong.

    ``apply=False`` reports what the reviewer would change without changing it, which is
    what an on-demand audit wants: asking the house to look at itself should not be able
    to delete an intent the person only wanted inspected.

    ``replace=False`` means a removed intent is not replaced. That is how the replacement
    path stops itself from looping: the attempt it spawns is reviewed normally, so a bad
    replacement is still removed, but it does not spawn another attempt of its own.

    The guards catch a sentence the matcher will not accept, a binding whose words
    name nothing, and a bind whose target reads a different entity. None of them
    catches an answer that is confidently about the wrong thing - and those are the
    ones that stick: "is the bedroom TV on" was answered with both TVs and
    "is the living room TV on" with the bedroom's, and every mechanical check passed
    both.
    """
    spec = next((item for item in entry.runtime_data.intents if item.name == intent), None)
    if spec is None:
        return None

    # Cheap screen first: six snap judgements for about a hundredth of a cent. A
    # learning that passes it never reaches the text model, which is what makes it
    # affordable to check everything rather than only the interesting cases.
    screen = await async_screen_learning(
        hass, entry, row, spec, utterance=utterance, answer=answer
    )
    if screen is not None and row is not None:
        # The same record as the router keeps: what the judge said, and how sure it was.
        row.judgement = dict(row.judgement or {}, screen=screen.as_dict())
        await async_save(hass, entry)
    if screen is not None and screen.clean:
        return {"review": "ok", "intent": intent, "screen": screen.as_dict()}

    entity_id = _ai_task_for(hass, entry)
    if entity_id is None or _review_limited(entry.entry_id):
        # No text reviewer available: the screen's opinion is all we have, and it is
        # better than nothing. Report what it found rather than silence.
        return {
            "review": "screen-only",
            "intent": intent,
            "screen": screen.as_dict() if screen is not None else None,
        }

    phrasings = (
        await async_run(hass, learned_phrases, hass, language_of(entry))
    ).get(intent, [])
    if not phrasings:
        # Nothing here for the text reviewer to rewrite. A binding to a built-in
        # intent has no phrasings of its own, and those are exactly the learnings
        # nothing else looks at - so report the screen's own verdict rather than
        # dropping it.
        if screen is not None and not screen.clean:
            return {
                "review": "screen-only",
                "intent": intent,
                "screen": screen.as_dict(),
            }
        return None

    context = ""
    if utterance:
        context += (
            f"THE REQUEST THAT WAS LEARNED\n{utterance}\n\n"
            f"WHAT THE AGENT ANSWERED AT THE TIME\n{answer or '(nothing recorded)'}\n\n"
        )
    if screen is not None and not screen.clean:
        context += (
            "A cheaper model already looked at this and said: "
            + ", ".join(screen.reasons)
            + ". Check that, and anything else you find.\n\n"
        )

    seen = spec.response or spec.template
    reads = "\n".join(
        f"- {entity_id}: {state}"
        for entity_id, state in _screen_entities(hass, seen).items()
    )
    absent = missing_attributes(hass, seen)
    if absent:
        reads += (
            "\n\nATTRIBUTES THIS TEMPLATE READS THAT ARE NOT THERE (a missing attribute "
            "renders empty instead of failing, so the answer silently falls back to "
            "something else):\n" + "\n".join(f"- {warning}" for warning in absent)
        )

    instructions = Template(REVIEW_INSTRUCTIONS).substitute(
        context=context,
        intent=intent,
        template=seen,
        rendered="\n".join(f"- {who}: {says}" for who, says in _rendered_variants(hass, spec).items()),
        reads=reads,
        now=dt_util.now().strftime("%A %-I:%M %p %Z"),
        phrasings="\n".join(f"- {phrase}" for phrase in phrasings),
        entities=_exposed_entities(hass),
        attributes=_prompt_attributes(hass, utterance or "", seen),
    )
    try:
        response = await async_generate_data(
            hass,
            {
                "task_name": "review a learned answer",
                "entity_id": entity_id,
                "instructions": instructions,
                "structure": REVIEW_STRUCTURE,
            },
        )
    except Exception as err:  # noqa: BLE001 - a failed review must undo nothing
        _LOGGER.warning("Irabelle Home Learning review failed for %s: %s", intent, err)
        return None

    payload = (response or {}).get("data")
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except ValueError:
            payload = None
    if not isinstance(payload, dict):
        return None

    verdict = str(payload.get("verdict") or "ok").strip().lower()
    problem = str(payload.get("problem") or "").strip()
    if verdict != "fix":
        _LOGGER.debug("Irabelle Home Learning review of %s: ok", intent)
        return {
            "review": "ok",
            "intent": intent,
            "screen": screen.as_dict() if screen is not None else None,
        }

    actions: dict[str, Any] = {"review": "fix", "intent": intent, "problem": problem}
    if screen is not None:
        # Reported even when the text model agreed there was a fault: the reasons are
        # what a threshold is later tuned against, and a fix with no record of why is
        # not something that can be measured.
        actions["screen"] = screen.as_dict()

    new_template = str(payload.get("template") or "").strip()
    replaced = False
    if new_template and new_template != seen:
        if not apply:
            actions["would_replace_template"] = new_template
        else:
            # A replacement is installed only when it is no worse than what it replaces.
            # The screen is noisy at the margin (a correct weather answer came back at
            # reads_like_a_sentence=0.52), and an auto-applied rewrite of a good reply is
            # how "Tomorrow's weather will be..." got installed over a working forecast.
            # The candidate is screened exactly as the current one was; if that cannot see
            # an improvement, the template is reported and left alone.
            candidate = SimpleNamespace(
                name=spec.name,
                template=new_template,
                response=None,
                slots=spec.slots,
                file=spec.file,
            )
            check = await async_screen_learning(
                hass, entry, row, candidate, utterance=utterance, answer=answer
            )
            current = len(screen.reasons) if screen is not None else None
            if (
                check is not None
                and not check.clean
                and current is not None
                and len(check.reasons) >= current
            ):
                actions["template"] = (
                    "not replaced: the replacement is no cleaner than the one it replaces "
                    f"({', '.join(check.reasons) or 'no reason given'})"
                )
                actions["would_replace_template"] = new_template
                actions["candidate_screen"] = check.as_dict()
                _LOGGER.warning(
                    "Irabelle Home Learning left %s alone: the proposed replacement is no cleaner (%s)",
                    intent,
                    ", ".join(check.reasons),
                )
            else:
                try:
                    await async_update_template(hass, entry, intent, new_template)
                    actions["template"] = "replaced"
                    replaced = True
                except LearnerError as err:
                    actions["template"] = f"refused: {err}"

    wanted = {
        normalize_sentence(line.strip(" -•\t"))
        for line in str(payload.get("remove_phrasings") or "").splitlines()
        if line.strip(" -•\t")
    }
    removed: list[str] = []
    for phrase in phrasings:
        if normalize_sentence(phrase) not in wanted:
            continue
        if not apply:
            removed.append(phrase)
            continue
        try:
            await async_forget(hass, entry, phrase)
            removed.append(phrase)
        except LearnerError:
            continue
    forgotten = False
    if removed:
        actions["removed" if apply else "would_remove"] = removed
        stripped = {normalize_sentence(phrase) for phrase in removed}
        if not [p for p in phrasings if normalize_sentence(p) not in stripped]:
            # Nothing can reach it any more. An unreachable intent is dead weight,
            # and leaving one behind is how "registered but never matched" piles up.
            if not apply:
                actions["would_forget_intent"] = intent
            else:
                try:
                    await async_forget_intent(hass, entry, intent)
                    forgotten = True
                    actions["forgotten_intent"] = intent
                    _LOGGER.warning(
                        "Irabelle Home Learning review removed every phrasing of %s, so the intent "
                        "went with them: %s",
                        intent,
                        problem or "the answer was wrong",
                    )
                except LearnerError:
                    pass

    if forgotten and replace and row is not None and row.utterance:
        # Deleting a wrong local answer is only half the job: the request now goes back
        # to the cloud model every time it is said, for ever. So the request is handed
        # back with what was wrong written down, and the generaliser gets one attempt at
        # answering it *properly* - which is how "where is Alex's watch", answered by a
        # template that follows the speaker instead of naming Alex, becomes an intent
        # that names her.
        #
        # One attempt only. A replacement that is itself removed is not replaced again:
        # a model that keeps proposing the same wrong template would otherwise loop.
        replacement = await _generalize(
            hass,
            entry,
            row,
            force=True,
            correction=(
                f"{intent} was removed because {problem or 'its answer was wrong'}. Any "
                "intent you proposed before is gone. Choose again knowing that: create "
                "one that answers the request correctly, or ignore it."
            ),
        )
        if replacement is not None:
            actions["replacement"] = replacement

    if replaced and apply:
        # A replaced template is a *new* template and nothing has looked at it. The
        # mechanical checks ran - the entities exist, there are no undeclared variables -
        # but nothing asked whether it now answers the request, and one measured
        # replacement answered "the office LED is off" from an entity whose state is
        # `unknown`. Re-screen it and report. Not fix again: a second opinion at this
        # point rests on a threshold nobody has calibrated, and the honest thing is to
        # record it and let `irabelle_home_learning.review` be run again.
        fresh = next(
            (item for item in entry.runtime_data.intents if item.name == intent), spec
        )
        rescreen = await async_screen_learning(
            hass, entry, row, fresh, utterance=utterance, answer=answer
        )
        actions["rescreen"] = (
            rescreen.as_dict() if rescreen is not None else {"checked": False}
        )

    if row is not None:
        row.note = f"review: {problem or 'the answer was wrong'}"
        await async_save(hass, entry)
    if removed or "template" in actions:
        _LOGGER.warning(
            "Irabelle Home Learning review corrected %s: %s (%s)",
            intent,
            problem or "the answer was wrong",
            actions,
        )
    return actions


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


async def async_consider_turn(
    hass: HomeAssistant, entry: ConfigEntry, turn: Turn
) -> dict[str, Any] | None:
    """Decide what a finished turn teaches the house - then tidy what it learned.

    A learning adds one line to a group; once a group holds a few, the fold pass proposes the
    template that says them all at once (`fold.py`). It runs after the learning, never instead
    of it, and a fold that cannot be proved changes nothing.
    """
    result = await _consider_turn(hass, entry, turn)
    if result and result.get("ok") and result.get("intent"):
        from .fold import async_fold  # noqa: PLC0415 - fold imports this module

        try:
            folded = await async_fold(
                hass, entry, str(result["intent"]), apply=True, only_ready=True
            )
        except Exception:  # noqa: BLE001 - tidying must never undo a learning
            _LOGGER.warning("Irabelle Home Learning could not fold %s", result["intent"], exc_info=True)
        else:
            if folded["folded"]:
                result["fold"] = folded["folded"]
    return result


async def _consider_turn(
    hass: HomeAssistant,
    entry: ConfigEntry,
    turn: Turn,
) -> dict[str, Any] | None:
    """Decide what, if anything, a finished turn teaches the house."""
    data: LearnerData = entry.runtime_data
    utterance = " ".join((turn.utterance or "").split())
    if not utterance:
        return None
    normalized = normalize_sentence(utterance)
    if not normalized:
        return None

    # Already answered locally - so this phrase *was* answered locally, and the person
    # asked again anyway. A pipeline whose local matcher matched never calls this agent,
    # so a phrase that reaches here while the house already claims to answer it means the
    # local answer did not satisfy. That is the only negative signal a person gives
    # without being asked for one, and it used to be dropped on the floor.
    index = await async_run(hass, sentence_index, hass, language_of(entry))
    if owner := index.get(normalized):
        count = data.repeats.get(owner, 0) + 1
        data.repeats[owner] = count
        _LOGGER.info(
            "Irabelle Home Learning: %r is answered locally by %s, yet the request came here (%d)",
            utterance,
            owner,
            count,
        )
        if count >= REPEAT_REVIEW_AFTER and owner in {
            spec.name for spec in data.intents
        }:
            # Enough evidence to look at it. The reviewer is the same one a new learning
            # gets; if the answer is wrong it can remove a phrasing or replace the
            # template, and if it is right it says so.
            data.repeats.pop(owner, None)
            await async_save(hass, entry)
            outcome = await async_review_learning(
                hass, entry, owner, utterance=utterance, answer=turn.answer
            )
            data.last_result = {"repeated": owner, "review": outcome}
            await async_save(hass, entry)
            return None
        await async_save(hass, entry)
        return None

    # Asked before the row exists, so a sentence about the assistant leaves no trace at
    # all. Only sentences containing one of the hint words are put to the judge.
    if await async_is_about_the_assistant(hass, entry, utterance):
        _LOGGER.info(
            "Irabelle Home Learning left %r alone: it is about the assistant, not for the house",
            utterance,
        )
        return None

    row = record_observation(
        data,
        Observation(
            utterance=utterance,
            normalized=normalized,
            agent_id=turn.agent_id,
            tools=turn.tool_names,
            tool_calls=turn.tool_calls,
            targets=turn.targets,
            answer=(turn.answer or "")[:1000],
        ),
    )
    # An ignored row is not closed for ever. A judge can be wrong - twice today it was,
    # and one of those phrases was then said eleven times while nothing changed, because
    # the only thing that could have changed its mind refused to look again. Every
    # doubling of the evidence buys it one more look: 2 sightings, then 4, then 8.
    if row.state == "ignored" and row.hits < min_occurrences(entry) * (row.looks + 1):
        await async_save(hass, entry)
        return None
    if row.state == "ignored":
        _LOGGER.info(
            "Irabelle Home Learning looks at %r again: said %d times since it was ignored",
            row.utterance,
            row.hits,
        )
    row.looks += 1

    # A follow-up is not a phrasing. "when?" is answered by whatever came before
    # it, so learning it would pin those words to one answer forever - and the
    # generaliser will happily say "bind" for it, because in context it did.
    if names_nothing(utterance):
        row.state = "ignored"
        row.note = (
            "a follow-up rather than a phrasing: every word in it depends on the "
            "previous turn, so learning it would pin them to one answer"
        )
        await async_save(hass, entry)
        _LOGGER.debug("Irabelle Home Learning will not learn the follow-up %r", utterance)
        return None

    bound = await intent_from_turn(hass, turn)
    if bound is not None:
        intent, args = bound
        slots = _reproducible_slots(args, utterance)
        if slots is None:
            # The words did not contain the target literally - but a speech-to-text
            # mistake ("the D2 keyword") still names it. The judge tells the two apart.
            slots = await async_heard_target(hass, entry, utterance, args)
        if slots is not None:
            # A binding to a built-in intent is right by construction *because that intent
            # ran for the whole sentence* - which a compound sentence breaks. "Turn off the
            # office light and tell me the time" ran HassTurnOff, and binding the sentence
            # to HassTurnOff answers half of it and swallows the rest, for ever. It got
            # installed exactly that way before this check existed.
            if any(hint in f" {normalized} " for hint in COMPOUND_HINTS):
                if await async_is_compound(hass, entry, utterance):
                    row.state = "ignored"
                    row.note = (
                        "two requests in one sentence: one local intent would answer half "
                        "of it and swallow the other half"
                    )
                    await async_save(hass, entry)
                    _LOGGER.info(
                        "Irabelle Home Learning will not learn the compound request %r", utterance
                    )
                    return None
            return await _bind(hass, entry, row, intent, slots)

    # The judge is asked on the first sighting too, but only a bind that is as good as
    # certain is acted on then - see `certain_bind`. Everything else waits for the second
    # sighting, which is what makes the wait affordable in the first place.
    first_sighting = row.hits < min_occurrences(entry)

    # Judgement first, generation second. Deciding *which* intent this is, is a choice
    # over a catalogue; only writing a template needs a text model. So the cheap judge
    # goes first, and the expensive one is reached only when there is something for it
    # to write.
    if (route := await async_route_learning(hass, entry, row)) is not None:
        # Kept with the observation, not just logged. A threshold is only a guess until
        # the answer it was applied to can be compared with what happened next, and
        # this is that record.
        row.judgement = dict(row.judgement or {}, route=route.as_dict())
        if first_sighting and not certain_bind(route):
            _LOGGER.debug(
                "Irabelle Home Learning waits for a second sighting of %r (%s)",
                row.utterance,
                route.reason,
            )
            await async_save(hass, entry)
            return None
        if route.decision == "bind" and route.intent:
            refusal = _binding_scope_problem(hass, entry, row.utterance, route.intent)
            if refusal is None:
                answer = await _bind(hass, entry, row, route.intent, {})
                if answer is not None:
                    answer["route"] = route.as_dict()
                return answer
            _LOGGER.info(
                "Irabelle Home Learning router chose %s for %r, but %s; asking the generaliser",
                route.intent,
                row.utterance,
                refusal,
            )
        elif route.decision == "ignore":
            # The router is the only model that looked at this, and it said there is
            # nothing here worth answering locally. Skipping the generaliser is where
            # the saving is: most observed turns are exactly this, and each one used to
            # cost a 4,000-token text call.
            row.state = "ignored"
            row.note = f"router: {route.reason}"
            await async_save(hass, entry)
            return {
                "action": "ignore",
                "reason": route.reason,
                "route": route.as_dict(),
            }
        else:
            _LOGGER.debug(
                "Irabelle Home Learning router handed %r to the generaliser: %s",
                row.utterance,
                route.reason,
            )

    return await _generalize(hass, entry, row)


async def _bind(
    hass: HomeAssistant,
    entry: ConfigEntry,
    row: Observation,
    intent: str,
    slots: dict[str, Any],
) -> dict[str, Any] | None:
    """Bind the observed words to an intent that demonstrably answers them."""
    if (
        intent in TARGETED_INTENTS
        and not slots
        and intent not in {spec.name for spec in entry.runtime_data.intents}
    ):
        # "Toggle the D2 keyword" was bound to HassTurnOn with no name: the sentence then
        # matched, found nothing to turn on, and answered with an error for ever.
        row.state = "ignored"
        row.note = f"{intent} needs a target and these words were bound without one"
        await async_save(hass, entry)
        _LOGGER.info(
            "Irabelle Home Learning refused to bind %r to %s without a target",
            row.utterance,
            intent,
        )
        return {"action": "bind", "ok": False, "error": row.note}
    try:
        result = await async_learn(
            hass,
            entry,
            utterances=[row.utterance],
            response=row.answer or None,
            intent=intent,
            slots=slots or None,
            note="bound from conversation",
            source=row.agent_id or "conversation",
        )
    except LearnerError as err:
        row.state = "ignored"
        row.note = f"binding refused: {err}"
        await async_save(hass, entry)
        return {"action": "bind", "ok": False, "error": str(err)}

    row.state = "learned"
    row.intent = intent
    row.note = f"bound to {intent}"
    await async_save(hass, entry)
    _LOGGER.info("Irabelle Home Learning bound %r to %s", row.utterance, intent)

    answer: dict[str, Any] = {
        "action": "bind",
        "ok": True,
        "intent": intent,
        "result": result,
    }
    # A binding to a built-in intent is right by construction: that intent ran and
    # produced the answer. A binding to a *learned* intent is only as good as the
    # template behind it, so only that one needs checking.
    if intent in {spec.name for spec in entry.runtime_data.intents}:
        if (review := await async_review_learning(hass, entry, intent, utterance=row.utterance, answer=row.answer, row=row)) is not None:
            answer["review"] = review
    return answer


async def async_reconsider(
    hass: HomeAssistant,
    entry: ConfigEntry,
    *,
    utterance: str | None = None,
) -> dict[str, Any]:
    """Re-run the judgement for observations the house did not learn from.

    Generalisation is a judgement, and a judgement can be wrong. This is the
    handle for saying so without hand-editing a sentence file.
    """
    data: LearnerData = entry.runtime_data
    index = await async_run(hass, sentence_index, hass, language_of(entry))

    if utterance:
        wanted = normalize_sentence(utterance)
        rows = [row for row in data.observations if row.normalized == wanted]
    else:
        # Anything not learned, plus anything learned into an intent that has
        # since been forgotten - that is how a bad automatic intent heals.
        owned = {spec.name for spec in data.intents}
        rows = [
            row
            for row in data.observations
            if row.state != "learned"
            or (row.intent is not None and row.intent not in owned)
        ]

    done: list[dict[str, Any]] = []
    for row in rows:
        if row.normalized in index:
            row.state = "learned"
            row.note = "already answered locally"
            continue
        row.state = "new"
        row.note = "reconsidered"
        result = await _generalize(hass, entry, row, force=True)
        done.append(
            {
                "utterance": row.utterance,
                "state": row.state,
                "intent": row.intent,
                "note": row.note,
                "result": result,
            }
        )
    await async_save(hass, entry)
    async_dispatcher_send(hass, SIGNAL_UPDATED)
    return {"reconsidered": len(done), "rows": done[:25]}
