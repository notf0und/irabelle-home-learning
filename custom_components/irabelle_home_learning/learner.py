"""Core of the Irabelle Home Learning: validate, install and forget learned phrasings.

The assistant is good at deciding what an utterance means and bad at safely
editing configuration. So the model calls a tool, and this module does the
mechanics: validate against the intent registry, merge into a file this
integration owns, record provenance, and reload `conversation` so the phrase is
answered locally from then on.

Two invariants matter more than features:

* **A bad write must never break local voice.** One unparseable file in
  `custom_sentences/` takes down sentence matching for every agent, so files are
  parsed and quarantined before anything is reloaded.
* **Never silently re-target an existing phrase.** An utterance already bound to
  another intent is rejected, not overwritten.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers import intent as intent_helper
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.storage import Store
from homeassistant.util import yaml as yaml_util

from .fileio import async_run
from .upstream import bundled as upstream_bundle
from .upstream import PROVIDED_SLOTS, expected_slots, slot_name
from .const import (
    BUILTIN_INTENTS,
    SPEAKER_VARIABLES,
    CONF_AI_TASK_ENTITY,
    CONF_ALLOWED_PLATFORMS,
    CONF_AUTO_LEARN,
    CONF_LANGUAGE,
    JEV_PRICE_PER_MTOK,
    CONF_MIN_OCCURRENCES,
    DEFAULT_ALLOWED_PLATFORMS,
    DEFAULT_AUTO_LEARN,
    DEFAULT_LANGUAGE,
    DEFAULT_MIN_OCCURRENCES,
    FILE_PREFIX,
    MAX_OBSERVATIONS,
    MAX_UTTERANCES,
    MAX_UTTERANCE_CHARS,
    QUARANTINE_DIR,
    SENTENCES_DIR,
    SIGNAL_UPDATED,
    STORAGE_KEY,
    STORAGE_VERSION,
)

_LOGGER = logging.getLogger(__name__)

#: A slot as a sentence spells it: `{day}`, `{list:slot}`. `intents.py` has its own copy for
#: the same reason this one exists - the two modules cannot import each other.
_SENTENCE_SLOT = re.compile(r"\{([^{}]+)\}")


class LearnerError(ServiceValidationError):
    """A proposal the learner refused.

    A ServiceValidationError, not a bare HomeAssistantError, so a caller of the
    service gets 400 and the reason instead of an opaque 500 - the same reason the
    LLM tools return as JSON.
    """


@dataclass(slots=True)
class LearnedPhrase:
    """One utterance and what it was bound to."""

    utterance: str
    intent: str
    slots: dict[str, Any] | None = None
    note: str | None = None
    source: str | None = None
    learned_at: str = field(default_factory=lambda: dt.datetime.now().isoformat(timespec="seconds"))
    file: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "utterance": self.utterance,
            "intent": self.intent,
            "slots": self.slots,
            "note": self.note,
            "source": self.source,
            "learned_at": self.learned_at,
            "file": self.file,
        }


@dataclass(slots=True)
class LearnedIntent:
    """A whole new intent the house learned, with the template that answers it.

    Registered with `intent.async_register` at setup, so nothing is written to
    `intent_script.yaml` and the user's own file is never touched.
    """

    name: str
    template: str
    #: The reply, when it is not simply the template. Written into the sentence file under
    #: `responses: intents: <name>: default`, which is where Home Assistant keeps the reply
    #: for a built-in intent too - so changing how the house answers is a change to a file
    #: a person can read, and one the learner can make on request.
    response: str | None = None
    description: str | None = None
    note: str | None = None
    source: str | None = None
    #: The list slots this intent's sentences refer to, exactly as they are written under
    #: `lists:` in its sentence file (`{"day": {"values": [...]}}`). A request that only
    #: swaps one word for another value in the same class - a weekday, a room, a person -
    #: is *one* intent with a slot rather than one intent per value, and these are what
    #: tell the template checker that `{{ day }}` is defined rather than a mistake. The
    #: render checks fill each slot with its first value, so the answer can be read before
    #: anyone asks it.
    slots: dict[str, Any] | None = None
    learned_at: str = field(default_factory=lambda: dt.datetime.now().isoformat(timespec="seconds"))
    file: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "template": self.template,
            "response": self.response or self.template,
            "description": self.description,
            "note": self.note,
            "source": self.source,
            "slots": self.slots,
            "learned_at": self.learned_at,
            "file": self.file,
        }


@dataclass(slots=True)
class Observation:
    """A request the local matcher missed, and what the assistant did with it.

    Recorded mechanically from the chat log, never proposed by a model. `hits`
    is what makes generalisation affordable: the first time a phrasing is heard
    it may well be a one-off, the second time it is demonstrably not.
    """

    utterance: str
    normalized: str
    agent_id: str = ""
    tools: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)
    answer: str = ""
    hits: int = 1
    state: str = "new"  # new | learned | ignored
    intent: str | None = None
    note: str | None = None
    #: How many times the judge has looked at this request. Evidence buys another look:
    #: a row that was ignored once is re-judged when it has been heard twice as often.
    looks: int = 1
    #: What the judge decided about this request, with the probabilities behind it.
    #: Every threshold in the learner is a guess until these numbers can be compared
    #: against what actually happened, and the docs' own advice is to set them from
    #: that comparison rather than from taste.
    judgement: dict[str, Any] | None = None
    first_seen: str = field(default_factory=lambda: dt.datetime.now().isoformat(timespec="seconds"))
    last_seen: str = field(default_factory=lambda: dt.datetime.now().isoformat(timespec="seconds"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "utterance": self.utterance,
            "normalized": self.normalized,
            "agent_id": self.agent_id,
            "tools": self.tools,
            "tool_calls": self.tool_calls,
            "targets": self.targets,
            "answer": self.answer,
            "hits": self.hits,
            "state": self.state,
            "intent": self.intent,
            "note": self.note,
            "looks": self.looks,
            "judgement": self.judgement,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
        }


@dataclass(slots=True)
class LearnerData:
    """Runtime state for one config entry."""

    store: Store
    phrases: list[LearnedPhrase] = field(default_factory=list)
    intents: list[LearnedIntent] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    quarantined: list[dict[str, Any]] = field(default_factory=list)
    last_result: dict[str, Any] | None = None
    #: Diagnostics only, and deliberately not persisted.
    turns_seen: int = 0
    #: Sentences whose intent nothing answers: ours are removed at startup, and what
    #: is in a file we do not own is reported here because only its owner can fix it.
    orphans: dict[str, list[str]] = field(default_factory=dict)
    #: What the one-file-per-intent pass did on the last run.
    consolidated: dict[str, list[str]] = field(default_factory=dict)
    #: Installed intents that no longer pass the checks a new one passes: an entity
    #: renamed, a template that reads something that is not there, a sentence that no
    #: longer reads. A learning is checked when it is made and never again, which is how a
    #: rename turns a good answer into a silent wrong one.
    drift: list[dict[str, Any]] = field(default_factory=list)
    #: How often a phrase the house *already* answers locally came to the agent anyway.
    #: The only negative signal a person gives without being asked for one.
    repeats: dict[str, int] = field(default_factory=dict)
    #: What the judge has cost today, and what was learned, ignored and refused.
    spend: dict[str, Any] = field(default_factory=dict)
    #: The most recent thing learned, as a sentence a person can read. A count of learned
    #: phrases answers "how many"; this answers "what, and when", and being an entity state
    #: is what puts it in the Activity log rather than in an attribute nobody opens.
    last_learned: dict[str, Any] = field(default_factory=dict)


def language_of(entry: ConfigEntry) -> str:
    """Return the configured language."""
    return str(
        entry.options.get(CONF_LANGUAGE)
        or entry.data.get(CONF_LANGUAGE)
        or DEFAULT_LANGUAGE
    )


def slugify(value: str) -> str:
    """Return a filesystem-safe slug."""
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return value.strip("-")[:60] or "learned"


#: Optional groups ("[{speaker} ]where is my phone") and slot placeholders.
_SENTENCE_NOISE = re.compile(r"\[[^\]]*\]|\{[^}]*\}")

#: Punctuation that makes a stored sentence permanently unmatchable. hassil
#: treats these as separators, and Home Assistant strips them from the *input*
#: before matching but not from the stored sentence - so a sentence written with
#: one matches nothing, not even the identical input. Measured twice: a comma
#: ("when was my phone last seen, please"), and a question mark, which is how
#: "when was my phone last seen?" was learned, reported as learned, and then
#: silently never matched.
_BREAKING_PUNCTUATION = re.compile(r"[,;?!]+")

#: A full stop is fine mid-sentence ("3.5 degrees") but not at the end.
_TRAILING_PERIOD = re.compile(r"\.+$")


def sentence_for_storage(text: str) -> str:
    """Return a sentence as it must be written to be matchable.

    Apostrophes and hyphens stay: they occur inside words ("where's",
    "tv-guide") and match fine.
    """
    text = _BREAKING_PUNCTUATION.sub(" ", text)
    text = _TRAILING_PERIOD.sub("", text)
    return " ".join(text.split())


#: Two or more punctuation marks in a row, which is what a rewrite of an existing answer
#: produces when it joins a new clause onto the sentence the old text already ended with:
#: the house then says "…in the afternoon., the high will be 19".
_MIXED_PUNCTUATION = re.compile(r"[.!?,;:]{2,}")


def tidy_punctuation(text: str) -> str:
    """Clean up the one punctuation mistake that is a typo rather than a wording choice.

    A model asked to improve an answer sends the whole text back, and the join is where it
    slips: two finished halves, each fine, glued with a comma after the first one's full
    stop. The mechanical check can see it (`reply_problems`) but nothing could *fix* it - the
    polish pass asked the same model that had just made the mistake, and answered "no rewrite
    offered" often enough that the sloppy text was installed anyway.

    This is not a wording decision, so it does not need a model:

    * a comma next to a full stop keeps the comma when the next word continues the sentence,
      and the full stop when the next word starts a new one;
    * a repeated mark collapses to one (``!!`` -> ``!``, ``..`` -> ``.``);
    * three or more full stops are an ellipsis and are left alone.
    """
    if not text:
        return text

    def fix(match: re.Match[str]) -> str:
        marks = match.group()
        if set(marks) == {"."} and len(marks) >= 3:
            return marks  # a deliberate pause
        following = text[match.end() :].lstrip()[:1]
        if "," in marks and ("." in marks or ";" in marks):
            return "." if (not following or following.isupper()) else ","
        return marks[0]

    return _MIXED_PUNCTUATION.sub(fix, text)


#: Words that carry no topic on their own. A request made only of these is a
#: follow-up: what it means comes from the previous turn, so learning it as fixed
#: wording pins it to whatever it happened to mean once. That is how the bare
#: word "when" came to answer "your phone was last seen at 4:46 AM" forever.
#:
#: First- and second-person words are deliberately *not* here: "who am I" names
#: the speaker and stands on its own, while "when" and "when was that" do not.
_FOLLOW_UP_WORDS: frozenset[str] = frozenset(
    """
    a about again all also am an and any are as at back be both but by can could
    correct did do does doing done each either fine first for from get give go going
    had happen happened happens has have having he her here hers him his how in
    into is it its just know like made make many may me mean means meant might
    mine more most much must neither next no not now of off okay ok on once one
    last only or other our out over please previous right same s say said second
    should so some still such sure tell than that the their them then there these
    they think this those through to too told up us very was we were what when
    where which while who whom why will with would wrong yes yet
    """.split()
)


def names_nothing(text: str) -> bool:
    """Whether an utterance is only follow-up words, and so means nothing alone."""
    words = normalize_sentence(text).split()
    return bool(words) and all(word in _FOLLOW_UP_WORDS for word in words)


#: How a Home Assistant template reads an entity: the call forms, and attribute
#: access. Deliberately not a bare `word.word` pattern, so ordinary prose with a
#: full stop in it cannot be mistaken for an entity id.
_TEMPLATE_ENTITY = re.compile(
    r"""states\(\s*['"]([a-z_]+\.[a-z0-9_]+)['"]"""
    r"""|is_state\(\s*['"]([a-z_]+\.[a-z0-9_]+)['"]"""
    r"""|state_attr\(\s*['"]([a-z_]+\.[a-z0-9_]+)['"]"""
    r"""|states\.([a-z_]+\.[a-z0-9_]+)""",
    re.VERBOSE,
)


def template_entities(source: str) -> set[str]:
    """Return the entity ids a template reads, by how it reads them."""
    found: set[str] = set()
    for groups in _TEMPLATE_ENTITY.findall(source):
        found.update(value for value in groups if value)
    return found


#: "my", "mine" and friends: the phrasing is about whoever is speaking.
_FIRST_PERSON = re.compile(r"\b(?:my|mine|my own|our|ours)\b", re.IGNORECASE)


def entity_aliases(hass: HomeAssistant, entity_id: str) -> list[str]:
    """The extra names people use for an entity, as set in the entity registry.

    This is how an entity keeps the name it came with - `Area Last Seen`, `mi_rc` - and
    is still findable by the words a person actually says. Nothing here renames
    anything: an alias is additional, and the entity keeps its own name everywhere else
    in Home Assistant. Adding one is the cheap way to make a hardware-named entity
    legible to a model without taking away the name its owner recognises.
    """
    from homeassistant.helpers import entity_registry as er  # noqa: PLC0415

    row = er.async_get(hass).async_get(entity_id)
    if row is None:
        return []
    # `aliases` is a list of strings *and* a sentinel that stands for the entity's own
    # computed name, which is not an alias anyone wrote. Only strings are real.
    return sorted(alias for alias in row.aliases if isinstance(alias, str) and alias)


def entity_label(hass: HomeAssistant, entity_id: str, name: str) -> str:
    """A name, plus whatever else it is called, for a model to read."""
    aliases = entity_aliases(hass, entity_id)
    if not aliases:
        return name
    return f"{name} aka {', '.join(repr(alias) for alias in aliases)}"


def entity_owners(hass: HomeAssistant) -> dict[str, str]:
    """Map every entity on a person's devices to that person's name.

    Home Assistant already knows this: a `person` lists its `device_trackers`, and
    a tracker sits on a device. So ownership is derivable rather than guessed,
    which matters because these entities are named after hardware - Alex's watch
    is `sensor.bermuda_aa_bb_cc_dd_ee_ff_area_last_seen`, and nothing in that says
    whose it is.

    A phone typically has two devices - the app's and Bermuda's - and the person
    usually lists only one of them, so a device's name is used as a second link.
    That is what joins `sensor.sam_phone_area_last_seen` to Sam: its device
    is named "OnePlus 7 Pro", the same name the tracker he lists sits on.
    """
    from homeassistant.helpers import device_registry as dr  # noqa: PLC0415
    from homeassistant.helpers import entity_registry as er  # noqa: PLC0415

    entities = er.async_get(hass)
    devices = dr.async_get(hass)

    links: dict[str, str] = {}
    for state in hass.states.async_all("person"):
        person = state.name
        if not person:
            continue
        for tracker in state.attributes.get("device_trackers") or []:
            row = entities.async_get(tracker)
            if row is None:
                continue
            links.setdefault(tracker, person)
            if row.device_id and (device := devices.async_get(row.device_id)):
                if name := (device.name_by_user or device.name):
                    links.setdefault(f"device:{name}", person)

    owners: dict[str, str] = {}
    if not links:
        return owners
    for state in hass.states.async_all():
        if person := links.get(state.entity_id):
            owners[state.entity_id] = person
            continue
        row = entities.async_get(state.entity_id)
        if row is None or not row.device_id:
            continue
        if (device := devices.async_get(row.device_id)) is None:
            continue
        if name := (device.name_by_user or device.name):
            if person := links.get(f"device:{name}"):
                owners[state.entity_id] = person
    return owners


def owned_entities(hass: HomeAssistant, template: str) -> dict[str, str]:
    """Entities a template reads that belong to one person, and to whom."""
    owners = entity_owners(hass)
    return {
        entity_id: owners[entity_id]
        for entity_id in template_entities(template)
        if entity_id in owners
    }


def _asks_the_speaker(template: str) -> bool:
    """Whether a template consults who is asking."""
    return any(variable in template for variable in SPEAKER_VARIABLES) or (
        "voicebm_current_speaker" in template
    )


def speaker_scope_problem(
    hass: HomeAssistant, template: str, utterances: list[str]
) -> str | None:
    """Why a template cannot answer these phrasings for everyone the same way.

    Measured failure: "where is my watch" was learned against Alex's Garmin, so
    Sam asking it got Alex's watch - and the reverse, "your watch" hardcoded
    to the Amazfit, gave Alex Sam's. Both templates were checked for syntax,
    missing entities and nonsense values, and both passed.
    """
    if not any(_FIRST_PERSON.search(utterance) for utterance in utterances):
        return None
    scoped = owned_entities(hass, template)
    if not scoped or _asks_the_speaker(template):
        return None
    described = ", ".join(f"{eid} ({person}'s)" for eid, person in sorted(scoped.items()))
    return (
        "the phrasings say \"my\" but the template always reads "
        + described
        + " and never looks at who is asking, so it answers one person's device for "
        "everyone. Branch on the speaker, e.g. {% set who = (person_name or user_name "
        "or states('sensor.voicebm_current_speaker')) | lower %}"
    )


def normalize_sentence(text: str) -> str:
    """Reduce a sentence template to the literal words it always contains.

    Custom sentences may carry optional groups and slots, so comparing raw strings
    misses real collisions: ``"[{speaker} ]do you know who I am"`` and a learned
    ``"do you know who I am"`` are the same request, and whichever intent wins is
    arbitrary. Both sides are reduced before comparison so the learner refuses to
    shadow (or be shadowed by) a hand-written sentence.
    """
    text = _SENTENCE_NOISE.sub(" ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return " ".join(text.split())


def sentences_root(hass: HomeAssistant, language: str) -> Path:
    """Return the directory sentences for `language` live in."""
    return Path(hass.config.path(SENTENCES_DIR, language))


def normalize_sentence_files(hass: HomeAssistant, language: str) -> list[str]:
    """Rewrite learner-owned sentences into their matchable form.

    A sentence written with punctuation is not merely untidy, it matches
    nothing - and it would have been reported as learned. So a file written
    before that was understood is repaired here rather than left in place as a
    phrase that silently never works.
    """
    changed: list[str] = []
    root = sentences_root(hass, language)
    if not root.is_dir():
        return changed
    for path in sorted(root.glob(f"{FILE_PREFIX}*.yaml")):
        try:
            data = _load(path)
        except LearnerError:
            continue
        touched = False
        for body in (data.get("intents") or {}).values():
            for item in (body or {}).get("data") or []:
                sentences = (item or {}).get("sentences") or []
                fixed: list[str] = []
                for sentence in sentences:
                    if not isinstance(sentence, str):
                        continue
                    cleaned = sentence_for_storage(sentence)
                    if cleaned and cleaned not in fixed:
                        fixed.append(cleaned)
                if fixed != sentences:
                    item["sentences"] = fixed
                    touched = True
        if touched:
            path.write_text(yaml_util.dump(data), encoding="utf-8")
            _LOGGER.warning("Irabelle Home Learning repaired unmatchable sentences in %s", path.name)
            changed.append(path.name)
    return changed


async def async_matches_locally(
    hass: HomeAssistant, language: str, utterance: str, intent: str
) -> bool | None:
    """Ask the local matcher whether it now answers these words with this intent.

    Writing a sentence is not the same as the matcher accepting one. A stray
    question mark was stored, reported as learned, and then matched nothing - the
    phrase reached the agent on every following attempt while the learner went on
    saying it had been learned.

    Strictly best effort: ``None`` means the check could not be made, so a hiccup
    in a diagnostic never turns a good write into a reported failure.
    """
    from homeassistant.components import conversation  # noqa: PLC0415 - avoid a cycle
    from homeassistant.core import Context  # noqa: PLC0415

    try:
        agent = conversation.async_get_agent(hass, conversation.HOME_ASSISTANT_AGENT)
        recognize = getattr(agent, "async_recognize_intent", None)
        if recognize is None:
            return None
        result = await recognize(
            conversation.ConversationInput(
                text=utterance,
                context=Context(),
                conversation_id=None,
                device_id=None,
                satellite_id=None,
                language=language,
                agent_id=conversation.HOME_ASSISTANT_AGENT,
            )
        )
    except Exception:  # noqa: BLE001 - a diagnostic, not a gate
        _LOGGER.debug(
            "Irabelle Home Learning could not verify %r against the matcher", utterance, exc_info=True
        )
        return None
    if result is None:
        return False
    return str(getattr(result.intent, "name", "")) == intent


async def async_verify(
    hass: HomeAssistant, language: str, utterances: list[str], intent: str
) -> str:
    """Return "ok", "unmatched" or "unknown" for a set of just-learned sentences."""
    if not utterances:
        return "unknown"
    verdicts = [
        await async_matches_locally(hass, language, utterance, intent)
        for utterance in utterances
    ]
    if False in verdicts:
        for utterance, verdict in zip(utterances, verdicts, strict=True):
            if verdict is False:
                _LOGGER.warning(
                    "Irabelle Home Learning stored %r for %s but the local matcher does not "
                    "accept it - the words will keep reaching the agent",
                    utterance,
                    intent,
                )
        return "unmatched"
    return "unknown" if all(verdict is None for verdict in verdicts) else "ok"


def learned_phrases(hass: HomeAssistant, language: str) -> dict[str, list[str]]:
    """Every phrasing this integration wrote, keyed by the intent it answers.

    Read from the sentence files rather than the provenance store. A learned
    intent's sentences are written for it and never become phrase rows, and
    forgetting needs the *stored* wording - `async_forget` matches the sentence,
    not a paraphrase of it.
    """
    phrases: dict[str, list[str]] = {}
    root = sentences_root(hass, language)
    if not root.is_dir():
        return phrases
    for path in sorted(root.glob(f"{FILE_PREFIX}*.yaml")):
        try:
            data = _load(path)
        except LearnerError:
            continue
        for intent, body in (data.get("intents") or {}).items():
            for item in (body or {}).get("data") or []:
                for sentence in (item or {}).get("sentences") or []:
                    if isinstance(sentence, str) and sentence.strip():
                        phrases.setdefault(str(intent), []).append(sentence.strip())
    return {intent: sorted(set(values)) for intent, values in phrases.items()}


#: The suffix a learner file carries while this integration is not loaded. A sentence whose
#: intent nothing answers does not fail quietly - recognition raises for that phrase, so it gets
#: an error instead of an answer - and every sentence this integration wrote becomes exactly
#: that the moment its handlers go. So the files step aside on unload, and come back on setup.
DISARMED_SUFFIX = ".off"


def disarmed_files(hass: HomeAssistant, language: str) -> list[Path]:
    """The learner files this integration has stepped aside."""
    root = sentences_root(hass, language)
    if not root.is_dir():
        return []
    return sorted(root.glob(f"{FILE_PREFIX}*.yaml{DISARMED_SUFFIX}"))


def disarm_learner_files(
    hass: HomeAssistant, language: str, still_registered: set[str] | None = None
) -> list[str]:
    """Rename learner sentence files out of the matcher's reach, unless something still answers.

    Called on unload, which is what removing or reloading the integration does. The text is
    untouched and the file still reads as ordinary YAML; it simply stops being loaded, so the
    phrases it answers fall through to the conversation agent instead of failing recognition.

    ``still_registered`` is every intent that has a handler *after* this integration released
    its own. A file whose intents are all in there is left alone: an `intent_script` stub that
    the user added through `export` answers it, and stepping the sentences aside would break
    exactly the phrases the stub was written to keep working.
    """
    keep = still_registered or set()
    renamed: list[str] = []
    for path in sorted(sentences_root(hass, language).glob(f"{FILE_PREFIX}*.yaml")):
        try:
            content = _load(path)
        except LearnerError:
            content = None
        names = set((content.get("intents") or {})) if content else set()
        if names and names <= keep:
            continue
        target = path.with_name(path.name + DISARMED_SUFFIX)
        try:
            path.rename(target)
            renamed.append(target.name)
        except OSError as err:  # pragma: no cover - filesystem trouble
            _LOGGER.error("Could not disarm %s: %s", path.name, err)
    if renamed:
        _LOGGER.warning(
            "Irabelle Home Learning stepped %d sentence file(s) aside while it is unloaded; they are "
            "still there as *.yaml%s and come back when it is set up again",
            len(renamed),
            DISARMED_SUFFIX,
        )
    return renamed


def adopt_disarmed(hass: HomeAssistant, language: str) -> list[LearnedIntent]:
    """Rebuild learned intents from the sentence files themselves.

    The file is the whole intent - sentences, slot lists, the reply and the template behind
    it - so a house whose store is gone can be handed its learnings back: a fresh install, an
    integration removed and added again, or a `custom_sentences` folder copied from another
    installation. Nothing is invented: an intent with no reply is skipped, because it would
    not answer, and its file is left aside rather than armed into a failure.
    """
    adopted: list[LearnedIntent] = []
    for path in disarmed_files(hass, language):
        try:
            content = _load(path)
        except LearnerError:
            continue
        lists = content.get("lists") or {}
        responses = (content.get("responses") or {}).get("intents") or {}
        for name, body in (content.get("intents") or {}).items():
            if not (body or {}).get("data"):
                continue
            reply = (responses.get(name) or {}).get("default")
            if not isinstance(reply, str) or not reply.strip():
                continue
            reply = visible_reply(reply)
            template = (responses.get(name) or {}).get("template") or reply
            adopted.append(
                LearnedIntent(
                    name=str(name),
                    template=str(template),
                    response=str(reply),
                    slots=dict(lists) or None,
                    note="adopted from its own sentence file",
                    source="file",
                    file=f"{SENTENCES_DIR}/{language}/{path.name[: -len(DISARMED_SUFFIX)]}",
                )
            )
    return adopted


def arm_learner_files(
    hass: HomeAssistant, language: str, known: set[str]
) -> list[str]:
    """Rename disarmed files back, but only the ones an intent now answers.

    A file whose intents are not all registered is left aside: arming it would put a sentence
    back in front of the matcher with nothing to answer it, which is the failure this whole
    mechanism exists to avoid.
    """
    armed: list[str] = []
    for path in disarmed_files(hass, language):
        try:
            content = _load(path)
        except LearnerError:
            continue
        names = set((content.get("intents") or {}).keys())
        if not names or not names <= known:
            continue
        target = path.with_name(path.name[: -len(DISARMED_SUFFIX)])
        try:
            path.rename(target)
            armed.append(target.name)
        except OSError as err:  # pragma: no cover - filesystem trouble
            _LOGGER.error("Could not put %s back: %s", path.name, err)
    if armed:
        _LOGGER.warning("Irabelle Home Learning put %d sentence file(s) back", len(armed))
    return armed


def purge_sentence_files(hass: HomeAssistant, language: str) -> list[str]:
    """Delete every sentence file this integration owns, and nothing else.

    Hand-written files (`find_phone.yaml`, `timers.yaml`, ...) are never touched:
    the integration only ever creates and deletes `learned-*`.
    """
    removed: list[str] = []
    root = sentences_root(hass, language)
    if not root.is_dir():
        return removed
    for path in sorted(root.glob(f"{FILE_PREFIX}*.yaml")):
        try:
            path.unlink()
            removed.append(path.name)
        except OSError as err:  # pragma: no cover - filesystem trouble
            _LOGGER.error("Could not remove %s: %s", path, err)
    return removed


def _load(path: Path) -> dict[str, Any]:
    """Load one sentence file, tolerating an empty file."""
    try:
        data = yaml_util.load_yaml(path)
    except Exception as err:  # noqa: BLE001 - any parse failure means "broken file"
        raise LearnerError(f"{path.name} does not parse: {err}") from err
    return data if isinstance(data, dict) else {}


async def async_intent_names(hass: HomeAssistant) -> set[str]:
    """Return every intent a sentence may target.

    The intent registry is the authoritative source: a sentence bound to an
    intent that is not registered is silently useless, which is exactly what we
    want to refuse.
    """
    names = set(BUILTIN_INTENTS)
    for handler in intent_helper.async_get(hass):
        names.add(handler.intent_type)
    return names


def remove_intent_sentences(hass: HomeAssistant, language: str, name: str) -> list[str]:
    """Remove every sentence written for one intent, from every file we wrote.

    The intent's own file is the obvious one, and it is not the only one: a phrasing
    *bound* to the intent lives in whichever file the binding was written to. Removing
    only the intent's file is how a sentence outlives its handler, and a sentence with no
    handler is worse than no sentence at all - see `async_prune_orphans`.
    """
    root = sentences_root(hass, language)
    if not root.is_dir():
        return []
    touched: list[str] = []
    for path in sorted(root.glob(f"{FILE_PREFIX}*.yaml")):
        try:
            content = _load(path)
        except LearnerError:
            continue
        intents = content.get("intents") or {}
        if name not in intents:
            continue
        intents.pop(name, None)
        touched.append(path.name)
        if intents:
            path.write_text(yaml_util.dump(content), encoding="utf-8")
        else:
            path.unlink(missing_ok=True)
    return touched


async def async_prune_orphans(
    hass: HomeAssistant, entry: ConfigEntry
) -> dict[str, list[str]]:
    """Find sentences whose intent nothing answers, and remove the ones we wrote.

    A sentence with no handler does not fail quietly: the matcher reports a match, nothing
    handles it, and the pipeline answers

        intent-failed: Unexpected error during intent recognition

    for that phrase - one leftover takes voice down, and its own phrase gets no answer at
    all rather than reaching an agent that could handle it.

    Two things this must not get wrong, both learned the hard way:

    * **Only after every integration has loaded.** `intent_script` registers its handlers
      after this integration is set up, so a prune that runs during startup calls real
      intents orphans and deletes their wording. The caller waits for
      `EVENT_HOMEASSISTANT_STARTED`.
    * **Never a name another file also declares.** If a sentence file this integration did
      not write names the same intent, the intent belongs to whoever wrote that file, and
      our copy is a binding to it. Removing it because its handler is not visible *yet* is
      how "who am I" lost its phrasings once already.
    """
    known = await async_intent_names(hass)
    root = sentences_root(hass, language_of(entry))
    removed, reported = await async_run(hass, _prune_files, root, known)

    if removed:
        _LOGGER.warning(
            "Irabelle Home Learning removed %d sentence(s) for intents nothing answers, which would "
            "have failed the whole pipeline: %s",
            len(removed),
            ", ".join(removed),
        )
    if reported:
        _LOGGER.warning(
            "Irabelle Home Learning found sentences for intents nothing answers in files it does not "
            "own, so it left them alone. These fail the pipeline until they are fixed: %s",
            ", ".join(reported),
        )
    return {"removed": removed, "reported": reported}


def _prune_files(root: Path, known: set[str]) -> tuple[list[str], list[str]]:
    """Remove our sentences for intents nothing answers. Returns (removed, reported)."""
    removed: list[str] = []
    reported: list[str] = []
    if not root.is_dir():
        return removed, reported

    ours: list[tuple[Path, dict[str, Any]]] = []
    declared: set[str] = set()
    for path in sorted(root.glob("*.yaml")):
        try:
            content = _load(path)
        except LearnerError:
            continue
        if path.name.startswith(FILE_PREFIX):
            ours.append((path, content))
        else:
            declared.update((content.get("intents") or {}).keys())

    for path, content in ours:
        intents = content.get("intents") or {}
        stale = [
            name for name in intents if name not in known and name not in declared
        ]
        if not stale:
            continue
        for name in stale:
            intents.pop(name, None)
            removed.append(f"{name} ({path.name})")
        if intents:
            path.write_text(yaml_util.dump(content), encoding="utf-8")
        else:
            path.unlink(missing_ok=True)

    # Anything left that nothing answers is in a file this integration does not own. It
    # cannot be fixed from here, so it is named instead.
    for name in sorted(declared):
        if name not in known:
            reported.append(name)
    return removed, reported


#: One slot binding, exactly as this component writes it, so it can be recognised again. The
#: first form is what was written before the bug below was found, and is still recognised so
#: existing replies are upgraded rather than left with it.
_BINDING = re.compile(
    r"\{%\s*set\s+(?P<name>[a-z][a-z0-9_]*)\s*=\s*(?:"
    r"slots\.(?P=name)\s+or\s+''"
    r"|slots\.(?P=name)\s+if\s+slots\s+is\s+defined\s+and\s+slots\.(?P=name)\s+is\s+"
    r"defined\s+else\s+(?P=name)"
    r")\s*%\}"
)


def bind_slots(reply: str, slots: Iterable[str]) -> str:
    """Bind the slot names a reply reads without `slots.`, so Home Assistant can render it.

    A learned reply is stored in `responses:`, and Home Assistant renders that template with
    `slots` - the values the recogniser matched - in scope. Every reply written before this
    existed reads the bare name instead (`{{ day }}`), because this component's own handler
    also defines each slot as a variable of its own. Once the component is gone only `slots`
    remains, and a template asking for the bare name does not fail loudly: it renders empty,
    or takes its "no day was given" branch, and answers about the wrong day.

    So the binding is written into the reply itself, which is the one text both renderers
    agree on:

        {% set day = slots.day if slots is defined and slots.day is defined else day %}

    The value when the slot is there is the one the handler was using all along. When it is
    *not* there, the name is left **undefined** rather than set to empty - and that is the
    whole reason for this shape. A reply is allowed to ask whether a slot was given
    (`{% if who is defined %}` picks between "Sam's phone" and "your phone"), and an empty
    string answers that question wrongly: measured on a real answer, `PhoneLastSeenAgo`
    stopped reaching its speaker branch, so the house answered the wrong person's phone and
    greeted them as "Unknown". An undefined name is falsy, prints as empty, and keeps
    `is defined` honest.
    """
    text = visible_reply(reply or "")
    wanted = [
        name
        for name in sorted(set(slots or ()))
        if re.search(rf"(?<![\w.]){re.escape(name)}\b", text)
    ]
    if not wanted:
        return text
    bindings = "".join(
        "{% set " + name + " = slots." + name + " if slots is defined and slots." + name
        + " is defined else " + name + " %}"
        for name in wanted
    )
    return bindings + text


def visible_reply(reply: str) -> str:
    """The reply without its bindings: the text as a person or a model should read it."""
    text = reply or ""
    while match := _BINDING.match(text):
        text = text[match.end() :]
    return text


def file_responses(hass: HomeAssistant, language: str) -> dict[str, str]:
    """Every reply written into a learner-owned sentence file, by intent name.

    The sentence file is where a reply belongs: it is the file a person reads to find out
    what the house will say, and the one this component rewrites when asked to answer
    differently. The store keeps no copy of the text at all - what a learned intent says and
    the template behind it live in this folder, so removing the component leaves the intents
    whole rather than leaving a store nobody can read.

    The slot bindings (`bind_slots`) are stripped here, so everything that reads a reply -
    the handler, the prompts, the checks, the audit - works on the same readable text that
    was written, and nothing has to know the bindings exist.
    """
    found: dict[str, str] = {}
    root = sentences_root(hass, language)
    if not root.is_dir():
        return found
    for path in sorted(root.glob(f"{FILE_PREFIX}*.yaml")):
        try:
            data = _load(path)
        except LearnerError:
            continue
        for name, body in ((data.get("responses") or {}).get("intents") or {}).items():
            reply = (body or {}).get("default")
            if isinstance(reply, str) and reply.strip():
                found[str(name)] = visible_reply(reply)
    return found


def file_templates(hass: HomeAssistant, language: str) -> dict[str, str]:
    """Every computation written beside a reply, by intent name.

    The `template` key next to `default` in a learner file is one this component added:
    Home Assistant ignores a response key it never asks for, and a learned intent often says
    something a person asked for while the computation that builds it is a different text.
    Keeping it in the file means the intent is whole on disk.
    """
    found: dict[str, str] = {}
    root = sentences_root(hass, language)
    if not root.is_dir():
        return found
    for path in sorted(root.glob(f"{FILE_PREFIX}*.yaml")):
        try:
            data = _load(path)
        except LearnerError:
            continue
        for name, body in ((data.get("responses") or {}).get("intents") or {}).items():
            template = (body or {}).get("template")
            if isinstance(template, str) and template.strip():
                found[str(name)] = template
    return found


def sentence_index(hass: HomeAssistant, language: str) -> dict[str, str]:
    """Map every existing utterance -> the intent it is bound to."""
    index: dict[str, str] = {}
    root = sentences_root(hass, language)
    if not root.is_dir():
        return index
    for path in sorted(root.glob("*.yaml")):
        try:
            data = _load(path)
        except LearnerError:
            continue
        for intent, body in (data.get("intents") or {}).items():
            for entry in ((body or {}).get("data") or []):
                for sentence in (entry or {}).get("sentences") or []:
                    if isinstance(sentence, str):
                        normalized = normalize_sentence(sentence)
                        if normalized:
                            index.setdefault(normalized, str(intent))
    return index


def quarantine_broken(hass: HomeAssistant, language: str) -> list[dict[str, Any]]:
    """Move aside any sentence file that does not parse.

    Returns the list of quarantined files, so the caller can surface it. This
    runs before every reload: a broken file must never reach the conversation
    integration.
    """
    quarantined: list[dict[str, Any]] = []
    root = sentences_root(hass, language)
    if not root.is_dir():
        return quarantined

    target_dir = Path(hass.config.path(QUARANTINE_DIR))
    for path in sorted(root.glob("*.yaml")):
        try:
            _load(path)
            continue
        except LearnerError as err:
            reason = str(err)
        stamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{path.stem}-{stamp}.yaml"
        try:
            path.replace(target)
        except OSError as err:  # pragma: no cover - filesystem trouble
            _LOGGER.error("Could not quarantine %s: %s", path, err)
            continue
        _LOGGER.warning("Quarantined unparseable sentence file %s: %s", path.name, reason)
        quarantined.append({"file": path.name, "moved_to": str(target), "error": reason})
    return quarantined


def _validate(
    utterances: list[str],
    intent: str,
    index: dict[str, str],
    known_intents: set[str],
    declared: set[str] | None = None,
) -> None:
    """Raise LearnerError if the proposal must not be installed.

    ``declared`` is the set of list slots the same proposal writes under `lists:`. A
    `{name}` reference is allowed only for one of those: a sentence naming a slot that no
    list defines stops sentence matching for *every* agent, so it is refused rather than
    stored, while a declared slot is how one phrasing covers every value of a word-class
    (any weekday, any room) instead of one intent per value.
    """
    if not utterances:
        raise LearnerError("no utterances given")
    if len(utterances) > MAX_UTTERANCES:
        raise LearnerError(f"too many utterances ({len(utterances)} > {MAX_UTTERANCES})")
    for utterance in utterances:
        if not utterance:
            raise LearnerError("an utterance is empty")
        if len(utterance) > MAX_UTTERANCE_CHARS:
            raise LearnerError(f"utterance too long: {utterance[:40]!r}...")
        checked = utterance
        if declared is not None:
            for reference in re.findall(r"\{([^{}]*)\}", checked):
                if reference.strip() not in declared:
                    raise LearnerError(
                        f"{utterance!r} refers to the slot "
                        f"{reference.strip()!r}, which this proposal does not declare. "
                        "Declare its values in `slots` (for example day = today, "
                        "tomorrow, monday) or write the wording out in full."
                    )
            checked = re.sub(r"\{[^{}]*\}", " ", checked)
        if any(char in checked for char in "{}[]"):
            # Brackets are an optional group, and a brace that survived the check above
            # is a malformed slot reference. Storing either one breaks sentence matching
            # for every agent, and speech never produces either, so a sentence carrying
            # one is a model writing the wrong kind of thing.
            raise LearnerError(
                f"{utterance!r} contains braces or brackets, which the sentence "
                "matcher reads as a slot or an optional group; learned sentences "
                "are fixed wording (a declared {slot} is the one exception)"
            )
    if intent not in known_intents:
        raise LearnerError(
            f"unknown intent {intent!r}; it must already be registered "
            f"(a built-in intent, or one declared in intent_script.yaml)"
        )
    for utterance in utterances:
        existing = index.get(normalize_sentence(utterance))
        if existing is not None and existing != intent:
            raise LearnerError(
                f"{utterance!r} is already answered by {existing!r} "
                f"(including its hand-written sentence templates); refusing to shadow it"
            )


def _merge_into_file(
    path: Path,
    intent: str,
    utterances: list[str],
    slots: dict | None,
    *,
    response_key: str | None = None,
    response: str | None = None,
) -> list[str]:
    """Merge utterances into a learner-owned file, returning what was added.

    ``response_key`` ties one group of sentences to its own reply. Two pinned groups of the
    same intent carry different values - "warm" is 2700 K and "cold" 4000 K - so they need
    two replies, or the second one inherits the first one's words and says the light was set
    to warm when it was set to cold.
    """
    data: dict[str, Any] = {"language": "en", "intents": {}}
    if path.exists():
        try:
            data = _load(path) or data
        except LearnerError:
            data = {"language": "en", "intents": {}}
    data.setdefault("language", "en")
    data.setdefault("intents", {})

    body = data["intents"].setdefault(intent, {"data": []})
    body.setdefault("data", [])

    present = {
        normalize_sentence(s)
        for item in body["data"]
        for s in (item or {}).get("sentences") or []
        if isinstance(s, str)
    }
    fresh = [
        sentence_for_storage(u)
        for u in utterances
        if normalize_sentence(u) not in present
    ]
    # Stripping punctuation can collapse two utterances onto the same sentence.
    seen: set[str] = set()
    fresh = [u for u in fresh if u and not (u.lower() in seen or seen.add(u.lower()))]
    if not fresh:
        return []

    target = None
    for item in body["data"]:
        if (item or {}).get("slots") == (slots or None):
            target = item
            break
    if target is None:
        target = {"sentences": []}
        if slots:
            target["slots"] = slots
        if response_key:
            target["response"] = response_key
        body["data"].append(target)
    elif response_key:
        # Always the key derived from the values, so a group re-learned under a better name
        # does not keep pointing at a reply that was renamed away from it.
        target["response"] = response_key
    target.setdefault("sentences", []).extend(fresh)

    if response_key and response:
        from .intents import _intent_slots  # noqa: PLC0415 - intents imports this module

        replies = (
            data.setdefault("responses", {})
            .setdefault("intents", {})
            .setdefault(intent, {})
        )
        if not replies.get(response_key):
            replies[response_key] = bind_slots(response, _intent_slots(data, intent))
        referenced = {
            str(item.get("response"))
            for item in body["data"]
            if isinstance(item, dict) and item.get("response")
        }
        for stale in [key for key in replies if key not in referenced]:
            # A reply no group points at any more is left over from an earlier naming of the
            # same values; keeping it would be a dead key in a file people read.
            replies.pop(stale)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml_util.dump(data), encoding="utf-8")
    return fresh


#: The judge's counters at the last save, per entry. Its own counters reset on restart, so
#: today's total is kept as a running sum of the differences rather than as a reading.
_SPEND_SEEN: dict[str, tuple[int, int, int]] = {}


def record_spend(entry: ConfigEntry) -> dict[str, Any]:
    """Fold the judge's counters into today's total and return it.

    The judge counts since the last restart, which is the wrong unit for a question like
    "what has this cost me today". Counting the difference at every save makes the total
    survive restarts without a timer, and the estimate is only as good as the price in
    `JEV_PRICE_PER_MTOK` - which is why it is reported as an estimate.
    """
    from .judge import usage_of  # noqa: PLC0415 - judge imports only const, so no cycle

    data: LearnerData = entry.runtime_data
    usage = usage_of(entry)
    seen = _SPEND_SEEN.get(entry.entry_id, (0, 0, 0))
    current = (usage.calls, usage.input_tokens, usage.output_tokens)
    delta = [max(0, now - before) for now, before in zip(current, seen)]
    _SPEND_SEEN[entry.entry_id] = current

    today = dt.date.today().isoformat()
    spend = dict(data.spend) if data.spend.get("day") == today else {}
    spend["day"] = today
    spend["calls"] = int(spend.get("calls", 0)) + delta[0]
    spend["input_tokens"] = int(spend.get("input_tokens", 0)) + delta[1]
    spend["output_tokens"] = int(spend.get("output_tokens", 0)) + delta[2]
    spend["estimated_cost"] = round(
        spend["input_tokens"] / 1_000_000 * JEV_PRICE_PER_MTOK, 6
    )
    for state in ("learned", "ignored"):
        spend[f"{state}_today"] = sum(
            1
            for row in data.observations
            if row.state == state and (row.last_seen or "").startswith(today)
        )
    spend["still_new"] = sum(1 for row in data.observations if row.state == "new")
    data.spend = spend
    return spend


def note_learned(
    data: LearnerData, phrase: str, intent: str, how: str, **extra: Any
) -> None:
    """Record what was learned, for the entity state the Activity log reads."""
    data.last_learned = {
        "phrase": phrase,
        "intent": intent,
        "how": how,
        "at": dt.datetime.now().isoformat(timespec="seconds"),
        **extra,
    }


async def async_save(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Persist the provenance store."""
    data: LearnerData = entry.runtime_data
    record_spend(entry)
    await data.store.async_save(
        {
            "phrases": [p.as_dict() for p in data.phrases],
            "intents": [i.as_dict() for i in data.intents],
            "observations": [o.as_dict() for o in data.observations],
            "repeats": data.repeats,
            "spend": data.spend,
            "last_learned": data.last_learned,
        }
    )


async def async_load(hass: HomeAssistant, entry: ConfigEntry) -> LearnerData:
    """Load (or create) the provenance store for an entry."""
    store: Store = Store(hass, STORAGE_VERSION, f"{STORAGE_KEY}.{entry.entry_id}")
    stored = await store.async_load() or {}
    phrases = [
        LearnedPhrase(**row)
        for row in stored.get("phrases", [])
        if isinstance(row, dict) and row.get("utterance") and row.get("intent")
    ]
    intents = [
        LearnedIntent(**row)
        for row in stored.get("intents", [])
        if isinstance(row, dict) and row.get("name") and row.get("template")
    ]
    observations = [
        Observation(**row)
        for row in stored.get("observations", [])
        if isinstance(row, dict) and row.get("utterance")
    ]
    # The text comes from the sentence file, not the store: a reply a person edited there is
    # the one the house says, and the store's copy of it is only what was written last time.
    # This is what makes the folder - not the storage file - the thing that survives.
    replies = await async_run(hass, file_responses, hass, language_of(entry))
    templates = await async_run(hass, file_templates, hass, language_of(entry))
    for spec in intents:
        if reply := replies.get(spec.name):
            # The reply IS the template unless the file says otherwise. A learned intent's
            # answer is a Jinja text, so an older store row holding a different one is a
            # leftover from an earlier wording - not a computation worth keeping, and not
            # something to promote back into the file as if it were the logic.
            spec.template = templates.get(spec.name, reply)
            spec.response = reply
        elif template := templates.get(spec.name):
            spec.template = template
    return LearnerData(
        store=store,
        phrases=phrases,
        intents=intents,
        observations=observations,
        # These three are written by `async_save`, so they have to be read back here or
        # they reset with every restart - which is what made the spend counter and the
        # last-learned sensor read as empty right after a restart.
        repeats=dict(stored.get("repeats") or {}),
        spend=dict(stored.get("spend") or {}),
        last_learned=dict(stored.get("last_learned") or {}),
    )


def auto_learn(entry: ConfigEntry) -> bool:
    """Whether this entry watches conversations and learns by itself."""
    value = entry.options.get(CONF_AUTO_LEARN, entry.data.get(CONF_AUTO_LEARN))
    return DEFAULT_AUTO_LEARN if value is None else bool(value)


def min_occurrences(entry: ConfigEntry) -> int:
    """How many times a request must be heard before it is generalised."""
    value = entry.options.get(CONF_MIN_OCCURRENCES, entry.data.get(CONF_MIN_OCCURRENCES))
    try:
        number = int(value) if value is not None else DEFAULT_MIN_OCCURRENCES
    except (TypeError, ValueError):
        return DEFAULT_MIN_OCCURRENCES
    return max(1, min(20, number))


def ai_task_entity(entry: ConfigEntry) -> str | None:
    """The configured `ai_task.*` entity, if one was named."""
    from .agents import settings  # noqa: PLC0415 - agents imports nothing of ours

    value = settings(entry).get(CONF_AI_TASK_ENTITY)
    text = str(value).strip() if value else ""
    return text or None


def record_observation(data: LearnerData, row: Observation) -> Observation:
    """Fold a fresh turn into the observation window, returning the live row.

    Repeat phrasings are counted on the row that already exists - that count is
    the whole reason generalisation stays cheap.
    """
    for existing in data.observations:
        if existing.normalized == row.normalized:
            existing.hits += 1
            existing.last_seen = row.last_seen
            existing.agent_id = row.agent_id or existing.agent_id
            existing.answer = row.answer or existing.answer
            for tool in row.tools:
                if tool not in existing.tools:
                    existing.tools.append(tool)
            for call in row.tool_calls:
                if call not in existing.tool_calls:
                    existing.tool_calls.append(call)
            for target in row.targets:
                if target not in existing.targets:
                    existing.targets.append(target)
            if row.utterance and len(row.utterance) < len(existing.utterance or " " * 999):
                existing.utterance = row.utterance
            if existing.state == "learned":
                # It came back: whatever we installed does not answer it.
                existing.state = "new"
                existing.intent = None
            return existing
    data.observations.append(row)
    if len(data.observations) > MAX_OBSERVATIONS:
        del data.observations[: len(data.observations) - MAX_OBSERVATIONS]
    return row


def allowed_platforms(entry: ConfigEntry) -> list[str]:
    """Integration domains allowed to teach this house (empty = all)."""
    value = entry.options.get(CONF_ALLOWED_PLATFORMS, entry.data.get(CONF_ALLOWED_PLATFORMS))
    if value is None:
        return list(DEFAULT_ALLOWED_PLATFORMS)
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(item) for item in value]


async def async_learn(
    hass: HomeAssistant,
    entry: ConfigEntry,
    *,
    utterances: list[str],
    intent: str,
    slots: dict[str, Any] | None = None,
    note: str | None = None,
    source: str | None = None,
    response: str | None = None,
) -> dict[str, Any]:
    """Validate and install a phrasing, then reload conversation."""
    data: LearnerData = entry.runtime_data
    language = language_of(entry)

    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in utterances:
        text = str(raw).strip()
        if text and text.lower() not in seen:
            seen.add(text.lower())
            cleaned.append(text)

    index = await async_run(hass, sentence_index, hass, language)
    known = await async_intent_names(hass)
    _validate(cleaned, intent, index, known)
    if impossible := _values_the_intent_needs(
        language, intent, cleaned, data.intents, set(slots or {})
    ):
        raise LearnerError(impossible)

    # A learned intent is only as good as its template, and a template that
    # ignores who is asking cannot answer "my ..." for two people.
    for spec in data.intents:
        if spec.name != intent:
            continue
        if problem := speaker_scope_problem(hass, spec.template, cleaned):
            raise LearnerError(problem)

    # Anything the matcher already answers needs no new sentence, wherever the
    # existing copy lives - otherwise every call re-adds it to a new file.
    fresh = [u for u in cleaned if normalize_sentence(u) not in index]
    if not fresh:
        result = {
            "intent": intent,
            "added": [],
            "already_present": cleaned,
            "note": "already answered locally; nothing to install",
        }
        data.last_result = result
        async_dispatcher_send(hass, SIGNAL_UPDATED)
        return result

    # A phrase bound to a learned intent goes in that intent's own file, beside the rest
    # of its wording and its reply. Writing it to a file named after the note instead is
    # how one intent's sentences ended up scattered over three files, with the reply in
    # whichever of them happened to be the intent's.
    existing = next((spec for spec in data.intents if spec.name == intent), None)
    if existing is not None and existing.file:
        relative = existing.file
    else:
        # Built-in intents have no learned spec, so the file is named after the intent
        # itself. A note is a sentence ("migrated", "no idea where my keys are"), and a
        # file named after one says nothing about what is inside it or which intent it
        # belongs to.
        slug = slugify(intent)
        relative = f"{SENTENCES_DIR}/{language}/{FILE_PREFIX}{slug}.yaml"
    path = Path(hass.config.path(relative))
    # A phrase bound to a *built-in* intent has no reply of ours, and Home Assistant's own
    # answer for that intent only exists for the sentence groups it shipped. Without one the
    # intent acts and the house says nothing at all - measured: `action_done`, empty speech,
    # silence. The reply the person actually heard on the turn being learned is the right one
    # to keep, and it belongs to *that group*: two pinned groups of one intent carry different
    # values and must not share a reply.
    reply_key = _response_key(intent, slots) if (response and existing is None) else None
    added = await async_run(
        hass,
        _merge_into_file,
        path,
        intent,
        fresh,
        slots,
        response_key=reply_key,
        response=response if reply_key else None,
    )

    for utterance in added:
        data.phrases.append(
            LearnedPhrase(
                utterance=utterance,
                intent=intent,
                slots=slots or None,
                note=note,
                source=source,
                file=relative,
            )
        )
    if added:
        # Before the save, not after: this is the only save on this path, so a record
        # written afterwards lives in memory until something else happens to save - which
        # is how "last learned" read as "nothing yet" after every restart.
        note_learned(
            data,
            added[0],
            intent,
            f"bound {len(added)} phrasing(s) to {intent}",
            file=relative,
        )
        data.quarantined = await async_run(hass, quarantine_broken, hass, language)
        await async_save(hass, entry)
        await hass.services.async_call("conversation", "reload", {}, blocking=True)

    result = {
        "intent": intent,
        "file": relative,
        "added": added,
        "already_present": [u for u in cleaned if u not in added],
        "slots": slots or None,
        "quarantined": data.quarantined,
        "verified": await async_verify(hass, language, added, intent),
    }
    if warnings := _slots_the_intent_ignores(language, intent, added, data.intents):
        result["warnings"] = warnings
        _LOGGER.warning(
            "Irabelle Home Learning bound %s to %s, but %s",
            ", ".join(repr(text) for text in added),
            intent,
            "; ".join(warnings),
        )
    data.last_result = result
    async_dispatcher_send(hass, SIGNAL_UPDATED)
    return result


def _response_key(intent: str, slots: dict[str, Any] | None) -> str:
    """A stable name for one pinned group's reply, derived from its values.

    Derived from the values, so re-learning the same group reuses its reply. Named after the
    values rather than after this component, because the key is Home Assistant's own
    mechanism - `response:` on a sentence group, `responses: intents:` for the text - and the
    file has to keep working when this integration is gone. A key called `irabelle_home_learning_...`
    reads like a dependency and is not one.
    """
    words = "_".join(str(value) for _, value in sorted((slots or {}).items()))
    readable = re.sub(r"[^a-z0-9]+", "_", words.lower()).strip("_")[:32]
    digest = hashlib.sha1(
        repr(sorted((slots or {}).items())).encode("utf-8")
    ).hexdigest()[:6]
    return f"{readable}_{digest}" if readable else f"reply_{digest}"


def _values_the_intent_needs(
    language: str,
    intent: str,
    sentences: list[str],
    learned: list[Any],
    supplied: set[str] | None = None,
) -> str | None:
    """Why this phrasing cannot drive this intent, when it names a value and supplies none.

    Measured on the live house: the assistant taught the exact words "set the office colour
    temperature to warm" to `HassLightSet`. The words carry a temperature, but as *fixed
    wording* rather than a slot, so the intent matched, called `light.turn_on` with no
    temperature, changed nothing, and returned a response with no speech at all - the person
    asked twice and heard silence.

    Every one of `HassLightSet`'s own sentences fills a value slot (brightness, temperature or
    colour); a phrasing bound to it that fills none is a phrase that matches and does nothing.
    That is worse than not learning it, so it is refused with the two ways to do it properly.
    """
    if any(getattr(spec, "name", None) == intent for spec in learned):
        return None
    block = (upstream_bundle(language).get("intents") or {}).get(intent)
    if not block:
        return None
    groups = [block, *(block.get("data") or [])]
    value_slots: set[str] = set()
    every_sentence_passes_one = True
    for group in groups:
        slots = set(group.get("slots") or {})
        for sentence in group.get("sentences") or []:
            used = {slot_name(match) for match in _SENTENCE_SLOT.findall(str(sentence))}
            value = (used | slots) - set(PROVIDED_SLOTS)
            value_slots |= value
            if not value:
                every_sentence_passes_one = False
    if not value_slots or not every_sentence_passes_one:
        return None
    mine = {slot_name(match) for match in _SENTENCE_SLOT.findall(" ".join(sentences))}
    # A value is supplied either as a `{slot}` in the words or as a pinned `slots:` value on
    # the sentence's group, and both are ways of doing it properly.
    if (mine | set(supplied or ())) & value_slots:
        return None
    return (
        f"{intent} always acts on a value ({', '.join(sorted(value_slots))}), and this "
        "phrasing supplies none of them, so it would match and do nothing at all. Pin the "
        "value on the sentence's group (slots: {temperature: 2700}), or write it as a {slot} "
        "whose values the intent accepts - a number for temperature, or a list that maps the "
        "words to numbers"
    )


def _slots_the_intent_ignores(
    language: str, intent: str, sentences: list[str], learned: list[Any]
) -> list[str]:
    """A slot a built-in intent's handler does not read: understood, and then thrown away.

    Home Assistant's own sentence files for the intent are the authority on the names its
    handler reads (`HassLightSet` reads `brightness`, so a phrasing that says `{level}` fills
    a slot nobody looks at). A learned intent is its own authority, so it is left alone, and
    an intent with no bundled sentences is left alone too - nothing known is not the same as
    nothing allowed.
    """
    if any(getattr(spec, "name", None) == intent for spec in learned):
        return []
    known = expected_slots(language, intent)
    if not known:
        return []
    used: set[str] = set()
    for sentence in sentences:
        used.update(slot_name(match) for match in _SENTENCE_SLOT.findall(str(sentence)))
    unknown = sorted(name for name in used if name and name not in known)
    if not unknown:
        return []
    return [
        f"{', '.join(repr(name) for name in unknown)} is not a slot {intent} reads, so the "
        f"value would be dropped (it reads: {', '.join(sorted(known))})"
    ]


async def async_forget(hass: HomeAssistant, entry: ConfigEntry, utterance: str) -> dict[str, Any]:
    """Remove one learned utterance, from every file it appears in.

    Two things this has to get right.

    It works for *any* sentence this integration wrote, not only for a bound
    phrasing. A learned intent's own sentences never become phrase rows, so
    looking the utterance up in the store alone made "forget that phrasing"
    impossible for them - the only remedy was to forget the whole intent.

    Both sides are compared in their normalised form. The phrase row holds what
    the person said and the file holds what was written for the matcher, so they
    differ whenever punctuation was stripped; comparing raw strings left the
    sentence in the file while dropping the only record of it.
    """
    data: LearnerData = entry.runtime_data
    wanted = normalize_sentence(utterance)
    if not wanted:
        raise LearnerError("no utterance given")

    removed = await async_run(
        hass, _forget_in_files, sentences_root(hass, language_of(entry)), wanted
    )

    rows = [p for p in data.phrases if normalize_sentence(p.utterance) == wanted]
    if not removed and not rows:
        raise LearnerError(f"{utterance!r} was not learned by this integration")

    data.phrases = [p for p in data.phrases if normalize_sentence(p.utterance) != wanted]
    await async_save(hass, entry)
    await hass.services.async_call("conversation", "reload", {}, blocking=True)
    data.last_result = {"forgotten": utterance, "rows": len(rows), "files": removed}
    async_dispatcher_send(hass, SIGNAL_UPDATED)
    return data.last_result


def _forget_in_files(root: Path, wanted: str) -> list[str]:
    """Remove one normalised sentence from every file of ours. Returns the files touched."""
    removed: list[str] = []
    for path in sorted(root.glob(f"{FILE_PREFIX}*.yaml")) if root.is_dir() else []:
        try:
            content = _load(path)
        except LearnerError:
            continue
        touched = False
        intents = content.get("intents") or {}
        for intent in list(intents):
            body = intents.get(intent) or {}
            for item in list(body.get("data") or []):
                original = list((item or {}).get("sentences") or [])
                kept = [s for s in original if normalize_sentence(s) != wanted]
                if kept != original:
                    touched = True
                if kept:
                    item["sentences"] = kept
                elif item in (body.get("data") or []):
                    body["data"].remove(item)
            if not body.get("data"):
                intents.pop(intent, None)
        if not touched:
            continue
        removed.append(path.name)
        if intents:
            path.write_text(yaml_util.dump(content), encoding="utf-8")
        else:
            path.unlink(missing_ok=True)
    return removed
