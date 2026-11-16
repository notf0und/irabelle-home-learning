"""LLM tools: how the assistant teaches the house.

Home Assistant's `llm` platform protocol lets an integration contribute tools to
the Assist API, which the conversation agents already use. So the model gets real
tools instead of being told to hand-edit YAML:

* ``irabelle_home_learning__ListIntents``    - what exists, and what a phrase may bind to
* ``irabelle_home_learning__LearnUtterance`` - bind new wording to an existing intent
* ``irabelle_home_learning__LearnIntent``    - create a whole new intent (template + words)
* ``irabelle_home_learning__ForgetUtterance``/``irabelle_home_learning__ForgetIntent`` - undo

The guidance about *when* learning is appropriate travels with the tools, so it
is not duplicated into every agent's system prompt. Which agents may teach at all
is configurable (`allowed_platforms`) - by default only the production DeepSeek
agent, so an experimental agent cannot quietly write sentences.
"""

from __future__ import annotations

import logging
import re

from homeassistant.components.homeassistant import async_should_expose

import voluptuous as vol
from homeassistant.components.llm import LLMTools
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers import intent as intent_helper
from homeassistant.helpers.llm import (
    LLM_API_ASSIST,
    LLMContext,
    IntentTool,
    Tool,
    ToolInput,
)
from homeassistant.util.json import JsonObjectType

from .autolearn import async_change_response, async_review_learning
from .fileio import async_run
from .const import DOMAIN
from .intents import (
    LearnedIntentHandler,
    _reads,
    _render,
    async_forget_intent,
    async_learn_intent,
    missing_attributes,
    slot_sample,
)

#: How many learned intents an agent is shown. Enough to be useful, few enough
#: that the tool list does not swell the prompt.
MAX_EXPOSED_INTENTS = 25
from .learner import (
    LearnerError,
    allowed_platforms,
    async_forget,
    async_intent_names,
    async_learn,
    entity_aliases,
    entity_label,
    language_of,
    learned_phrases,
)

_LOGGER = logging.getLogger(__name__)

#: Words that carry no topic on their own, so a search for "what is the weather" does not
#: match every entity whose name happens to contain "the" or "is".
_ENTITY_SEARCH_STOP = frozenset(
    {
        "the", "and", "for", "what", "whats", "how", "does", "did", "was", "were", "is",
        "are", "can", "you", "your", "mine", "our", "with", "from", "that", "this",
        "there", "here", "please", "tell", "give", "show", "much", "many", "will",
        "would", "could", "should", "get", "set", "use", "using", "when", "where",
        "which", "who", "why", "about", "into", "out", "now", "today", "tomorrow",
    }
)

PROMPT = """You can teach this house to answer a request locally, without a model. Learning
is mostly automatic - the house already learns by itself from requests the local matcher
missed - so use these tools for the cases automation cannot see:

- the user explicitly asks you to remember, or to forget, something;
- you know a specific wording should be answered by a specific intent;
- irabelle_home_learning__LearnIntent is the only way to add a whole new question type from here.

The tools:
- irabelle_home_learning__LearnUtterance - bind wording to an intent that ALREADY exists
- irabelle_home_learning__LearnIntent - a new intent: you supply a name, a Jinja template that
  produces the spoken answer, and the words people say
- irabelle_home_learning__SetResponse - change the wording of an answer the house already gives
- irabelle_home_learning__ListEntities - the entity ids that really exist. Call it whenever a template
  is involved, and copy an id exactly as listed
- irabelle_home_learning__ListIntents - what exists, and what a phrase may bind to
- irabelle_home_learning__ForgetUtterance / irabelle_home_learning__ForgetIntent - undo

**Never write an entity id from memory.** Before you write or change a template, call
irabelle_home_learning__ListEntities with the words from the request and use an `entity_id` from what
it returns, character for character. A friendly name is not an id: "Home Home Forecast" is
`weather.home`, and a name turned into an id is the usual reason a write is refused. If the
entity you need is not in that list, say so - do not invent one.

When the user asks you to forget, unlearn or undo something the house learned, do it in
that turn: call irabelle_home_learning__ListIntents to find the stored wording, then
irabelle_home_learning__ForgetUtterance once per phrasing - or irabelle_home_learning__ForgetIntent to drop a
whole learned question along with its wording. Never guess the wording, because
forgetting matches the stored sentence; and never say you have forgotten something you
did not remove - the result tells you what actually went.

Teach only when it will be said again, when it is unambiguous, and - for a new intent -
when one short template over already-exposed entities answers it every time. Every
phrasing you supply must ask the SAME request with the SAME scope: do not give a question
about all the TVs the phrasing "is the bedroom TV on", because that asks about one TV and
would then be answered with all of them. Never invent an intent name: check with
irabelle_home_learning__ListIntents.

Templates are rendered on every request, so keep them simple and safe: states(),
state_attr() and is_state() over existing entities, one short sentence. times: use
states.<entity_id>.last_changed or .last_updated through as_timestamp and
timestamp_custom (they are UTC, so .strftime on them directly says AM for PM). NEVER state_attr(x, 'last_updated') - it is on the state, not in its
attributes, so it renders the word None and the house says "at None" out loud; a
template that renders a None is refused.

These variables are always defined: user_name, user_id, person_name, person_state,
user_area, user_floor, device_id, satellite_id. user_name is the person chatting in
Home Assistant; from a voice satellite it is empty, so fall back to the recognised
voice, e.g.
{% set who = user_name or (states('sensor.voicebm_current_speaker') if speaker_certain else '') %}, and check speaker_certain before answering per person - a wrong speaker gives the wrong person's answer and sounds just as sure. Use them for
anything "my"/"here" - it is the only way a locally answered intent knows who is asking.
A wrong binding is worse than no binding.
When the user asks for a DIFFERENT FORMAT of an answer the house already gives, do not
re-learn the intent: use irabelle_home_learning__SetResponse to change its reply, keeping the same
facts.

"Improve this template", "where it says X it should say Y", "add the units", "turn W into
west", "round that to one decimal", "do not split that with a full stop" - all of those are
changes to the text of one reply, and none of them is a new intent. Read the current text
with irabelle_home_learning__GetIntent, edit it, and send the WHOLE reply back with
irabelle_home_learning__SetResponse: whatever you do not retype is lost. A learned intent's template is
in this integration's store and its sentence file under `custom_sentences/`; do not go
looking for it with file tools - GetIntent is how you read it, and the file tools cannot
see the store at all.

If a tool refuses a write, the refusal names the exact problem - an entity that does not
exist (with what to use instead), a variable nothing defines, a slot with too few values.
Read it and send a corrected call; never send the same call again unchanged. An entity's
**friendly name is not its entity id**: "Home Home Forecast" is `weather.home`, and only
ids with a dot in them belong in a template. If a change is still refused after one
correction, tell the user what the refusal said instead of apologising that you did not
understand."""


def _entries(hass: HomeAssistant) -> list[ConfigEntry]:
    """Return loaded Irabelle Home Learning config entries."""
    return [
        entry
        for entry in hass.config_entries.async_entries(DOMAIN)
        if getattr(entry, "runtime_data", None) is not None
    ]


def _may_teach(hass: HomeAssistant, llm_context: LLMContext) -> bool:
    """Whether this platform is allowed to teach the house."""
    entries = _entries(hass)
    if not entries:
        return False
    allowed = allowed_platforms(entries[0])
    if not allowed:
        return True
    return (llm_context.platform or "") in allowed


def _source(llm_context: LLMContext) -> str:
    """Describe where a proposal came from, for the provenance record."""
    parts = [llm_context.platform or "unknown"]
    if llm_context.device_id:
        parts.append(llm_context.device_id)
    return ":".join(parts)


class ListEntitiesTool(Tool):
    """Show the entity ids that really exist, so a template copies one instead of guessing."""

    name = "irabelle_home_learning__ListEntities"
    description = (
        "List the entities this house really has, with their exact entity_id, friendly "
        "name, current state and the names of their attributes. Call this BEFORE writing or "
        "changing any template, and copy an entity_id character for character: a friendly "
        "name is NOT an id ('Home Home Forecast' is weather.home), and an id nothing in "
        "this house has is refused. Pass `search` with the words from the request to see "
        "the entities that could answer it - including ones an answer could use but the "
        "person did not name; with no search it lists what Assist is allowed to read."
    )
    parameters = vol.Schema({vol.Optional("search"): cv.string})

    async def async_call(
        self, hass: HomeAssistant, tool_input: ToolInput, llm_context: LLMContext
    ) -> JsonObjectType:
        search = str(tool_input.tool_args.get("search") or "").strip()
        words = {
            word
            for word in re.split(r"[^a-z0-9]+", search.lower())
            if len(word) > 2 and word not in _ENTITY_SEARCH_STOP
        }
        scored: list[tuple[int, int, str]] = []
        for state in hass.states.async_all():
            exposed = async_should_expose(hass, "conversation", state.entity_id)
            if not search and not exposed:
                continue
            aliases = entity_aliases(hass, state.entity_id)
            haystack = " ".join(
                [state.entity_id, state.name or "", *aliases]
            ).lower()
            score = sum(1 for word in words if word in haystack)
            if words and not score:
                continue
            scored.append((score, 1 if exposed else 0, state.entity_id))
        scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        rows = []
        for _, _, entity_id in scored[:40]:
            state = hass.states.get(entity_id)
            if state is None:
                continue
            rows.append(
                {
                    "entity_id": entity_id,
                    "name": entity_label(hass, entity_id, state.name),
                    "state": str(state.state),
                    "attributes": sorted(state.attributes)[:12],
                    "assist_can_read": async_should_expose(
                        hass, "conversation", entity_id
                    ),
                }
            )
        return {
            "success": True,
            "result": {
                "entities": rows,
                "count": len(rows),
                "note": (
                    "Use these entity_id values exactly as written. An id is the string "
                    "with a dot and no spaces; the name is what a person calls it. If the "
                    "entity you need is not here, say so instead of inventing an id."
                ),
            },
        }


class ListIntentsTool(Tool):
    """List the intents that exist, and what has been learned for them."""

    name = "irabelle_home_learning__ListIntents"
    description = (
        "List the intents this house knows AND the exact phrasings this integration has "
        "learned for them. Use it before teaching anything, so a phrase is bound to an "
        "intent that exists (LearnUtterance) or you can see that none fits and a new one "
        "is warranted (LearnIntent). Use it before forgetting anything too, so you pass "
        "the stored wording instead of a paraphrase. Optionally filter by a search string. "
        "The templates and replies are not listed here - they are long; call "
        "irabelle_home_learning__GetIntent to read the one you are about to change."
    )
    parameters = vol.Schema({vol.Optional("search"): cv.string})

    async def async_call(
        self, hass: HomeAssistant, tool_input: ToolInput, llm_context: LLMContext
    ) -> JsonObjectType:
        search = (tool_input.tool_args.get("search") or "").strip().lower()
        names = sorted(await async_intent_names(hass))
        if search:
            names = [name for name in names if search in name.lower()]

        learned: dict[str, list[str]] = {}
        if entries := _entries(hass):
            learned = await async_run(hass, learned_phrases, hass, language_of(entries[0]))
        if search:
            learned = {
                intent: [phrase for phrase in phrases if search in phrase.lower()]
                for intent, phrases in learned.items()
                if search in intent.lower()
                or any(search in phrase.lower() for phrase in phrases)
            }

        return {
            "success": True,
            "result": {
                "intents": names[:120],
                "total": len(names),
                "learned_phrases": learned,
                "learned_note": (
                    "learned_phrases is what this integration wrote, by intent. "
                    "ForgetUtterance must be given that exact wording."
                ),
            },
        }


class LearnUtteranceTool(Tool):
    """Bind new wording to an intent that already exists."""

    name = "irabelle_home_learning__LearnUtterance"
    description = (
        "Teach the house new wording for an intent that ALREADY exists. After this the "
        "exact words (and the variants you supply) are answered by the local intent "
        "matcher in milliseconds, with no model call. Refuses ambiguous phrases, unknown "
        "intents, and anything that would re-target an existing phrase."
    )
    parameters = vol.Schema(
        {
            vol.Required("utterances"): vol.All(cv.ensure_list, [cv.string]),
            vol.Required("intent"): cv.string,
            vol.Optional("slots"): dict,
            vol.Optional("note"): cv.string,
        }
    )

    async def async_call(
        self, hass: HomeAssistant, tool_input: ToolInput, llm_context: LLMContext
    ) -> JsonObjectType:
        entries = _entries(hass)
        if not entries:
            return {"success": False, "error": "the Irabelle Home Learning integration is not set up"}
        try:
            result = await async_learn(
                hass,
                entries[0],
                utterances=tool_input.tool_args["utterances"],
                intent=tool_input.tool_args["intent"],
                slots=tool_input.tool_args.get("slots"),
                note=tool_input.tool_args.get("note"),
                source=_source(llm_context),
            )
        except LearnerError as err:
            return {"success": False, "error": str(err)}
        except Exception as err:  # noqa: BLE001 - never fail the conversation turn
            return {"success": False, "error": f"could not learn that: {err}"}
        reviewed = await _review(hass, entries[0], tool_input.tool_args["intent"])
        return {"success": True, "result": result, **_reviewed(reviewed)}


class LearnIntentTool(Tool):
    """Create a whole new intent, with a template that answers it."""

    name = "irabelle_home_learning__LearnIntent"
    description = (
        "Create a NEW intent when no existing intent fits the request - typically a "
        "question whose answer must be built from entity state. You supply a name "
        "(letters, digits, underscores; e.g. WashingMachineStatus), a Jinja template that "
        "renders the spoken answer, and the words people say. The template is validated "
        "before it is installed and is rendered locally on every request, so no model is "
        "involved afterwards. Only reference entities already exposed to Assist. "
        "When the same request is asked with one word swapped for another value in the "
        "same class - a weekday, a room, a person - do NOT create one intent per value "
        "and do NOT hard-code one of them: pass `slots` with the whole class (for a "
        "weekday, all seven days plus today and tomorrow) and write {day} where that word "
        "goes in the wording. The list is matched by the local sentence matcher, and the "
        "template may then read `day`, so every value works - including one nobody has "
        "said yet. "
        "One request has ONE answer: if an intent listed by ListIntents already answers "
        "the same question, this is refused - bind the wording to it with LearnUtterance "
        "instead, and use SetResponse if the existing answer is what should change."
    )
    parameters = vol.Schema(
        {
            vol.Required("name"): cv.string,
            vol.Required("template"): cv.string,
            vol.Required("utterances"): vol.All(cv.ensure_list, [cv.string]),
            vol.Optional("description"): cv.string,
            vol.Optional("note"): cv.string,
            vol.Optional("response"): cv.string,
            vol.Optional("slots"): dict,
        }
    )

    async def async_call(
        self, hass: HomeAssistant, tool_input: ToolInput, llm_context: LLMContext
    ) -> JsonObjectType:
        entries = _entries(hass)
        if not entries:
            return {"success": False, "error": "the Irabelle Home Learning integration is not set up"}
        try:
            result = await async_learn_intent(
                hass,
                entries[0],
                name=tool_input.tool_args["name"],
                template=tool_input.tool_args["template"],
                utterances=tool_input.tool_args["utterances"],
                description=tool_input.tool_args.get("description"),
                note=tool_input.tool_args.get("note"),
                source=_source(llm_context),
                response=tool_input.tool_args.get("response"),
                slots=tool_input.tool_args.get("slots"),
            )
        except LearnerError as err:
            return {"success": False, "error": str(err)}
        except Exception as err:  # noqa: BLE001
            return {"success": False, "error": f"could not learn that intent: {err}"}
        reviewed = await _review(hass, entries[0], tool_input.tool_args["name"])
        return {"success": True, "result": result, **_reviewed(reviewed)}


class GetIntentTool(Tool):
    """Read one learned intent in full, so its answer can be rewritten rather than guessed."""

    name = "irabelle_home_learning__GetIntent"
    description = (
        "Read one learned intent in full: the template that computes its answer, the reply it "
        "actually says (the text to edit), what it renders right now, its slots and phrasings, "
        "and the entities it reads with their current states. Call this BEFORE changing how an "
        "answer is worded or formatted: irabelle_home_learning__SetResponse replaces the whole reply, so "
        "anything not carried over from the current text is lost. Do not look for a learned "
        "intent's template in the configuration files - it lives in this integration's store "
        "and its sentence file, and this is how to read it."
    )
    parameters = vol.Schema({vol.Required("name"): cv.string})

    async def async_call(
        self, hass: HomeAssistant, tool_input: ToolInput, llm_context: LLMContext
    ) -> JsonObjectType:
        entries = _entries(hass)
        if not entries:
            return {"success": False, "error": "the Irabelle Home Learning integration is not set up"}
        name = str(tool_input.tool_args.get("name") or "").strip()
        spec = next((item for item in entries[0].runtime_data.intents if item.name == name), None)
        if spec is None:
            known = sorted(item.name for item in entries[0].runtime_data.intents)
            return {
                "success": False,
                "error": (
                    f"{name!r} is not a learned intent, so it has no template here. "
                    f"The learned intents are: {', '.join(known) or '(none)'}. "
                    "A built-in intent's answer is not this component's to rewrite."
                ),
            }
        source = spec.response or spec.template
        sample = slot_sample(spec.slots)
        return {
            "success": True,
            "result": {
                "intent": spec.name,
                "template": spec.template,
                "response_in_the_sentence_file": spec.response,
                "the_reply_to_edit": source,
                "says_right_now": _render(hass, source, sample)[:600],
                "slots": spec.slots,
                "phrasings": (
                    await async_run(hass, learned_phrases, hass, language_of(entries[0]))
                ).get(spec.name, []),
                "reads": _reads(hass, source),
                "warnings": missing_attributes(hass, source),
                "file": spec.file,
                "note": (
                    "`the_reply_to_edit` is what the house says and what "
                    "irabelle_home_learning__SetResponse replaces. Send the whole text back with your "
                    "changes in it. `reads` is every entity that text reads, with its state: "
                    "an entity id in it is real, and one that is not there renders empty."
                ),
            },
        }


class SetResponseTool(Tool):
    """Change how a learned intent answers, without re-learning it."""

    name = "irabelle_home_learning__SetResponse"
    description = (
        "Change how a learned intent answers, when the user asks for different wording OR a "
        "different format - shorter, more detail, a time in it, units on a number, one decimal "
        "place at most, a bearing code turned into a word, two sentences joined. Formatting is "
        "part of the reply template, so this is the tool for all of it. Read the current text "
        "with irabelle_home_learning__GetIntent first and send the WHOLE reply back with your changes in "
        "it: this replaces the reply, so a part you did not retype is gone. Same rules as a "
        "template: one spoken sentence, only entities that exist, and the speaker variables "
        "(user_name, person_name, ...) when the answer is about whoever is asking. It is "
        "rendered and checked before it is installed, so the result reports `says` - what the "
        "house will now say - and any warning about an attribute that is not there."
    )
    parameters = vol.Schema(
        {vol.Required("name"): cv.string, vol.Required("response"): cv.string}
    )

    async def async_call(
        self, hass: HomeAssistant, tool_input: ToolInput, llm_context: LLMContext
    ) -> JsonObjectType:
        entries = _entries(hass)
        if not entries:
            return {"success": False, "error": "the Irabelle Home Learning integration is not set up"}
        try:
            answer: JsonObjectType = await async_change_response(
                hass,
                entries[0],
                tool_input.tool_args["name"],
                tool_input.tool_args["response"],
            )
        except LearnerError as err:
            return {"success": False, "error": str(err)}
        except Exception as err:  # noqa: BLE001
            return {"success": False, "error": f"could not change that response: {err}"}
        return {"success": True, "result": answer}


class ForgetUtteranceTool(Tool):
    """Remove phrasing this integration previously learned."""

    name = "irabelle_home_learning__ForgetUtterance"
    description = (
        "Remove one phrase that was previously learned, for example when the user says it "
        "was interpreted wrongly. Only phrases installed by this integration can be "
        "removed; hand-written sentence files are left alone."
    )
    parameters = vol.Schema({vol.Required("utterance"): cv.string})

    async def async_call(
        self, hass: HomeAssistant, tool_input: ToolInput, llm_context: LLMContext
    ) -> JsonObjectType:
        entries = _entries(hass)
        if not entries:
            return {"success": False, "error": "the Irabelle Home Learning integration is not set up"}
        try:
            result = await async_forget(hass, entries[0], tool_input.tool_args["utterance"])
        except LearnerError as err:
            return {"success": False, "error": str(err)}
        except Exception as err:  # noqa: BLE001
            return {"success": False, "error": f"could not forget that: {err}"}
        return {"success": True, "result": result}


class ForgetIntentTool(Tool):
    """Remove a learned intent entirely."""

    name = "irabelle_home_learning__ForgetIntent"
    description = (
        "Remove an intent that was previously learned by this integration, together with "
        "its sentences. Use it when a learned intent was wrong or is no longer wanted. "
        "Intents that Home Assistant or intent_script.yaml define cannot be removed."
    )
    parameters = vol.Schema({vol.Required("name"): cv.string})

    async def async_call(
        self, hass: HomeAssistant, tool_input: ToolInput, llm_context: LLMContext
    ) -> JsonObjectType:
        entries = _entries(hass)
        if not entries:
            return {"success": False, "error": "the Irabelle Home Learning integration is not set up"}
        try:
            result = await async_forget_intent(
                hass, entries[0], tool_input.tool_args["name"]
            )
        except LearnerError as err:
            return {"success": False, "error": str(err)}
        except Exception as err:  # noqa: BLE001
            return {"success": False, "error": f"could not forget that intent: {err}"}
        return {"success": True, "result": result}


@callback
def async_get_tools(
    hass: HomeAssistant, llm_context: LLMContext, api_id: str
) -> LLMTools | None:
    """Contribute the learner's intents, and - where allowed - its teaching tools.

    Learned intents are exposed to *every* agent, whatever the teaching policy:
    a question the house already knows how to answer locally is exactly what the
    model should reach for, and reaching for it is also how a new wording for it
    gets learned (the observer binds whatever intent a turn used).
    """
    if api_id != LLM_API_ASSIST:
        return None

    handlers = sorted(
        (
            handler
            for handler in intent_helper.async_get(hass)
            if isinstance(handler, LearnedIntentHandler)
        ),
        key=lambda handler: handler.intent_type,
    )
    tools: list[Tool] = [
        IntentTool(f"{DOMAIN}__{handler.intent_type}", handler)
        for handler in handlers[:MAX_EXPOSED_INTENTS]
    ]

    prompt = None
    if _may_teach(hass, llm_context):
        tools.extend(
            [
                LearnUtteranceTool(),
                LearnIntentTool(),
                SetResponseTool(),
                GetIntentTool(),
                ListEntitiesTool(),
                ListIntentsTool(),
                ForgetUtteranceTool(),
                ForgetIntentTool(),
            ]
        )
        prompt = PROMPT

    if not tools:
        return None
    return LLMTools(tools=tools, prompt=prompt)

async def _review(
    hass: HomeAssistant, entry: ConfigEntry, intent: str
) -> dict[str, object] | None:
    """Check a learning the model just made, and undo it if the answer is wrong.

    The teaching tools are the path that produced the first wrong TV answer: the
    model added "is the bedroom TV on" to a question about all the TVs, so that
    phrasing was answered with both. Nothing mechanical notices that, so a second
    model reads back what the house will now say and removes what is wrong.
    """
    try:
        return await async_review_learning(hass, entry, intent)
    except Exception:  # noqa: BLE001 - a review must never fail the teaching call
        _LOGGER.exception("Irabelle Home Learning could not review %s", intent)
        return None


def _reviewed(review: dict[str, object] | None) -> dict[str, object]:
    """Fold a review outcome into a tool result the model will read."""
    if not review:
        return {}
    if review.get("review") == "ok":
        return {"review": "ok"}
    return {"review": review}
