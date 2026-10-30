"""Ask the classifier a typed question and get an answer code can branch on.

The learner makes the same shape of decision over and over — is this phrasing worth
learning, which existing intent does it mean, is this answer actually right — and every
one of those is a snap judgement over text. Three question types cover all of them:
``Choice`` (pick one option, with probabilities), ``Noul`` (how likely is this true) and
``Score`` (place it on an ordered rubric).

Who answers them is configuration, not code. Every caller asks through `async_ask`, and
the configured *classification model* decides where the questions go:

* ``jev_integration`` - the Jev integration's own ``jev.ask`` action. Its API key, daily
  budget and spend counters are used, so there is one secret and one budget to watch.
* ``jev`` - the TypeSafe System One API, called directly with the endpoint, model and key
  in this integration's options. Needs nothing else installed.
* ``ai_task.<entity>`` or ``conversation.<entity>`` - any other model: a small local one,
  a cheap cloud one, whatever classifies well enough. An ``ai_task`` entity answers
  through structured output, a conversation agent with JSON (see `models.py`).
* ``ai_task`` - "the same as the processing model", from an earlier configuration.
* ``none`` - nothing is asked; every judgement-backed step is skipped, as before.

The answers come back in one shape whichever model gave them, so the thresholds and the
decisions downstream do not know or care. A System One model is built for this - typed
answers with probabilities, a fraction of a cent and a quarter of a second per call - and
a text model is slower and its probabilities are self-reported rather than measured. The
thresholds were set from Jev's numbers; with a text model they are a starting point.

Design notes that come straight from the TypeSafe docs and hold for any model:

* **Ask every question in one request.** Questions run in parallel and adding one
  "barely changes the response time" — so the honest way to ask five judgements is one
  call, not five. Anything dependent on an earlier answer needs a second call.
* **One snap judgement per question.** A broad question hides several judgements behind
  one answer; decompose, then combine in code.
* **Confidence is group-level, not a guarantee.** It says how sure the model is, and the
  docs' own advice is to set thresholds from measured accuracy rather than taste. Every
  threshold below is a starting point to be replaced by data.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .agents import settings
from .models import async_generate_data, label_of
from .const import (
    CONF_AI_TASK_ENTITY,
    CONF_JEV_API_KEY,
    CONF_JEV_ENDPOINT,
    CONF_JEV_MODEL,
    CONF_JUDGE,
    DEFAULT_JEV_ENDPOINT,
    DEFAULT_JEV_MODEL,
    DEFAULT_JUDGE,
    DOMAIN,
    JUDGE_AI_TASK,
    JUDGE_JEV,
    JUDGE_JEV_INTEGRATION,
    JUDGE_NONE,
    MAX_JUDGE_CALLS_PER_HOUR,
)

_LOGGER = logging.getLogger(__name__)

#: Long enough for a large state, short enough that a hung request does not hold a turn.
REQUEST_TIMEOUT = 30.0

#: The endpoint is documented to return 429 (rate limited) and 529 (overloaded); the
#: official SDKs back off and honour `retry-after`, so this does too.
RETRY_STATUSES = (429, 529)
MAX_ATTEMPTS = 3


class JudgeError(HomeAssistantError):
    """A question could not be answered."""


class JudgeNotConfigured(JudgeError):
    """No judge is configured, or the configured one cannot be used."""


# --------------------------------------------------------------------------
# Questions
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Choice:
    """Pick one option. Returns ``choice``, ``probabilities`` and ``confidence``."""

    instructions: Any
    criteria: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": "choice",
            "instructions": self.instructions,
            "criteria": self.criteria,
        }


@dataclass(slots=True)
class Noul:
    """Is this true? Returns ``noul`` (0–1). No confidence field is provided.

    The ``true``/``false`` descriptions belong inside ``criteria``, as the API reference
    shows. Sending them as top-level fields is accepted and silently scores differently:
    measured on one identical state, a question answered 0.55 with ``criteria`` and 0.86
    without, and the difference was reproducible in both directions. The labels are part
    of the prompt, so they change the answer - which makes them worth writing carefully
    and worth measuring, not worth removing.
    """

    instructions: Any
    true: str | None = None
    false: str | None = None

    def as_dict(self) -> dict[str, Any]:
        question: dict[str, Any] = {"type": "noul", "instructions": self.instructions}
        if self.true or self.false:
            question["criteria"] = {
                "true": self.true or "",
                "false": self.false or "",
            }
        return question


@dataclass(slots=True)
class Score:
    """Score the state against an ordered rubric of 2–10 levels."""

    instructions: Any
    criteria: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": "score",
            "instructions": self.instructions,
            "criteria": self.criteria,
        }


Question = Choice | Noul | Score


@dataclass(slots=True)
class Answer:
    """One answer, with the parts worth branching on lifted out.

    ``probabilities`` is kept whole because the top choice alone hides a near-tie, and a
    near-tie is the difference between "bind this" and "this is ambiguous, leave it".
    """

    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def type(self) -> str:
        return str(self.raw.get("type") or "")

    @property
    def choice(self) -> str | None:
        value = self.raw.get("choice")
        return str(value) if value is not None else None

    @property
    def noul(self) -> float | None:
        value = self.raw.get("noul")
        return float(value) if isinstance(value, (int, float)) else None

    @property
    def score(self) -> float | None:
        value = self.raw.get("score")
        return float(value) if isinstance(value, (int, float)) else None

    @property
    def confidence(self) -> float | None:
        value = self.raw.get("confidence")
        return float(value) if isinstance(value, (int, float)) else None

    @property
    def probabilities(self) -> dict[str, float]:
        value = self.raw.get("probabilities")
        return dict(value) if isinstance(value, dict) else {}

    def margin(self) -> float | None:
        """How far ahead the chosen option is. None when there is nothing to compare.

        A choice made at 0.40 against 0.35 is a coin toss that reports the same top
        answer as 0.95 against 0.02, and the learner's two worst failures were exactly
        that: a phrasing that fit two intents, and a plural question that fitted a
        singular template.
        """
        if not self.raw.get("choice"):
            return None
        values = sorted(self.probabilities.values(), reverse=True)
        if len(values) < 2:
            return 1.0 if values else None
        return float(values[0] - values[1])


# --------------------------------------------------------------------------
# Usage
# --------------------------------------------------------------------------


@dataclass(slots=True)
class JudgeUsage:
    """What this integration has spent on the judge, since the last restart."""

    calls: int = 0
    failures: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    calls_this_hour: int = 0
    hour_started: float = field(default_factory=time.monotonic)

    def record(self, usage: dict[str, Any] | None) -> None:
        self.calls += 1
        now = time.monotonic()
        if now - self.hour_started > 3600:
            self.calls_this_hour = 0
            self.hour_started = now
        self.calls_this_hour += 1
        for key, attribute in (
            ("input_tokens", "input_tokens"),
            ("output_tokens", "output_tokens"),
        ):
            value = (usage or {}).get(key)
            if isinstance(value, (int, float)):
                setattr(self, attribute, getattr(self, attribute) + int(value))


_USAGE: dict[str, JudgeUsage] = {}


def usage_of(entry: ConfigEntry) -> JudgeUsage:
    return _USAGE.setdefault(entry.entry_id, JudgeUsage())


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


def _jev_key_from_home(hass: HomeAssistant) -> tuple[str | None, str]:
    """The API key already configured for the Jev integration, if there is one.

    Looking here rather than asking for the key twice means one secret to rotate. It is
    read-only, and a missing Jev integration is not an error — it just means no key.
    """
    for entry in hass.config_entries.async_entries("jev"):
        key = str(entry.data.get("api_key") or "").strip()
        if key:
            return key, "jev integration"
    return None, "none"


def endpoint_of(entry: ConfigEntry) -> str:
    value = settings(entry).get(CONF_JEV_ENDPOINT)
    return str(value or DEFAULT_JEV_ENDPOINT).strip()


def model_of(entry: ConfigEntry) -> str:
    value = settings(entry).get(CONF_JEV_MODEL)
    return str(value or DEFAULT_JEV_MODEL).strip()


def api_key_of(hass: HomeAssistant, entry: ConfigEntry) -> tuple[str | None, str]:
    """The key to use, and where it came from, for diagnostics."""
    configured = str(settings(entry).get(CONF_JEV_API_KEY) or "").strip()
    if configured:
        return configured, f"{DOMAIN} options"
    return _jev_key_from_home(hass)


def judge_of(entry: ConfigEntry) -> str:
    """Which classification model answers the learner's snap judgements.

    The stored value is a mode (``jev_integration``, ``jev``, ``ai_task``, ``none``) or an
    ``ai_task.*`` entity id. Entity ids keep their case; everything else is normalised.
    """
    value = str(settings(entry).get(CONF_JUDGE) or "").strip()
    if value.startswith(("ai_task.", "conversation.")):
        return value
    return value.lower() or DEFAULT_JUDGE


def has_judge(entry: ConfigEntry) -> bool:
    """Whether a classification model is configured at all.

    Configured is not the same as reachable: a model that is missing or out of quota
    raises `JudgeError` when asked, and every caller treats that as "no judgement".
    """
    return judge_of(entry) != JUDGE_NONE


def _ai_task_entity_of(hass: HomeAssistant, entry: ConfigEntry) -> str | None:
    """The model that classifies, when the judge is a text model rather than Jev."""
    judge = judge_of(entry)
    if judge.startswith(("ai_task.", "conversation.")):
        return judge
    if judge != JUDGE_AI_TASK:
        return None
    # "Same as the processing model" - the setting an older configuration may still hold.
    configured = str(settings(entry).get(CONF_AI_TASK_ENTITY) or "").strip()
    if configured:
        return configured
    states = hass.states.async_all("ai_task")
    return states[0].entity_id if states else None


def _jev_integration_loaded(hass: HomeAssistant) -> bool:
    return hass.services.has_service("jev", "ask") and bool(
        hass.config_entries.async_loaded_entries("jev")
    )


def model_label(hass: HomeAssistant, entry: ConfigEntry) -> str:
    """A name for the model that answers, for reports."""
    judge = judge_of(entry)
    if judge == JUDGE_JEV:
        return model_of(entry)
    if judge == JUDGE_JEV_INTEGRATION:
        return "jev integration (jev.ask)"
    if judge == JUDGE_NONE:
        return "none"
    entity_id = _ai_task_entity_of(hass, entry)
    return label_of(hass, entity_id) if entity_id else "no model"


def describe(hass: HomeAssistant, entry: ConfigEntry) -> dict[str, Any]:
    """Everything the diagnostic service reports about the judge."""
    judge = judge_of(entry)
    usage = usage_of(entry)
    report: dict[str, Any] = {"judge": judge, "model": model_label(hass, entry)}
    if judge == JUDGE_JEV:
        key, source = api_key_of(hass, entry)
        report |= {
            "available": key is not None,
            "endpoint": endpoint_of(entry),
            "key_source": source,
        }
    elif judge == JUDGE_JEV_INTEGRATION:
        report["available"] = _jev_integration_loaded(hass)
    elif judge == JUDGE_NONE:
        report["available"] = False
    else:
        entity_id = _ai_task_entity_of(hass, entry)
        report |= {
            "available": entity_id is not None and hass.states.get(entity_id) is not None,
            "ai_task_entity": entity_id,
        }
    return report | {
        "calls_since_restart": usage.calls,
        "calls_this_hour": usage.calls_this_hour,
        "calls_per_hour_limit": MAX_JUDGE_CALLS_PER_HOUR,
        "input_tokens_since_restart": usage.input_tokens,
        "output_tokens_since_restart": usage.output_tokens,
        "failures": usage.failures,
    }


# --------------------------------------------------------------------------
# The call
# --------------------------------------------------------------------------


async def async_ask(
    hass: HomeAssistant,
    entry: ConfigEntry,
    state: Any,
    questions: dict[str, Question],
    *,
    model: str | None = None,
    rate_limited: bool = True,
) -> dict[str, Answer]:
    """Ask every question in one request and return the answers by id.

    ``rate_limited=False`` is for the places where a human asked for the answer (a
    diagnostic service, a review they triggered) rather than the learner deciding on its
    own to spend money. ``model`` overrides the Jev model name for a direct API call and
    is ignored by the other backends.
    """
    if not questions:
        raise JudgeError("no questions given")

    judge = judge_of(entry)
    if judge == JUDGE_NONE:
        raise JudgeNotConfigured("no classification model is configured")

    usage = usage_of(entry)
    if rate_limited and usage.calls_this_hour >= MAX_JUDGE_CALLS_PER_HOUR:
        raise JudgeError(
            f"{usage.calls_this_hour} judge calls this hour is the limit "
            f"({MAX_JUDGE_CALLS_PER_HOUR}); refusing to spend more"
        )

    if judge == JUDGE_JEV:
        return await _ask_jev_api(hass, entry, state, questions, model=model)
    if judge == JUDGE_JEV_INTEGRATION:
        return await _ask_jev_integration(hass, entry, state, questions)
    return await _ask_ai_task(hass, entry, state, questions)


async def _ask_jev_integration(
    hass: HomeAssistant,
    entry: ConfigEntry,
    state: Any,
    questions: dict[str, Question],
) -> dict[str, Answer]:
    """Ask through the Jev integration's ``jev.ask`` action.

    That action takes the questions in the same shape the API does, except that a noul's
    labels are top-level ``true``/``false`` keys it moves into ``criteria`` itself.
    """
    usage = usage_of(entry)
    if not _jev_integration_loaded(hass):
        raise JudgeNotConfigured(
            "the classification model is the Jev integration, and it is not loaded"
        )
    raw: dict[str, Any] = {}
    for key_, question in questions.items():
        body = question.as_dict()
        if isinstance(question, Noul):
            body.pop("criteria", None)
            if question.true:
                body["true"] = question.true
            if question.false:
                body["false"] = question.false
        raw[key_] = body
    # jev.ask renders a string state that looks like a template. Ours is data, often
    # containing Jinja the learner is judging, so a string that could be mistaken for a
    # template goes as an object instead.
    if isinstance(state, str) and ("{{" in state or "{%" in state):
        state = {"text": state}
    try:
        response = await hass.services.async_call(
            "jev",
            "ask",
            {"state_template": state, "questions": raw},
            blocking=True,
            return_response=True,
        )
    except HomeAssistantError as err:
        usage.failures += 1
        raise JudgeError(f"jev.ask failed: {err}") from err
    except Exception as err:  # noqa: BLE001 - another integration's failure is ours to absorb
        usage.failures += 1
        raise JudgeError(f"jev.ask failed: {err}") from err
    response = response or {}
    usage.record(response.get("usage"))
    answers = response.get("answers") or {}
    return {key_: Answer(raw=answers.get(key_) or {}) for key_ in questions}


# --------------------------------------------------------------------------
# Any text model, through ai_task
# --------------------------------------------------------------------------

AI_TASK_INSTRUCTIONS = """You are a classifier. Read STATE, then answer every QUESTION \
below independently, each as one snap judgement about STATE. Do not explain.

Probabilities are your honest estimate from 0 to 1. 0.5 means you cannot tell, not that \
the answer is half true. Do not round towards 0 or 1 unless you are certain.

For a choice: give the option that fits best and its probability, then the option that \
fits second best and its probability (the two must not add up to more than 1).
For a true/false question: give the probability that the statement is TRUE.
For a score: give the level that fits best and its probability.

STATE:
{state}

QUESTIONS:
{questions}
"""


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=1, default=str)


def _probability_field(description: str) -> dict[str, Any]:
    return {
        "description": description,
        "required": True,
        "selector": {"number": {"min": 0, "max": 1, "step": 0.01, "mode": "box"}},
    }


def _select_field(description: str, options: list[str]) -> dict[str, Any]:
    return {
        "description": description,
        "required": True,
        "selector": {"select": {"options": options, "mode": "dropdown"}},
    }


def _ai_task_question(key_: str, question: Question) -> tuple[str, dict[str, Any]]:
    """The prompt text and the output fields for one question."""
    if isinstance(question, Choice):
        options = [str(option) for option in question.criteria]
        lines = "\n".join(
            f"  - {option}: {_text(meaning)}" if meaning else f"  - {option}"
            for option, meaning in question.criteria.items()
        )
        prompt = f"[{key_}] choice: {_text(question.instructions)}\n  options:\n{lines}"
        fields = {
            f"{key_}__choice": _select_field(f"{key_}: the option that fits best", options),
            f"{key_}__probability": _probability_field(f"{key_}: its probability"),
            f"{key_}__runner_up": _select_field(
                f"{key_}: the option that fits second best", options
            ),
            f"{key_}__runner_up_probability": _probability_field(
                f"{key_}: the second option's probability"
            ),
        }
        return prompt, fields
    if isinstance(question, Noul):
        prompt = f"[{key_}] true/false: {_text(question.instructions)}"
        if question.true:
            prompt += f"\n  true means: {question.true}"
        if question.false:
            prompt += f"\n  false means: {question.false}"
        return prompt, {
            f"{key_}__true": _probability_field(f"{key_}: probability that it is true")
        }
    levels = [str(level) for level in question.criteria]
    lines = "\n".join(f"  {index}: {level}" for index, level in enumerate(levels))
    prompt = f"[{key_}] score: {_text(question.instructions)}\n  levels, lowest first:\n{lines}"
    return prompt, {
        f"{key_}__level": _select_field(
            f"{key_}: the number of the level that fits best",
            [str(index) for index in range(len(levels))],
        ),
        f"{key_}__probability": _probability_field(f"{key_}: its probability"),
    }


def _probability(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number > 1 and number <= 100:
        number /= 100  # a model that answered in percent
    return min(1.0, max(0.0, number))


def _ai_task_answer(key_: str, question: Question, data: dict[str, Any]) -> Answer:
    """Read one question's fields back into the shape a Jev answer has."""
    if isinstance(question, Noul):
        noul = _probability(data.get(f"{key_}__true"))
        return Answer(raw={"type": "noul", "noul": noul} if noul is not None else {})

    if isinstance(question, Choice):
        options = [str(option) for option in question.criteria]
        choice = str(data.get(f"{key_}__choice") or "").strip()
        if choice not in options:
            return Answer(raw={})
        top = _probability(data.get(f"{key_}__probability"))
        top = 0.5 if top is None else top
        probabilities = {choice: top}
        runner_up = str(data.get(f"{key_}__runner_up") or "").strip()
        second = _probability(data.get(f"{key_}__runner_up_probability"))
        rest = max(0.0, 1.0 - top)
        if runner_up in options and runner_up != choice and second is not None:
            probabilities[runner_up] = min(second, rest, top)
        elif len(options) > 1:
            # No usable runner-up: the rest of the mass is unaccounted for, and a margin
            # computed against nothing would read as certainty it was never given.
            others = [option for option in options if option != choice]
            probabilities[others[0]] = min(rest, top)
        return Answer(
            raw={
                "type": "choice",
                "choice": choice,
                "probabilities": probabilities,
                "confidence": top,
            }
        )

    levels = [str(level) for level in question.criteria]
    try:
        index = int(str(data.get(f"{key_}__level")).strip())
    except (TypeError, ValueError):
        return Answer(raw={})
    if not 0 <= index < len(levels):
        return Answer(raw={})
    top = _probability(data.get(f"{key_}__probability"))
    top = 0.5 if top is None else top
    return Answer(
        raw={
            "type": "score",
            "score": float(index),
            "probabilities": {levels[index]: top},
            "confidence": top,
        }
    )


async def _ask_ai_task(
    hass: HomeAssistant,
    entry: ConfigEntry,
    state: Any,
    questions: dict[str, Question],
) -> dict[str, Answer]:
    """Ask a text model the same typed questions: an ``ai_task`` entity or a conversation agent."""
    usage = usage_of(entry)
    entity_id = _ai_task_entity_of(hass, entry)
    if entity_id is None or hass.states.get(entity_id) is None:
        raise JudgeNotConfigured(
            f"the classification model {entity_id or '(processing model)'} does not exist"
        )
    prompts: list[str] = []
    structure: dict[str, Any] = {}
    for key_, question in questions.items():
        prompt, fields = _ai_task_question(key_, question)
        prompts.append(prompt)
        structure.update(fields)
    instructions = AI_TASK_INSTRUCTIONS.format(
        state=_text(state), questions="\n\n".join(prompts)
    )
    try:
        response = await async_generate_data(
            hass,
            {
                "task_name": "classify",
                "entity_id": entity_id,
                "instructions": instructions,
                "structure": structure,
            },
        )
    except Exception as err:  # noqa: BLE001 - another integration's failure is ours to absorb
        usage.failures += 1
        raise JudgeError(f"{entity_id} could not classify: {err}") from err
    data = (response or {}).get("data")
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except ValueError:
            data = None
    if not isinstance(data, dict):
        usage.failures += 1
        raise JudgeError(f"{entity_id} returned nothing usable")
    # A text model does not report tokens through ai_task: the call is counted, the
    # tokens are not, and the spend estimate (priced for Jev) stays at what Jev cost.
    usage.record(None)
    return {
        key_: _ai_task_answer(key_, question, data) for key_, question in questions.items()
    }


# --------------------------------------------------------------------------
# Jev, called directly
# --------------------------------------------------------------------------


async def _ask_jev_api(
    hass: HomeAssistant,
    entry: ConfigEntry,
    state: Any,
    questions: dict[str, Question],
    *,
    model: str | None = None,
) -> dict[str, Answer]:
    """One POST to the System One API with this integration's own settings."""
    key, source = api_key_of(hass, entry)
    if key is None:
        raise JudgeNotConfigured(
            "no Jev API key: set one in the Irabelle Home Learning options, or configure "
            "the Jev integration"
        )
    usage = usage_of(entry)
    body = {
        "state": state,
        "model": model or model_of(entry),
        "questions": {key_: question.as_dict() for key_, question in questions.items()},
    }
    session = async_get_clientsession(hass)
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    url = endpoint_of(entry)

    last_error = "no attempt made"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            async with session.post(
                url, json=body, headers=headers, timeout=timeout
            ) as response:
                if response.status in RETRY_STATUSES:
                    last_error = f"HTTP {response.status}"
                    delay = _retry_delay(response, attempt)
                    _LOGGER.debug(
                        "Jev asked again after %s (attempt %d) in %.1fs",
                        last_error,
                        attempt,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                if response.status == 401:
                    usage.failures += 1
                    raise JudgeError("the Jev API key was rejected (401)")
                if response.status == 422:
                    usage.failures += 1
                    detail = (await response.text())[:300]
                    raise JudgeError(f"Jev rejected the request (422): {detail}")
                if response.status >= 400:
                    usage.failures += 1
                    detail = (await response.text())[:300]
                    raise JudgeError(f"Jev returned HTTP {response.status}: {detail}")
                payload = await response.json()
        except aiohttp.ClientError as err:
            last_error = str(err)
            if attempt < MAX_ATTEMPTS:
                await asyncio.sleep(_retry_delay(None, attempt))
                continue
        except TimeoutError:
            last_error = f"no answer within {REQUEST_TIMEOUT:.0f}s"
        else:
            usage.record(payload.get("usage"))
            _LOGGER.debug(
                "Jev answered %d question(s) using %s input tokens",
                len(questions),
                (payload.get("usage") or {}).get("input_tokens"),
            )
            answers = payload.get("answers") or {}
            return {
                key_: Answer(raw=answers.get(key_) or {})
                for key_ in questions
            }

    usage.failures += 1
    raise JudgeError(f"Jev did not answer ({last_error}); key from {source}")


def _retry_delay(response: Any, attempt: int) -> float:
    """Honour ``retry-after`` when the response offers one, else back off."""
    if response is not None:
        try:
            header = response.headers.get("retry-after")
        except (AttributeError, TypeError):
            header = None
        if header:
            try:
                return min(10.0, max(0.0, float(header)))
            except ValueError:
                pass
    return min(4.0, 0.5 * 2 ** (attempt - 1))
