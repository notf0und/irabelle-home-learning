"""the Irabelle Home Learning integration.

Lets the voice assistant teach itself. It goes into the pipeline as the
conversation agent and passes the sentence on to the real one:

    pipeline -> irabelle_home_learning -> your agent -> its fallback

A phrasing the local matcher could not understand comes back bound to the intent
that answered it - or, for a question nothing local can answer, as a learned
intent with a template - so the same words are answered locally next time. No
component this repo does not own is modified: this one only has to be in the
pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol
from types import MappingProxyType

from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.const import EVENT_HOMEASSISTANT_STARTED
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.dispatcher import (
    async_dispatcher_connect,
    async_dispatcher_send,
)
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.helpers.typing import ConfigType

from .autolearn import (
    async_reconsider,
    async_review_learning,
    AUDIT_EVERY,
    async_audit_one,
    async_change_response,
    async_check_installed,
    async_sweep_meta,
    async_route_learning,
)
from .combine import async_combine
from .agents import SUBENTRY_AGENT
from .const import AGENT_KEYS, CONF_ALLOWED_PLATFORMS, DOMAIN, SIGNAL_UPDATED
from .conversation import delegate_agent
from .fileio import async_run
from .upstream import bundled as upstream_bundle
from .judge import Answer, Choice, JudgeError, Noul, Score, async_ask, describe, has_judge
from .intents import (
    async_duplicate_verdicts,
    async_forget_intent,
    async_learn_intent,
    async_merge_intents,
    async_rename_intent,
    async_consolidate_intents,
    backfill_responses,
    duplicate_report,
    export_intent_script,
    refresh_export,
    register_intent,
    release_intents,
    restore_intents,
)
from .learner import (
    LearnerData,
    Observation,
    LearnerError,
    adopt_disarmed,
    allowed_platforms,
    arm_learner_files,
    async_forget,
    async_intent_names,
    async_learn,
    disarm_learner_files,
    async_load,
    async_prune_orphans,
    quarantine_broken,
    async_save,
    auto_learn,
    language_of,
    min_occurrences,
    normalize_sentence,
    normalize_sentence_files,
    purge_sentence_files,
)

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[str] = ["conversation", "sensor"]
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

LEARN_SCHEMA = vol.Schema(
    {
        vol.Required("utterances"): vol.All(cv.ensure_list, [cv.string]),
        vol.Required("intent"): cv.string,
        vol.Optional("slots"): dict,
        vol.Optional("note"): cv.string,
        vol.Optional("response"): cv.string,
    }
)
FORGET_SCHEMA = vol.Schema({vol.Required("utterance"): cv.string})
LEARN_INTENT_SCHEMA = vol.Schema(
    {
        vol.Required("name"): cv.string,
        vol.Required("template"): cv.string,
        vol.Required("utterances"): vol.All(cv.ensure_list, [cv.string]),
        vol.Optional("description"): cv.string,
        vol.Optional("note"): cv.string,
        vol.Optional("response"): cv.string,
        # The word-classes the wording varies by, e.g. {"day": ["today", "tomorrow",
        # "monday", ...]}: one intent that answers any weekday, instead of one intent per
        # weekday. Usable from the service so the behaviour can be proved without a model.
        vol.Optional("slots"): dict,
    }
)
FORGET_INTENT_SCHEMA = vol.Schema({vol.Required("name"): cv.string})
COMBINE_SCHEMA = vol.Schema(
    {
        vol.Optional("intent"): cv.string,
        vol.Optional("apply", default=False): cv.boolean,
    }
)
RENAME_INTENT_SCHEMA = vol.Schema(
    {vol.Required("name"): cv.string, vol.Required("to"): cv.string}
)
MERGE_SCHEMA = vol.Schema(
    {
        vol.Required("into"): cv.string,
        vol.Required("others"): vol.All(cv.ensure_list, [cv.string]),
        vol.Optional("template"): cv.string,
        vol.Optional("reply"): cv.string,
        # Wording that is not an intent of its own - a phrase bound to a built-in intent -
        # so a merge can reclaim it instead of leaving a second answer behind.
        vol.Optional("phrasings"): vol.All(cv.ensure_list, [cv.string]),
        # Widens the target's own word-class, so "today" and "tomorrow" can join a weekday
        # list and all of them become one {day} sentence.
        vol.Optional("slots"): dict,
        # Reports the plan without changing anything unless asked, the same way `review`
        # does: asking what would be merged must not be able to delete an intent.
        vol.Optional("apply", default=False): cv.boolean,
    }
)

type HomeLearningConfigEntry = ConfigEntry[LearnerData]


def _first_entry(hass: HomeAssistant) -> HomeLearningConfigEntry | None:
    for entry in hass.config_entries.async_entries(DOMAIN):
        if getattr(entry, "runtime_data", None) is not None:
            return entry
    return None


def _async_register_services(hass: HomeAssistant) -> None:
    """Register the services once, whatever the number of entries."""
    if hass.services.has_service(DOMAIN, "learn"):
        return

    async def _learn(call: ServiceCall) -> dict[str, Any]:
        entry = _first_entry(hass)
        if entry is None:
            raise LearnerError("the Irabelle Home Learning integration is not set up")
        result = await async_learn(
            hass,
            entry,
            utterances=call.data["utterances"],
            intent=call.data["intent"],
            slots=call.data.get("slots"),
            note=call.data.get("note"),
            response=call.data.get("response"),
            source="service",
        )
        async_dispatcher_send(hass, SIGNAL_UPDATED)
        return result

    async def _forget(call: ServiceCall) -> dict[str, Any]:
        entry = _first_entry(hass)
        if entry is None:
            raise LearnerError("the Irabelle Home Learning integration is not set up")
        result = await async_forget(hass, entry, call.data["utterance"])
        async_dispatcher_send(hass, SIGNAL_UPDATED)
        return result

    async def _validate(call: ServiceCall) -> dict[str, Any]:
        """Re-run the safety pass: quarantine broken files, reload, and look at the house now.

        The checks are *recomputed*, not read back. They also run once at startup, but that is
        the worst moment to ask: entities from integrations that load later are not in
        `hass.states` yet, so a reply that reads one renders as nothing and passes as clean -
        measured, `PhoneLastSeenAgo` was reported clean at every restart and answered "Unknown,
        your iPhone Alex was last seen" once the phone sensors were there. Somebody asking
        `validate` is asking about now.
        """
        entry = _first_entry(hass)
        if entry is None:
            raise LearnerError("the Irabelle Home Learning integration is not set up")
        data: LearnerData = entry.runtime_data
        data.quarantined = await async_run(
            hass, quarantine_broken, hass, language_of(entry)
        )
        await hass.services.async_call("conversation", "reload", {}, blocking=True)
        data.orphans = await async_prune_orphans(hass, entry)
        data.drift = await async_check_installed(hass, entry)
        await async_save(hass, entry)
        async_dispatcher_send(hass, SIGNAL_UPDATED)
        return {
            "quarantined": data.quarantined,
            "orphans": data.orphans,
            "consolidated": data.consolidated,
            "drift": data.drift,
            "repeats": data.repeats,
            "spend": data.spend,
        }

    async def _learn_intent(call: ServiceCall) -> dict[str, Any]:
        entry = _first_entry(hass)
        if entry is None:
            raise LearnerError("the Irabelle Home Learning integration is not set up")
        result = await async_learn_intent(
            hass,
            entry,
            name=call.data["name"],
            template=call.data["template"],
            utterances=call.data["utterances"],
            description=call.data.get("description"),
            note=call.data.get("note"),
            source="service",
            response=call.data.get("response"),
            slots=call.data.get("slots"),
        )
        async_dispatcher_send(hass, SIGNAL_UPDATED)
        return result

    async def _consolidate(call: ServiceCall) -> dict[str, Any]:
        """Move every intent's sentences and reply into one file, its own."""
        entry = _first_entry(hass)
        if entry is None:
            raise LearnerError("the Irabelle Home Learning integration is not set up")
        result = await async_consolidate_intents(hass, entry)
        await hass.services.async_call("conversation", "reload", {}, blocking=True)
        async_dispatcher_send(hass, SIGNAL_UPDATED)
        return result

    async def _export(call: ServiceCall) -> dict[str, Any]:
        """Write intent_script stubs, so the learned intents outlive this integration."""
        entry = _first_entry(hass)
        if entry is None:
            raise LearnerError("the Irabelle Home Learning integration is not set up")
        return await async_run(hass, export_intent_script, hass, entry)

    async def _duplicates(call: ServiceCall) -> dict[str, Any]:
        """Which learned intents read the same entity - and, with the judge, which are one question.

        The grouping is evidence and costs nothing; the verdict is a judgement, and it is the
        part that says whether a merge would lose an answer. Asked only when the judge is
        configured, and the report says so when it is not.
        """
        entry = _first_entry(hass)
        if entry is None:
            raise LearnerError("the Irabelle Home Learning integration is not set up")
        report = duplicate_report(hass, entry)
        if not has_judge(entry):
            report["same_question"] = []
            report["same_question_note"] = (
                "no classification model is configured, so nothing here decides whether two of these are "
                "one question; the grouping above is the evidence for that decision"
            )
            return report
        try:
            report["same_question"] = await async_duplicate_verdicts(hass, entry, report)
        except JudgeError as err:
            report["same_question"] = []
            report["same_question_note"] = f"the judge could not be asked: {err}"
        return report

    async def _fold(call: ServiceCall) -> dict[str, Any]:
        """Fold groups of phrasings into templates. A dry run unless `apply`."""
        entry = _first_entry(hass)
        if entry is None:
            raise LearnerError("the Irabelle Home Learning integration is not set up")
        from .fold import async_fold  # noqa: PLC0415 - imports autolearn

        return await async_fold(
            hass, entry, call.data.get("intent"), apply=bool(call.data.get("apply"))
        )

    async def _rename_intent(call: ServiceCall) -> dict[str, Any]:
        """Give a learned intent another name, and its file the matching one."""
        entry = _first_entry(hass)
        if entry is None:
            raise LearnerError("the Irabelle Home Learning integration is not set up")
        result = await async_rename_intent(
            hass, entry, call.data["name"], call.data["to"]
        )
        async_dispatcher_send(hass, SIGNAL_UPDATED)
        return result

    async def _merge(call: ServiceCall) -> dict[str, Any]:
        """Fold learned intents into one, so one question keeps one answer."""
        entry = _first_entry(hass)
        if entry is None:
            raise LearnerError("the Irabelle Home Learning integration is not set up")
        result = await async_merge_intents(
            hass,
            entry,
            into=call.data["into"],
            others=call.data["others"],
            template=call.data.get("template"),
            reply=call.data.get("reply"),
            phrasings=call.data.get("phrasings"),
            slots=call.data.get("slots"),
            apply=call.data.get("apply", False),
        )
        async_dispatcher_send(hass, SIGNAL_UPDATED)
        return result

    async def _sweep(call: ServiceCall) -> dict[str, Any]:
        """Mark recorded requests that are about the assistant rather than for the house.

        Only what was recorded before the gate existed; new ones are never recorded.
        """
        entry = _first_entry(hass)
        if entry is None:
            raise LearnerError("the Irabelle Home Learning integration is not set up")
        result = await async_sweep_meta(hass, entry)
        async_dispatcher_send(hass, SIGNAL_UPDATED)
        return result

    async def _audit(call: ServiceCall) -> dict[str, Any]:
        """Re-judge one installed intent now, the next in line, with nothing changed.

        The same check runs on a timer; this is for wanting an answer without waiting for
        it, and for seeing what the timer would have looked at.
        """
        entry = _first_entry(hass)
        if entry is None:
            raise LearnerError("the Irabelle Home Learning integration is not set up")
        finding = await async_audit_one(hass, entry)
        if finding is None:
            return {"audited": None, "reason": "no learned intents, or no judge configured"}
        return finding

    async def _set_response(call: ServiceCall) -> dict[str, Any]:
        """Change how a learned intent answers, without re-learning it."""
        entry = _first_entry(hass)
        if entry is None:
            raise LearnerError("the Irabelle Home Learning integration is not set up")
        return await async_change_response(
            hass, entry, call.data["name"], call.data["response"]
        )

    async def _forget_intent(call: ServiceCall) -> dict[str, Any]:
        entry = _first_entry(hass)
        if entry is None:
            raise LearnerError("the Irabelle Home Learning integration is not set up")
        result = await async_forget_intent(hass, entry, call.data["name"])
        async_dispatcher_send(hass, SIGNAL_UPDATED)
        return result

    async def _tools(call: ServiceCall) -> dict[str, Any]:
        """Report the tool names the Assist API exposes for a platform.

        The learner tools are only contributed for platforms allowed to teach, so
        this takes the platform to ask as - the point is to prove the gate works.
        """
        from homeassistant.helpers import llm as llm_helper

        entry = _first_entry(hass)
        allowed = allowed_platforms(entry) if entry is not None else []
        platform = call.data.get("platform") or (allowed[0] if allowed else DOMAIN)
        context = llm_helper.LLMContext(
            platform=platform,
            context=None,
            language=None,
            assistant="conversation",
            device_id=None,
        )
        instance = await llm_helper.async_get_api(hass, "assist", context)
        names = sorted(tool.name for tool in instance.tools)
        return {
            "platform": platform,
            "allowed_platforms": allowed,
            "may_teach": not allowed or platform in allowed,
            "count": len(names),
            "learner_tools": [n for n in names if n.startswith(f"{DOMAIN}__")],
            "tools": names,
        }

    async def _combine(call: ServiceCall) -> dict[str, Any]:
        """Fold the phrasings of an intent into one template per wording, or say why not.

        Dry by default: what the judge said about each pair is worth reading before the house
        starts answering with fewer lines. `apply: true` writes the folds.
        """
        entry = _first_entry(hass)
        if entry is None:
            raise LearnerError("the Irabelle Home Learning integration is not set up")
        return await async_combine(
            hass,
            entry,
            call.data.get("intent"),
            apply=bool(call.data.get("apply")),
        )

    async def _forget_all(call: ServiceCall) -> dict[str, Any]:
        """Give the house amnesia: remove every learned intent and sentence.

        The point of this service is to be able to prove the house can learn
        again from nothing, without hand-deleting files.
        """
        entry = _first_entry(hass)
        if entry is None:
            raise LearnerError("the Irabelle Home Learning integration is not set up")
        data: LearnerData = entry.runtime_data
        intents = [spec.name for spec in data.intents]
        for name in intents:
            await async_forget_intent(hass, entry, name)
        files = await async_run(hass, purge_sentence_files, hass, language_of(entry))
        data.phrases = []
        data.observations = []
        data.last_result = {"forgotten_intents": intents, "files_removed": files}
        await async_save(hass, entry)
        await hass.services.async_call("conversation", "reload", {}, blocking=True)
        async_dispatcher_send(hass, SIGNAL_UPDATED)
        _LOGGER.warning(
            "Irabelle Home Learning forgot %d learned intents and removed %d sentence files",
            len(intents),
            len(files),
        )
        return data.last_result

    async def _observations(call: ServiceCall) -> dict[str, Any]:
        """Report what the house has heard that it could not answer locally."""
        entry = _first_entry(hass)
        if entry is None:
            raise LearnerError("the Irabelle Home Learning integration is not set up")
        data: LearnerData = entry.runtime_data
        wanted = str(call.data.get("state") or "").strip().lower()
        rows = [
            row.as_dict() for row in data.observations if not wanted or row.state == wanted
        ]
        return {
            "auto_learn": auto_learn(entry),
            "min_occurrences": min_occurrences(entry),
            "delegate_agent": delegate_agent(hass, entry, None),
            "turns_seen": data.turns_seen,
            "count": len(rows),
            "observations": rows[-40:],
        }

    def _question(spec: dict[str, Any]) -> Choice | Noul | Score:
        """Build a question from the diagnostic service's plain form."""
        kind = str(spec.get("type") or "").lower()
        instructions = spec.get("instructions")
        if not instructions:
            raise LearnerError("a question needs instructions")
        if kind == "choice":
            criteria = spec.get("criteria")
            if not isinstance(criteria, dict) or not criteria:
                raise LearnerError("a choice needs criteria, as a mapping of option: meaning")
            return Choice(instructions, criteria)
        if kind == "noul":
            return Noul(instructions, true=spec.get("true"), false=spec.get("false"))
        if kind == "score":
            criteria = spec.get("criteria")
            if not isinstance(criteria, list) or len(criteria) < 2:
                raise LearnerError("a score needs at least two levels in criteria")
            return Score(instructions, [str(level) for level in criteria])
        raise LearnerError(f"unknown question type {kind!r}")

    async def _judge(call: ServiceCall) -> dict[str, Any]:
        """Ask the judge directly. Diagnostics: it reports itself when asked nothing.

        Kept as a service on purpose. Every judgement the learner makes is a decision
        about money and about what the house will say, so being able to ask the same
        question by hand — and see the probabilities behind the answer — is how the
        thresholds get set from data instead of taste.
        """
        entry = _first_entry(hass)
        if entry is None:
            raise LearnerError("the Irabelle Home Learning integration is not set up")

        specs = call.data.get("questions")
        if not specs:
            return {
                **describe(hass, entry),
                "today": dict(entry.runtime_data.spend),
                "repeats": dict(entry.runtime_data.repeats),
            }

        state = call.data.get("state")
        if state is None:
            raise LearnerError("a state is required when asking questions")
        questions = {
            str(name): _question(spec) for name, spec in dict(specs).items()
        }
        try:
            answers = await async_ask(hass, entry, state, questions, rate_limited=False)
        except JudgeError as err:
            raise LearnerError(str(err)) from err

        report = describe(hass, entry)
        report["today"] = dict(entry.runtime_data.spend)
        return {
            "model": report["model"],
            "answers": {name: _answer_json(answer) for name, answer in answers.items()},
            "today": report["today"],
            "where": {
                "confidence": "0-1, a derived statistic; calibration is group-level",
                "margin": "top probability minus the next; a near-tie is not a decision",
                "noul": "probability that the statement is true; no confidence field",
            },
        }

    async def _review(call: ServiceCall) -> dict[str, Any]:
        """Have a model read back what a learned intent will say, and fix it.

        Every new learning is reviewed as it happens; this is the same check on
        demand, for auditing what is already installed. It reports by default and
        changes nothing unless asked - asking the house to look at itself must not be
        able to delete an intent the person only wanted inspected. Pass `apply: true`
        to let it make the corrections it describes.
        """
        entry = _first_entry(hass)
        if entry is None:
            raise LearnerError("the Irabelle Home Learning integration is not set up")
        data: LearnerData = entry.runtime_data
        apply = bool(call.data.get("apply"))
        wanted = str(call.data.get("intent") or "").strip()
        names = [wanted] if wanted else [spec.name for spec in data.intents]
        outcomes: list[dict[str, Any]] = []
        for name in names:
            outcome = await async_review_learning(hass, entry, name, apply=apply)
            outcomes.append({"intent": name, "outcome": outcome})
        if apply:
            async_dispatcher_send(hass, SIGNAL_UPDATED)
        return {"reviewed": len(outcomes), "applied": apply, "outcomes": outcomes}

    async def _route(call: ServiceCall) -> dict[str, Any]:
        """Report what the router would decide, without learning anything.

        The router decides whether a turn becomes a binding, a new template, or nothing
        at all, and it decides it with a model. Being able to ask the same question by
        hand - and see the probabilities behind the answer - is how its thresholds get
        set from data rather than taste.
        """
        entry = _first_entry(hass)
        if entry is None:
            raise LearnerError("the Irabelle Home Learning integration is not set up")
        utterance = " ".join(str(call.data.get("utterance") or "").split())
        if not utterance:
            raise LearnerError("the router needs an utterance")
        row = Observation(
            utterance=utterance,
            normalized=normalize_sentence(utterance),
            agent_id="service",
            tools=[],
            answer=str(call.data.get("answer") or ""),
            targets=[
                target
                for target in (call.data.get("targets") or [])
                if isinstance(target, str)
            ],
        )
        route = await async_route_learning(hass, entry, row)
        if route is None:
            return {
                "checked": False,
                "decision": None,
                "reason": "no router: no classification model is configured, or it could not be reached",
            }
        return route.as_dict()

    async def _reconsider(call: ServiceCall) -> dict[str, Any]:
        """Ask the generaliser to look again at what it passed over."""
        entry = _first_entry(hass)
        if entry is None:
            raise LearnerError("the Irabelle Home Learning integration is not set up")
        return await async_reconsider(
            hass,
            entry,
            utterance=call.data.get("utterance"),
        )

    # Every one of these returns a result dict, so they must advertise response
    # support or a caller using return_response gets "does not return responses".
    hass.services.async_register(
        DOMAIN, "learn", _learn, schema=LEARN_SCHEMA, supports_response=True
    )
    hass.services.async_register(
        DOMAIN, "forget", _forget, schema=FORGET_SCHEMA, supports_response=True
    )
    hass.services.async_register(
        DOMAIN, "learn_intent", _learn_intent, schema=LEARN_INTENT_SCHEMA,
        supports_response=True,
    )
    hass.services.async_register(
        DOMAIN, "forget_intent", _forget_intent, schema=FORGET_INTENT_SCHEMA,
        supports_response=True,
    )
    hass.services.async_register(
        DOMAIN, "consolidate", _consolidate, schema=vol.Schema({}), supports_response=True
    )
    hass.services.async_register(
        DOMAIN, "duplicates", _duplicates, schema=vol.Schema({}), supports_response=True
    )
    hass.services.async_register(
        DOMAIN, "export", _export, schema=vol.Schema({}), supports_response=True
    )
    hass.services.async_register(
        DOMAIN, "combine", _combine, schema=COMBINE_SCHEMA, supports_response=True
    )
    hass.services.async_register(
        DOMAIN, "fold", _fold, schema=COMBINE_SCHEMA, supports_response=True
    )
    hass.services.async_register(
        DOMAIN, "merge", _merge, schema=MERGE_SCHEMA, supports_response=True
    )
    hass.services.async_register(
        DOMAIN, "rename_intent", _rename_intent, schema=RENAME_INTENT_SCHEMA,
        supports_response=True,
    )
    hass.services.async_register(
        DOMAIN, "audit", _audit, schema=vol.Schema({}), supports_response=True
    )
    hass.services.async_register(
        DOMAIN, "sweep", _sweep, schema=vol.Schema({}), supports_response=True
    )
    hass.services.async_register(
        DOMAIN,
        "set_response",
        _set_response,
        schema=vol.Schema(
            {vol.Required("name"): cv.string, vol.Required("response"): cv.string}
        ),
        supports_response=True,
    )
    hass.services.async_register(
        DOMAIN, "validate", _validate, schema=vol.Schema({}), supports_response=True
    )
    hass.services.async_register(
        DOMAIN,
        "tools",
        _tools,
        schema=vol.Schema({vol.Optional("platform"): cv.string}),
        supports_response=True,
    )
    hass.services.async_register(
        DOMAIN,
        "observations",
        _observations,
        schema=vol.Schema({vol.Optional("state"): cv.string}),
        supports_response=True,
    )
    hass.services.async_register(
        DOMAIN,
        "forget_all",
        _forget_all,
        schema=vol.Schema({}),
        supports_response=True,
    )
    hass.services.async_register(
        DOMAIN,
        "judge",
        _judge,
        schema=vol.Schema(
            {
                vol.Optional("state"): vol.Any(str, dict, list),
                vol.Optional("questions"): dict,
            }
        ),
        supports_response=True,
    )
    hass.services.async_register(
        DOMAIN,
        "review",
        _review,
        schema=vol.Schema(
            {
                vol.Optional("intent"): cv.string,
                vol.Optional("apply"): cv.boolean,
            }
        ),
        supports_response=True,
    )
    hass.services.async_register(
        DOMAIN,
        "route",
        _route,
        schema=vol.Schema(
            {
                vol.Required("utterance"): cv.string,
                vol.Optional("answer"): cv.string,
                vol.Optional("targets"): vol.All(cv.ensure_list, [cv.string]),
            }
        ),
        supports_response=True,
    )
    hass.services.async_register(
        DOMAIN,
        "reconsider",
        _reconsider,
        schema=vol.Schema({vol.Optional("utterance"): cv.string}),
        supports_response=True,
    )


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the integration (services are registered per entry)."""
    hass.data.setdefault(DOMAIN, {})
    return True


async def async_setup_entry(hass: HomeAssistant, entry: HomeLearningConfigEntry) -> bool:
    """Load provenance, run the safety pass, and start the agent and sensors.

    Wrapped so a failure is impossible to miss. Home Assistant logs a failed setup at
    ERROR, and a logger configuration of `default: critical` - which is a reasonable thing
    to run - hides it, leaving an integration that is simply absent with nothing said.
    """
    try:
        return await _async_setup_entry(hass, entry)
    except Exception:
        _LOGGER.critical("Irabelle Home Learning could not start", exc_info=True)
        raise


async def _reload_sentences(hass: HomeAssistant) -> None:
    """Make the matcher re-read `custom_sentences/`, because arming and disarming rename files.

    A file that steps aside is only out of the matcher's reach once the matcher has read the
    directory again - it caches what it found at the last load. Without this, a disabled
    integration has released the handler that answered a phrase while the matcher still
    matches the sentence from memory, and the answer is `intent-failed` instead of the agent:
    measured, once, on the live house. The other direction is quieter and just as wrong - a
    file put back stays invisible until the next restart, so the phrase it answers stops being
    local.
    """
    try:
        await hass.services.async_call("conversation", "reload", {}, blocking=True)
    except Exception:  # noqa: BLE001 - no conversation integration means nothing to reload
        _LOGGER.debug("Irabelle Home Learning could not reload conversation", exc_info=True)


async def _async_setup_entry(hass: HomeAssistant, entry: HomeLearningConfigEntry) -> bool:
    data = await async_load(hass, entry)
    entry.runtime_data = data
    # Home Assistant's own intents are read from disk once and cached; read them here, in the
    # executor, rather than on the event loop the first time a phrase is learned.
    await hass.async_add_executor_job(upstream_bundle, language_of(entry))

    # Validate before Home Assistant ever loads these sentences.
    data.quarantined = await async_run(hass, quarantine_broken, hass, language_of(entry))

    # A sentence written with punctuation matches nothing, so any written before
    # that was understood is rewritten into its matchable form here.
    await async_run(hass, normalize_sentence_files, hass, language_of(entry))

    # Re-register every learned intent, so learned question types survive a restart.
    restore_intents(hass, data)

    # And take back anything a previous run stepped aside: the files themselves say what they
    # are, so a store that is empty - a fresh install, a removed and re-added integration, a
    # folder copied from another house - still gets its learnings back.
    language = language_of(entry)
    known = {spec.name for spec in data.intents}
    adopted = [
        spec
        for spec in await async_run(hass, adopt_disarmed, hass, language)
        if spec.name not in known
    ]
    if adopted:
        data.intents.extend(adopted)
        known.update(spec.name for spec in adopted)
        for spec in adopted:
            register_intent(hass, spec)
        _LOGGER.warning(
            "Irabelle Home Learning adopted %d learned intent(s) from their own sentence files: %s",
            len(adopted),
            ", ".join(spec.name for spec in adopted),
        )
        await async_save(hass, entry)
    # `known` is every intent that has a handler, not only ours: a file bound to a built-in
    # intent is answered by Home Assistant itself, and leaving those aside because their name
    # is not in this integration's list took the air conditioner and monitor wordings out of
    # service for the length of a reload.
    handled = known | set(await async_intent_names(hass))
    if await async_run(hass, arm_learner_files, hass, language, handled):
        await _reload_sentences(hass)

    # One file per intent, before anything looks for what is wrong: the same intent in
    # three files is how a stray sentence hides, and a tidy file is what a person reads.
    data.consolidated = await async_consolidate_intents(hass, entry)

    # And re-judge one of them every few hours, for the vocabulary drift that mechanical
    # checks cannot see. Screening only: nothing here edits the house on a timer.
    entry.async_on_unload(
        async_track_time_interval(
            hass,
            lambda _now: hass.async_create_task(async_audit_one(hass, entry)),
            AUDIT_EVERY,
        )
    )

    # Then look for sentences whose intent nothing answers. Ours are removed; anything in
    # a file this integration does not own is reported instead, because a sentence with no
    # handler fails the whole pipeline for that phrase rather than merely going unanswered.
    # Deferred to the end of startup, both of them: `intent_script` registers its handlers
    # after this runs, so pruning now would call real intents orphans - and entities from
    # integrations that load later are not in `hass.states` yet, so every template that
    # reads one looks like it reads something that does not exist. Both checks are about
    # what is *there*, and nothing is there yet.
    async def _startup_checks(_event: Any = None) -> None:
        try:
            # Everything has registered by now, so this is the first moment the question
            # "is there a handler for this sentence?" can be answered properly. Files that
            # were left aside for want of a handler come back here.
            #
            # Ours are registered *again* first, and that is not redundant: `intent_script`
            # registers its handlers after this integration's setup (which is why the checks
            # below are deferred in the first place), and an exported registration file names
            # these same intents. Home Assistant keeps one handler per intent name, so the
            # stub - an empty action that says nothing - was left holding ours, and the reply
            # lost everything only this integration can put in it: measured on a restart,
            # "who am I" answered "I don't know who is asking" instead of naming the speaker.
            restore_intents(hass, data)
            await async_run(
                hass,
                arm_learner_files,
                hass,
                language,
                set(await async_intent_names(hass)),
            )
            data.orphans = await async_prune_orphans(hass, entry)
            data.drift = await async_check_installed(hass, entry)
            # One save, so the spend and drift readings are in the store and visible
            # before the first learning of the day.
            await async_save(hass, entry)
        except Exception:  # noqa: BLE001 - a listener that dies silently helps nobody
            _LOGGER.critical("Irabelle Home Learning could not run its startup checks", exc_info=True)
            return
        async_dispatcher_send(hass, SIGNAL_UPDATED)

    if hass.is_running:
        await _startup_checks()
    else:
        hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STARTED, _startup_checks)

    # Put every reply in its sentence file the first time this runs, so the file shows
    # what the house will say rather than only the sentences that reach it.
    if backfilled := await async_run(hass, backfill_responses, hass, entry):
        _LOGGER.info("Irabelle Home Learning wrote %d reply/replies into their sentence file(s)", backfilled)

    _async_register_services(hass)
    # An install that ran `export` has a registration file to keep honest, and every change to
    # what is installed announces itself on this signal - so the file names the intents that
    # exist rather than the ones that existed the day it was written. No file, no write.
    entry.async_on_unload(
        async_dispatcher_connect(
            hass, SIGNAL_UPDATED, lambda: refresh_export(hass, entry)
        )
    )
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    # Adding, changing or removing an agent changes which conversation entities exist.
    entry.async_on_unload(entry.add_update_listener(_async_reload_entry))
    async_dispatcher_send(hass, SIGNAL_UPDATED)
    return True


async def _async_reload_entry(hass: HomeAssistant, entry: HomeLearningConfigEntry) -> None:
    await hass.config_entries.async_reload(entry.entry_id)


async def async_migrate_entry(hass: HomeAssistant, entry: HomeLearningConfigEntry) -> bool:
    """Version 1 was one agent; version 2 is a hub of learning with agents as subentries.

    The one agent becomes the first subentry, named after the entry, and its conversation
    entity and device move onto it - same entity id, so every pipeline that picked it keeps
    working. The hub keeps what belongs to the house: language, learning, the export path.
    `allowed_platforms` is dropped: the teaching tools reach only an agent whose LLM API
    setting includes them, which is already the opt-in.
    """
    if entry.version > 2:
        return False
    if entry.version == 1:
        options = dict(entry.options)
        agent = {key: options.pop(key) for key in AGENT_KEYS if key in options}
        options.pop(CONF_ALLOWED_PLATFORMS, None)
        subentry = ConfigSubentry(
            data=MappingProxyType(agent),
            subentry_type=SUBENTRY_AGENT,
            title=entry.title,
            unique_id=None,
        )
        hass.config_entries.async_add_subentry(entry, subentry)

        entities = er.async_get(hass)
        if entity_id := entities.async_get_entity_id(
            "conversation", DOMAIN, f"{entry.entry_id}_conversation"
        ):
            entities.async_update_entity(
                entity_id,
                config_subentry_id=subentry.subentry_id,
                new_unique_id=subentry.subentry_id,
            )
        devices = dr.async_get(hass)
        if device := devices.async_get_device(identifiers={(DOMAIN, entry.entry_id)}):
            # The sensors stay on the hub's device; the agent gets one of its own.
            devices.async_update_device(device.id, name=entry.title)
        hass.config_entries.async_update_entry(entry, options=options, version=2)
        _LOGGER.info(
            "Irabelle Home Learning moved its agent %r into a subentry", entry.title
        )
    return True


async def async_unload_entry(hass: HomeAssistant, entry: HomeLearningConfigEntry) -> bool:
    """Unload the platforms and release learned intent handlers."""
    unloaded = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unloaded and getattr(entry, "runtime_data", None) is not None:
        release_intents(hass, entry.runtime_data)
        # Our handlers are gone, so a sentence nothing else answers is a sentence recognition
        # reports an error for. Those files step aside instead - except the ones something
        # *else* still answers, which is what the `export` service writes an intent_script stub
        # for. `arm_learner_files` puts the rest back on the next setup.
        if await async_run(
            hass,
            disarm_learner_files,
            hass,
            language_of(entry),
            set(await async_intent_names(hass)),
        ):
            await _reload_sentences(hass)
    return unloaded


def _answer_json(answer: Answer) -> dict[str, Any]:
    """An answer in the shape a person reading a service response wants."""
    out: dict[str, Any] = {"type": answer.type}
    if answer.noul is not None:
        out["noul"] = answer.noul
    if answer.choice is not None:
        out["choice"] = answer.choice
        out["confidence"] = answer.confidence
        out["margin"] = answer.margin()
        out["probabilities"] = answer.probabilities
    if answer.score is not None:
        out["score"] = answer.score
        out["confidence"] = answer.confidence
        out["probabilities"] = answer.probabilities
    return out
