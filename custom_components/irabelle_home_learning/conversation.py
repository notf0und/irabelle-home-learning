"""Irabelle Home Learning as a conversation agent: watch, delegate, learn.

Home Assistant lets any integration put a conversation agent into a pipeline, and
a pipeline's agent may hand the sentence on to another one. That is the whole
hook this needs:

    pipeline -> irabelle_home_learning -> your agent -> its fallback

This agent changes nothing about the answer. It passes the sentence to the
configured agent, returns that agent's result untouched, and reads the finished
turn on the way past. Nothing below it has to know it is being watched, so no
component this repo does not own has to be patched - and an update to the agent below
cannot break the learning.

Two things have to be advertised on the agent's own behalf, or standing in front
would quietly change how the pipeline behaves:

* ``supports_streaming`` - the pipeline only streams the answer into TTS when the
  pipeline's agent claims it. Sitting in front of a streaming agent while denying
  it would cost the whole chain its streaming latency.
* ``CONTROL`` - the pipeline uses this to decide whether to answer state
  questions locally instead of letting the agent do it. The chain below can
  control the house, and Jev answers those questions better than the fallback
  matcher, so the flag is kept.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.const import MATCH_ALL
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import intent
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .agents import AgentView, agent_subentries, settings
from .autolearn import async_consider_turn
from .const import (
    CONF_DELEGATE_AGENT,
    DEFAULT_DELEGATE_AGENT,
    DOMAIN,
    FALLBACK_AGENT,
)
from .learner import LearnerData, auto_learn
from .observer import Turn, ensure_answer_in_log, speech_of, turn_from_result

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """One watching conversation agent per agent subentry."""
    for subentry in agent_subentries(entry):
        async_add_entities(
            [HomeLearningConversationEntity(entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


def delegate_agent(
    hass: HomeAssistant, entry: ConfigEntry, self_id: str | None
) -> str | None:
    """Return the agent this one hands the sentence to.

    Falling back rather than failing matters here: a voice pipeline whose agent
    cannot find its delegate would stop answering altogether, and the safe
    version of that is Home Assistant's own agent.
    """
    configured = str(settings(entry).get(CONF_DELEGATE_AGENT) or "").strip()
    for candidate in (configured or DEFAULT_DELEGATE_AGENT, FALLBACK_AGENT):
        if not candidate or candidate == self_id:
            continue
        if conversation.async_get_agent(hass, candidate) is None:
            continue
        if candidate != (configured or DEFAULT_DELEGATE_AGENT):
            _LOGGER.warning(
                "Irabelle Home Learning: %r is not available, delegating to %s instead",
                configured or DEFAULT_DELEGATE_AGENT,
                candidate,
            )
        return candidate
    return None


def _error_result(
    user_input: conversation.ConversationInput, message: str
) -> conversation.ConversationResult:
    """Return a spoken sentence rather than letting the pipeline fail."""
    response = intent.IntentResponse(language=user_input.language or "")
    response.async_set_error(intent.IntentResponseErrorCode.UNKNOWN, message)
    return conversation.ConversationResult(
        response=response, conversation_id=user_input.conversation_id
    )


#: What is said when nothing below could answer. Short, human, and never the exception
#: text: an internal error read aloud is exactly what this agent exists to prevent.
NO_ANSWER = "I could not reach my assistant just now. Please try again in a moment."


def _did_not_answer(result: conversation.ConversationResult | None) -> str | None:
    """Why this is not an answer anyone should hear, or None if it is one.

    An agent can report an internal failure by *returning* an error response instead of
    raising, so watching for exceptions is not enough. `deepseek_conversation` does
    exactly that when a tool API it needs has not finished registering after a restart:
    it answers with `FAILED_TO_HANDLE` and speaks the exception text, which is how
    "Error preparing context: Error getting LLM API [...]" reaches a person's kitchen.

    Only `FAILED_TO_HANDLE` counts as a failure. `NO_INTENT_MATCH` and
    `NO_VALID_TARGETS` are legitimate spoken answers - "I could not find a device called
    that" is something a person can act on - and swallowing those would replace a real
    answer with a shrug.
    """
    if result is None:
        return "no result"
    response = result.response
    # `response_type` says the answer is an error; `error_code` says which kind. Both are
    # needed: an answer can be an error and still be the right thing to say.
    if (
        response.response_type is intent.IntentResponseType.ERROR
        and response.error_code is intent.IntentResponseErrorCode.FAILED_TO_HANDLE
    ):
        return speech_of(response) or "the agent failed to handle the request"
    return None


class HomeLearningConversationEntity(conversation.ConversationEntity):
    """Watch every turn, hand it on, and learn from what came back."""

    _attr_has_entity_name = True
    _attr_name = None
    _attr_supports_streaming = True
    _attr_supported_features = conversation.ConversationEntityFeature.CONTROL

    def __init__(self, entry: ConfigEntry, subentry: ConfigSubentry) -> None:
        # Everything this agent does reads its own settings through the view, and shares
        # the hub's store, files and counters.
        self.entry = AgentView(entry, subentry)
        self._hub = entry
        self._attr_unique_id = subentry.subentry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=subentry.title,
            entry_type=dr.DeviceEntryType.SERVICE,
        )
        # One turn is judged at a time, so two conversations cannot race on the
        # sentence files or on the reload that follows.
        self._lock = asyncio.Lock()

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        return MATCH_ALL

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        # The legacy per-entry agent slot holds one agent; pipelines pick by entity id, so
        # the first agent takes it and no other agent's removal can empty it.
        if self._is_first_agent():
            conversation.async_set_agent(self.hass, self._hub, self)

    async def async_will_remove_from_hass(self) -> None:
        if self._is_first_agent():
            conversation.async_unset_agent(self.hass, self._hub)
        await super().async_will_remove_from_hass()

    def _is_first_agent(self) -> bool:
        agents = agent_subentries(self._hub)
        return bool(agents) and agents[0].subentry_id == self.entry.subentry.subentry_id

    # -- the turn ----------------------------------------------------------

    async def _ask(
        self, user_input: conversation.ConversationInput, agent_id: str
    ) -> tuple[conversation.ConversationResult | None, str | None]:
        """Ask one agent, and report a failure as a value instead of an exception."""
        try:
            result = await conversation.async_converse(
                self.hass,
                user_input.text,
                user_input.conversation_id,
                user_input.context,
                language=user_input.language,
                agent_id=agent_id,
                device_id=user_input.device_id,
                satellite_id=user_input.satellite_id,
                extra_system_prompt=user_input.extra_system_prompt,
            )
        except Exception as err:  # noqa: BLE001 - a broken agent must not break voice
            return None, f"{type(err).__name__}: {err}"
        return result, None

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Pass the sentence on, return the answer, and read the turn."""
        delegate = delegate_agent(self.hass, self.entry, self.entity_id)
        if delegate is None:
            return _error_result(
                user_input, "There is no assistant configured to answer this."
            )

        result, failure = await self._ask(user_input, delegate)
        if failure is None:
            failure = _did_not_answer(result)
        answered = failure is None

        if failure is not None:
            # Something below this agent is broken - a model out of quota, a tool API
            # that has not finished registering after a restart, no network. Whatever it
            # is, the person is waiting, and what they must not get is the exception read
            # out loud. Home Assistant's own agent always exists and answers the commands
            # it knows, so the turn is handed to it rather than dropped.
            backup = FALLBACK_AGENT if FALLBACK_AGENT != delegate else None
            if backup and conversation.async_get_agent(self.hass, backup) is not None:
                _LOGGER.warning(
                    "Irabelle Home Learning: %s did not answer %r (%s); asking %s instead",
                    delegate,
                    user_input.text,
                    failure,
                    backup,
                )
                backup_result, backup_failure = await self._ask(user_input, backup)
                if backup_failure is None:
                    backup_failure = _did_not_answer(backup_result)
                if backup_failure is None:
                    result, failure = backup_result, None
                else:
                    result, failure = None, backup_failure

        if failure is not None or result is None:
            # Nothing answered. Say one short sentence, keep the detail in the log, and
            # learn nothing: a turn that was not answered is not a phrasing to bind.
            _LOGGER.error(
                "Irabelle Home Learning: nothing could answer %r: %s", user_input.text, failure
            )
            return _error_result(user_input, NO_ANSWER)

        # From here the answer exists and is going back to the pipeline
        # unchanged; everything below is bookkeeping that must never fail it.
        try:
            ensure_answer_in_log(chat_log, self.entity_id, speech_of(result.response))
            if answered and auto_learn(self.entry):
                if (turn := turn_from_result(user_input, chat_log, result, delegate)) :
                    self._consider(turn)
        except Exception:  # noqa: BLE001 - never break an answer for a lesson
            _LOGGER.exception("Irabelle Home Learning could not read the turn it just handled")

        return result

    @callback
    def _consider(self, turn: Turn) -> None:
        """Judge the turn off the hot path, once the answer is on its way."""
        data: LearnerData | None = getattr(self.entry, "runtime_data", None)
        if data is not None:
            data.turns_seen += 1

        async def _run() -> None:
            # Let the pipeline finish the turn before touching sentence files:
            # learning reloads `conversation`, and doing that mid-answer would be
            # a self-inflicted race.
            await asyncio.sleep(0.25)
            async with self._lock:
                try:
                    result = await async_consider_turn(self.hass, self.entry, turn)
                except Exception:  # noqa: BLE001 - learning must never break voice
                    # CRITICAL as well as the traceback: a logger configured `default:
                    # critical` hides ERROR, and a turn dropped in silence looks exactly
                    # like a learner with nothing to say. One NameError in this path went
                    # unnoticed for hours that way.
                    _LOGGER.critical(
                        "Irabelle Home Learning dropped a turn it could not consider: %r",
                        turn.utterance,
                        exc_info=True,
                    )
                    return
            if result:
                _LOGGER.debug("Irabelle Home Learning acted on %r: %s", turn.utterance, result)

        self.hass.async_create_task(
            _run(), name=f"irabelle_home_learning_consider_{turn.conversation_id}"
        )
