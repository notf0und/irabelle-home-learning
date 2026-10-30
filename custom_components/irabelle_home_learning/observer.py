"""Read a finished turn out of the pipeline, without patching anybody's agent.

`irabelle_home_learning` sits in front of the real conversation agent:

    pipeline -> irabelle_home_learning -> your agent -> its fallback

Two facts make that position enough to learn from, and neither of them needs a
third-party component changed:

* Home Assistant asks the local sentence matcher before it calls any agent, so a
  turn that arrives here is one the matcher missed. That is the trigger, and it
  is a fact rather than a judgement.
* The whole chain shares one ``ChatLog``. ``async_get_chat_log`` hands the active
  log back when the conversation id matches, so the agents further down write
  into the very log this agent was given. Whatever they did is already here -
  the requests, the tool calls with their arguments, the answer.

What a finished turn is read for:

* ``response.intent`` - the ``Intent`` Home Assistant actually ran. This is the
  reliable one: Jev routes by calling ``intent.async_handle``, and the handler's
  response carries the intent type and the slots it was called with, so a
  Jev-routed turn can be bound exactly.
* the chat log - every tool call, with arguments. This is the DeepSeek path: its
  final response is a fresh ``IntentResponse``, so the evidence is in the calls
  it made rather than in the response object.
* the spoken answer and the entities it acted on - what a generalisation needs
  when neither of the above identifies anything.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import logging
from dataclasses import dataclass, field
from typing import Any

from homeassistant.components import conversation
from homeassistant.core import callback

from .const import MAX_UTTERANCE_CHARS

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class Turn:
    """One complete request/answer turn, as seen from the front of the chain."""

    utterance: str
    conversation_id: str = ""
    agent_id: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    answer: str = ""
    #: The intent Home Assistant ran, when the turn ended by running one.
    intent: dict[str, Any] | None = None
    #: The things the answer acted on, as the response reported them.
    targets: list[str] = field(default_factory=list)
    observed_at: str = field(
        default_factory=lambda: dt.datetime.now().isoformat(timespec="seconds")
    )

    @property
    def tool_names(self) -> list[str]:
        return [call["name"] for call in self.tool_calls]

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def speech_of(response: Any) -> str:
    """Return the plain speech of an intent response, if it has any."""
    try:
        return (response.speech.get("plain") or {}).get("speech") or ""
    except (AttributeError, TypeError):
        return ""


def _last_user_index(content: list[Any]) -> int | None:
    for index in range(len(content) - 1, -1, -1):
        if getattr(content[index], "role", None) == "user":
            return index
    return None


def turn_from_result(
    user_input: conversation.ConversationInput,
    chat_log: conversation.ChatLog,
    result: conversation.ConversationResult,
    delegate: str = "",
) -> Turn | None:
    """Describe the turn that just finished, or None if there is nothing to see.

    ``delegate`` names the agent the sentence was handed to, which is what the
    provenance should say for a turn that agent routed itself without leaving
    anything in the chat log.
    """
    utterance = " ".join((user_input.text or "").split())
    if not utterance or len(utterance) > MAX_UTTERANCE_CHARS:
        _LOGGER.debug("Irabelle Home Learning ignoring an unusable utterance: %r", user_input.text)
        return None

    response = getattr(result, "response", None)
    turn = Turn(
        utterance=utterance,
        conversation_id=user_input.conversation_id or "",
        agent_id=delegate or user_input.agent_id or "",
        answer=speech_of(response),
    )

    # The intent that ran, straight off the response. Jev answers this way.
    intent_obj = getattr(response, "intent", None)
    if intent_obj is not None and getattr(intent_obj, "intent_type", None):
        turn.intent = {
            "name": str(intent_obj.intent_type),
            "slots": dict(getattr(intent_obj, "slots", None) or {}),
        }

    content = getattr(chat_log, "content", None) or []
    start = _last_user_index(content)
    for item in content[(start + 1) if start is not None else 0 :]:
        role = getattr(item, "role", None)
        if role not in ("assistant", "tool_result"):
            continue
        agent = getattr(item, "agent_id", "") or ""
        # The model below us signs its work; keep the innermost name as the source.
        if agent and agent != user_input.agent_id:
            turn.agent_id = agent
        if role == "assistant":
            for call in getattr(item, "tool_calls", None) or []:
                name = getattr(call, "tool_name", "") or ""
                if name and name not in turn.tool_names:
                    turn.tool_calls.append(
                        {
                            "name": name,
                            "args": dict(getattr(call, "tool_args", None) or {}),
                        }
                    )
            text = (getattr(item, "content", None) or "").strip()
            if text:
                turn.answer = text
        else:
            name = getattr(item, "tool_name", "") or ""
            # A streamed answer adds tool results without the message that asked
            # for them, so record the call here too - without arguments.
            if name and name not in turn.tool_names:
                turn.tool_calls.append({"name": name, "args": {}})

    for target in getattr(response, "success_results", None) or []:
        label = getattr(target, "name", None) or getattr(target, "id", None)
        if label and str(label) not in turn.targets:
            turn.targets.append(str(label))

    return turn


@callback
def ensure_answer_in_log(
    chat_log: conversation.ChatLog, agent_id: str, speech: str
) -> None:
    """Put the answer in the chat log if the agent below us did not.

    Home Assistant does this itself for turns it answers locally. An agent that
    routes without ever touching the log (Jev) leaves it ending on the user
    message, which loses the answer from the history and breaks a follow-up
    question that depends on it.
    """
    if not speech:
        return
    content = getattr(chat_log, "content", None) or []
    if content and getattr(content[-1], "role", None) == "assistant":
        return
    chat_log.async_add_assistant_content_without_tools(
        conversation.AssistantContent(agent_id=agent_id, content=speech)
    )
