"""Constants for the Irabelle Home Learning integration."""

from __future__ import annotations

from typing import Final

DOMAIN: Final = "irabelle_home_learning"
NAME: Final = "Irabelle Home Learning"
VERSION: Final = "1.0.0"

# Where learned sentences are written. `custom_sentences/<language>/` is read by
# Home Assistant's conversation integration and refreshed by `conversation.reload`.
SENTENCES_DIR: Final = "custom_sentences"

#: Files the integration owns. Anything else in custom_sentences/ is left alone.
FILE_PREFIX: Final = "learned-"

#: Files that fail to parse are moved here rather than deleted.
QUARANTINE_DIR: Final = "irabelle_home_learning/quarantine"

CONF_LANGUAGE: Final = "language"
DEFAULT_LANGUAGE: Final = "en"

#: Which integration domains may teach this house *with the teaching tools*.
#: Empty means "any". Restricting it keeps a chatty model on an experimental
#: agent from writing sentences. This does not gate the automatic learner: the
#: learner only sees a turn because this integration was put in the pipeline,
#: which is an explicit act by whoever built that pipeline.
CONF_ALLOWED_PLATFORMS: Final = "allowed_platforms"
#: Empty on purpose: a fresh install has no way to know which platform hosts the
#: user's assistant, and listing someone else's would be a default that does not
#: work. "Any" is the honest default, and the options flow is where it narrows.
DEFAULT_ALLOWED_PLATFORMS: Final[list[str]] = []

#: The agent this one hands the sentence to:
#:
#:     pipeline -> irabelle_home_learning -> delegate -> (its own fallback)
#:
#: Default: Home Assistant's own agent, because it is the only one guaranteed to
#: exist. Point it at whatever actually answers your house in the options.
CONF_DELEGATE_AGENT: Final = "delegate_agent"
DEFAULT_DELEGATE_AGENT: Final = "conversation.home_assistant"

#: Used when the configured agent is missing, so a pipeline containing this
#: integration degrades to Home Assistant's own agent rather than to silence.
FALLBACK_AGENT: Final = "conversation.home_assistant"

#: Watch conversations and learn unattended. Without it, learning only happens
#: when a model chooses to call a teaching tool - which is the thing that turned
#: out not to work.
CONF_AUTO_LEARN: Final = "auto_learn"
DEFAULT_AUTO_LEARN: Final = True

#: How many times a request must be heard before it is worth generalising into a
#: local intent. A phrasing said twice is one that will be said again; a one-off
#: is not worth a model call or a sentence file.
CONF_MIN_OCCURRENCES: Final = "min_occurrences"
DEFAULT_MIN_OCCURRENCES: Final = 2

#: The *processing model*: the `ai_task.*` entity that writes and reviews templates.
#: Empty means "pick one from an allowed platform".
CONF_AI_TASK_ENTITY: Final = "ai_task_entity"

#: The *classification model*: who answers the learner's snap judgements (is this worth
#: learning, which intent does it mean, is this answer right). Stored as one of the modes
#: below or as an ``ai_task.*`` entity id. See `judge.py`.
CONF_JUDGE: Final = "judge"
#: The processing model (``ai_task_entity``) classifies as well. Works with no Jev at all.
JUDGE_AI_TASK: Final = "ai_task"
#: The Jev integration's ``jev.ask`` action, with its key and budget.
JUDGE_JEV_INTEGRATION: Final = "jev_integration"
#: The System One API called directly, with the settings below.
JUDGE_JEV: Final = "jev"
#: Nothing is asked; every judgement-backed step is skipped.
JUDGE_NONE: Final = "none"
DEFAULT_JUDGE: Final = JUDGE_AI_TASK

#: Jev connection. The key may be left empty, in which case the key already configured
#: for the Jev integration is used — one secret to rotate rather than two.
CONF_JEV_ENDPOINT: Final = "jev_endpoint"
DEFAULT_JEV_ENDPOINT: Final = "https://api.typesafe.ai/v1/systemone"
CONF_JEV_MODEL: Final = "jev_model"
DEFAULT_JEV_MODEL: Final = "jev-latest"
CONF_JEV_API_KEY: Final = "jev_api_key"

#: Settings that belong to one agent (a config subentry) rather than to the hub.
AGENT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "delegate_agent",
        "judge",
        "ai_task_entity",
        "jev_endpoint",
        "jev_model",
        "jev_api_key",
    }
)

#: The file the `export` service writes its `intent_script` stubs to, when the house has not
#: declared where they belong (see `registration.py` - an existing `intent_script:` in
#: `configuration.yaml` decides, and this is only the fallback and the file name used inside a
#: directory the house named). Outside the component on purpose: anything inside its folder
#: goes when it does.
CONF_EXPORT_PATH: Final = "export_path"
DEFAULT_EXPORT_PATH: Final = "configurations/intent_script/irabelle_home_learning.yaml"

#: A runaway guard. The docs rate limit at 1,200 requests/min, which is far more rope
#: than an unsupervised learner should ever need.
MAX_JUDGE_CALLS_PER_HOUR: Final = 120

#: Observations are a rolling window, not a corpus.
MAX_OBSERVATIONS: Final = 300

#: Variables the learned-intent handler always defines, so a template may branch on
#: who is asking. Kept here because both the handler and the checks need the list.
#: What a million input tokens costs at the judge, for the spend estimate. Output is
#: free at the time of writing; change this if that changes, or the number is a lie.
JEV_PRICE_PER_MTOK: Final = 0.042

#: The voice-biometrics sensor a satellite has no user for, and the confidence below
#: which a name from it is a guess rather than an answer.
SPEAKER_SENSOR: Final = "sensor.voicebm_current_speaker"
SPEAKER_FLOOR: Final = 0.5

SPEAKER_VARIABLES: Final[frozenset[str]] = frozenset(
    {
        "user_id",
        "user_name",
        "person_entity_id",
        "person_name",
        "person_state",
        "device_id",
        "satellite_id",
        "user_area",
        "user_floor",
        # Whether the name above is trustworthy, and how sure the voice match was. A
        # template that answers per speaker should check this: a wrong guess answers with
        # the wrong person's phone, and it sounds just as confident as a right one.
        "speaker_certain",
        "speaker_confidence",
    }
)

#: A learned intent template is rendered on every request; keep it readable.
MAX_TEMPLATE_CHARS: Final = 4000
INTENT_NAME_RE: Final = r"^[A-Za-z][A-Za-z0-9_]{2,60}$"

STORAGE_VERSION: Final = 1
STORAGE_KEY: Final = f"{DOMAIN}.learned"

SIGNAL_UPDATED: Final = f"{DOMAIN}_updated"

#: Refuse absurd proposals: a phrase list is not a corpus.
MAX_UTTERANCES: Final = 40
MAX_UTTERANCE_CHARS: Final = 200

#: How many learned rows the sensor exposes.
SENSOR_ATTRIBUTE_LIMIT: Final = 25

#: Intents Home Assistant ships that a learned sentence may target, used as a
#: fallback when the intent registry does not enumerate them.
BUILTIN_INTENTS: Final[frozenset[str]] = frozenset(
    {
        "HassTurnOn",
        "HassTurnOff",
        "HassToggle",
        "HassGetState",
        "HassLightSet",
        "HassSetPosition",
        "HassStartTimer",
        "HassCancelTimer",
        "HassCancelAllTimers",
        "HassPauseTimer",
        "HassUnpauseTimer",
        "HassIncreaseTimer",
        "HassDecreaseTimer",
        "HassTimerStatus",
        "HassStopMoving",
        "HassMediaSearchAndPlay",
        "HassMediaPause",
        "HassMediaNext",
        "HassMediaPrevious",
        "HassMediaUnpause",
    }
)
