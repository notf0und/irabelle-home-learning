"""Config flow for Irabelle Home Learning: one hub of learning, any number of agents.

The **hub** (the config entry) is the house's: the language learned sentences are written
in, how eager the learner is, and where `export` writes. There is one, because two would
write the same sentence files.

An **agent** (a subentry, *Add agent* on the integration page) is what a voice pipeline
picks as its conversation agent. It has a name and three choices:

* the agent the sentence is handed to (the one that really answers);
* the **classification model** - who answers the quick typed judgements (is this worth
  learning, which intent does it mean, is this answer right): Jev, or any model;
* the **processing model** - who writes and reviews the templates behind new answers.

"Jev -> DeepSeek" and "ChatGPT -> DeepSeek" are two agents; what either learns, both answer.
Models are listed by the name a person knows them by, never by entity id.
"""

from __future__ import annotations

from typing import Any

import voluptuous as vol
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentryFlow,
    OptionsFlow,
    SubentryFlowResult,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.selector import (
    BooleanSelector,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)

from .agents import SUBENTRY_AGENT
from .const import (
    CONF_AI_TASK_ENTITY,
    CONF_AUTO_LEARN,
    CONF_DELEGATE_AGENT,
    CONF_EXPORT_PATH,
    CONF_JEV_API_KEY,
    CONF_JEV_ENDPOINT,
    CONF_JEV_MODEL,
    CONF_JUDGE,
    CONF_LANGUAGE,
    CONF_MIN_OCCURRENCES,
    DEFAULT_AUTO_LEARN,
    DEFAULT_DELEGATE_AGENT,
    DEFAULT_EXPORT_PATH,
    DEFAULT_JEV_ENDPOINT,
    DEFAULT_JEV_MODEL,
    DEFAULT_LANGUAGE,
    DEFAULT_MIN_OCCURRENCES,
    DOMAIN,
    JUDGE_JEV,
    JUDGE_JEV_INTEGRATION,
    JUDGE_NONE,
    NAME,
)
from .models import agent_choices, model_choices

CONF_NAME = "name"
DEFAULT_AGENT_NAME = "Home Learning"


def _jev_installed(hass: HomeAssistant) -> bool:
    return bool(hass.config_entries.async_entries("jev"))


def _classifier_choices(hass: HomeAssistant, current: str | None) -> list[SelectOptionDict]:
    """Jev first when it is installed, then every model, then none."""
    choices: list[SelectOptionDict] = []
    if _jev_installed(hass) or current == JUDGE_JEV_INTEGRATION:
        choices.append(SelectOptionDict(label="Jev", value=JUDGE_JEV_INTEGRATION))
    choices.extend(model_choices(hass, current))
    choices.append(
        SelectOptionDict(label="Jev API (direct, with its own key)", value=JUDGE_JEV)
    )
    choices.append(SelectOptionDict(label="None (learn by binding only)", value=JUDGE_NONE))
    return choices


def _default_classifier(hass: HomeAssistant) -> str | None:
    choices = _classifier_choices(hass, None)
    return choices[0]["value"] if len(choices) > 2 else None


def _default_processor(hass: HomeAssistant) -> str | None:
    choices = model_choices(hass)
    return choices[0]["value"] if choices else None


def _select(options: list[SelectOptionDict]) -> SelectSelector:
    return SelectSelector(
        SelectSelectorConfig(
            options=options, mode=SelectSelectorMode.DROPDOWN, custom_value=False
        )
    )


def _agent_schema(hass: HomeAssistant, values: dict[str, Any]) -> vol.Schema:
    """Name, who answers, and the two models - each a dropdown of names."""
    judge = values.get(CONF_JUDGE) or _default_classifier(hass)
    processor = values.get(CONF_AI_TASK_ENTITY) or _default_processor(hass)
    return vol.Schema(
        {
            vol.Required(CONF_NAME, default=values.get(CONF_NAME) or DEFAULT_AGENT_NAME): TextSelector(),
            vol.Required(
                CONF_DELEGATE_AGENT,
                default=values.get(CONF_DELEGATE_AGENT, DEFAULT_DELEGATE_AGENT),
            ): _select(agent_choices(hass)),
            vol.Required(CONF_JUDGE, default=judge or vol.UNDEFINED): _select(
                _classifier_choices(hass, judge)
            ),
            vol.Required(CONF_AI_TASK_ENTITY, default=processor or vol.UNDEFINED): _select(
                model_choices(hass, processor)
            ),
        }
    )


def _agent_errors(hass: HomeAssistant, user_input: dict[str, Any]) -> dict[str, str]:
    if user_input[CONF_JUDGE] == JUDGE_JEV_INTEGRATION and not _jev_installed(hass):
        return {CONF_JUDGE: "jev_not_installed"}
    return {}


def _jev_schema(values: dict[str, Any]) -> vol.Schema:
    return vol.Schema(
        {
            vol.Optional(
                CONF_JEV_ENDPOINT, default=values.get(CONF_JEV_ENDPOINT, DEFAULT_JEV_ENDPOINT)
            ): TextSelector(),
            vol.Optional(
                CONF_JEV_MODEL, default=values.get(CONF_JEV_MODEL, DEFAULT_JEV_MODEL)
            ): TextSelector(),
            vol.Optional(
                CONF_JEV_API_KEY,
                description={"suggested_value": values.get(CONF_JEV_API_KEY) or None},
            ): TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD)),
        }
    )


def _agent_data(values: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """The subentry's title and data from the form values."""
    data = dict(values)
    title = str(data.pop(CONF_NAME, "") or "").strip() or DEFAULT_AGENT_NAME
    return title, data


def _hub_schema(values: dict[str, Any], *, first: bool) -> vol.Schema:
    fields: dict[Any, Any] = {
        vol.Required(
            CONF_LANGUAGE, default=values.get(CONF_LANGUAGE, DEFAULT_LANGUAGE)
        ): TextSelector(),
    }
    if not first:
        fields[
            vol.Required(
                CONF_AUTO_LEARN, default=values.get(CONF_AUTO_LEARN, DEFAULT_AUTO_LEARN)
            )
        ] = BooleanSelector()
        fields[
            vol.Required(
                CONF_MIN_OCCURRENCES,
                default=values.get(CONF_MIN_OCCURRENCES, DEFAULT_MIN_OCCURRENCES),
            )
        ] = NumberSelector(
            NumberSelectorConfig(min=1, max=10, step=1, mode=NumberSelectorMode.BOX)
        )
    # Optional and empty unless a person typed a path: empty means the default.
    fields[
        vol.Optional(
            CONF_EXPORT_PATH,
            description={"suggested_value": values.get(CONF_EXPORT_PATH) or None},
        )
    ] = TextSelector()
    return vol.Schema(fields)


def _clean_hub(user_input: dict[str, Any]) -> dict[str, Any]:
    data = dict(user_input)
    if not str(data.get(CONF_EXPORT_PATH) or "").strip():
        data.pop(CONF_EXPORT_PATH, None)
    return data


class HomeLearningConfigFlow(ConfigFlow, domain=DOMAIN):
    """Set up the hub and its first agent. More agents are added from the integration page."""

    VERSION = 2

    def __init__(self) -> None:
        self._hub: dict[str, Any] = {}
        self._agent: dict[str, Any] = {}

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        await self.async_set_unique_id(DOMAIN)
        self._abort_if_unique_id_configured()
        if user_input is not None:
            self._hub = _clean_hub(user_input)
            return await self.async_step_agent()
        return self.async_show_form(
            step_id="user",
            data_schema=_hub_schema({}, first=True),
            description_placeholders={"default_path": DEFAULT_EXPORT_PATH},
        )

    async def async_step_agent(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        errors: dict[str, str] = {}
        if user_input is not None:
            errors = _agent_errors(self.hass, user_input)
            if not errors:
                self._agent.update(user_input)
                if user_input[CONF_JUDGE] == JUDGE_JEV:
                    return await self.async_step_jev()
                return self._create()
        return self.async_show_form(
            step_id="agent",
            data_schema=_agent_schema(self.hass, user_input or self._agent),
            errors=errors,
        )

    async def async_step_jev(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        if user_input is not None:
            self._agent.update(user_input)
            return self._create()
        return self.async_show_form(step_id="jev", data_schema=_jev_schema(self._agent))

    def _create(self) -> ConfigFlowResult:
        title, data = _agent_data(self._agent)
        return self.async_create_entry(
            title=NAME,
            data={},
            options=self._hub,
            subentries=[
                {"subentry_type": SUBENTRY_AGENT, "title": title, "data": data, "unique_id": None}
            ],
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> HomeLearningOptionsFlow:
        return HomeLearningOptionsFlow()

    @classmethod
    @callback
    def async_get_supported_subentry_types(
        cls, config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        return {SUBENTRY_AGENT: AgentSubentryFlow}


class HomeLearningOptionsFlow(OptionsFlow):
    """What belongs to the house rather than to one agent."""

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        if user_input is not None:
            return self.async_create_entry(data=_clean_hub(user_input))
        values = {
            CONF_LANGUAGE: self.config_entry.data.get(CONF_LANGUAGE, DEFAULT_LANGUAGE),
            **self.config_entry.options,
        }
        return self.async_show_form(
            step_id="init",
            data_schema=_hub_schema(values, first=False),
            description_placeholders={"default_path": DEFAULT_EXPORT_PATH},
        )


class AgentSubentryFlow(ConfigSubentryFlow):
    """Add or change one agent: a name, who answers, and the two models."""

    def __init__(self) -> None:
        super().__init__()
        self._values: dict[str, Any] = {}

    @property
    def _is_new(self) -> bool:
        return self.source == "user"

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        return await self.async_step_agent(user_input)

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        subentry = self._get_reconfigure_subentry()
        self._values = {CONF_NAME: subentry.title, **subentry.data}
        return await self.async_step_agent(user_input)

    async def async_step_agent(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        errors: dict[str, str] = {}
        if user_input is not None:
            errors = _agent_errors(self.hass, user_input)
            if not errors:
                self._values.update(user_input)
                if user_input[CONF_JUDGE] == JUDGE_JEV:
                    return await self.async_step_jev()
                return self._finish()
        return self.async_show_form(
            step_id="agent",
            data_schema=_agent_schema(self.hass, user_input or self._values),
            errors=errors,
        )

    async def async_step_jev(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Where the Jev API lives, when it is called directly."""
        if user_input is not None:
            self._values.pop(CONF_JEV_API_KEY, None)
            self._values.update(user_input)
            return self._finish()
        return self.async_show_form(step_id="jev", data_schema=_jev_schema(self._values))

    def _finish(self) -> SubentryFlowResult:
        title, data = _agent_data(self._values)
        if self._is_new:
            return self.async_create_entry(title=title, data=data)
        return self.async_update_and_abort(
            self._get_entry(), self._get_reconfigure_subentry(), title=title, data=data
        )
