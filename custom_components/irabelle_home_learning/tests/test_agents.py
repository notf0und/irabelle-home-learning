"""One hub, several agents: whose settings a learning uses, and how a model is asked.

An agent's own settings must win on a turn it handled - "Jev -> DeepSeek" must not be
classified by ChatGPT because another agent was set up first - while the store and the
counters stay the hub's. Anything not started by a turn reads the first agent.
"""

from __future__ import annotations

import unittest
from types import MappingProxyType

from custom_components.irabelle_home_learning.agents import (
    SUBENTRY_AGENT,
    AgentView,
    settings,
)
from custom_components.irabelle_home_learning.judge import judge_of
from custom_components.irabelle_home_learning.models import _describe_field, _json_object


class Subentry:
    def __init__(self, subentry_id: str, title: str, **data):
        self.subentry_id = subentry_id
        self.title = title
        self.subentry_type = SUBENTRY_AGENT
        self.data = MappingProxyType(data)


class Hub:
    def __init__(self, *agents: Subentry, **options):
        self.entry_id = "hub"
        self.data = {}
        self.options = options
        self.runtime_data = object()
        self.subentries = {agent.subentry_id: agent for agent in agents}


JEV = Subentry("a", "Jev -> DeepSeek", judge="jev_integration", ai_task_entity="ai_task.deepseek")
GPT = Subentry("b", "ChatGPT -> DeepSeek", judge="conversation.chatgpt", ai_task_entity="ai_task.deepseek")


class Settings(unittest.TestCase):
    def test_a_turn_uses_its_own_agents_models(self):
        hub = Hub(JEV, GPT, language="en")
        self.assertEqual(judge_of(AgentView(hub, GPT)), "conversation.chatgpt")
        self.assertEqual(judge_of(AgentView(hub, JEV)), "jev_integration")

    def test_the_view_shares_the_hubs_store_and_counters(self):
        hub = Hub(JEV, GPT)
        view = AgentView(hub, GPT)
        self.assertIs(view.runtime_data, hub.runtime_data)
        self.assertEqual(view.entry_id, "hub")
        self.assertEqual(view.title, "ChatGPT -> DeepSeek")

    def test_hub_settings_reach_every_agent(self):
        hub = Hub(JEV, language="es", min_occurrences=3)
        self.assertEqual(AgentView(hub, JEV).options["language"], "es")

    def test_without_a_turn_the_first_agent_answers(self):
        self.assertEqual(judge_of(Hub(JEV, GPT)), "jev_integration")

    def test_an_entry_with_no_agents_reads_its_own_options(self):
        self.assertEqual(settings(Hub(judge="none"))["judge"], "none")


class AskingAConversationAgent(unittest.TestCase):
    def test_json_is_found_wherever_the_model_put_it(self):
        self.assertEqual(_json_object('Sure! {"a": 1}'), {"a": 1})
        self.assertEqual(_json_object('```json\n{"a": 2}\n```\nDone.'), {"a": 2})
        self.assertIsNone(_json_object("I cannot help with that."))

    def test_each_field_says_what_shape_it_takes(self):
        select = _describe_field(
            "action",
            {"required": True, "description": "what to do", "selector": {"select": {"options": ["bind", "ignore"]}}},
        )
        self.assertIn('"action" (required): one of: "bind", "ignore"', select)
        number = _describe_field("p", {"selector": {"number": {"min": 0, "max": 1}}})
        self.assertIn("a number from 0 to 1", number)


if __name__ == "__main__":
    unittest.main()
