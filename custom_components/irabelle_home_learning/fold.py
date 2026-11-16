"""Folding a group's phrasings into templates: the model proposes, the matcher and the judge prove.

A group of learned phrasings for one target piles up line by line - every way it was asked,
every way speech-to-text misheard it::

    - Press the D2 keyboard
    - Toggle the D2 keyboard
    - Togl the D2 keyboard
    - Together the keyboard

Home Assistant's sentence syntax says all of that in one line, and covers the ways nobody has
said yet: ``(press|toggle|togl|together) [the] [D2] keyboard``. Writing that line is a language
task, so the **processing model** proposes it. Nothing it proposes is trusted:

1. **Coverage** - hassil, the matcher the house runs, must answer every original phrasing with
   the folded lines, with the group's pinned slots coming back out of the match.
2. **No theft** - the folded lines must not match any other group's sentence, in this file or
   any other custom sentence file, and no other group may already answer what they add.
3. **No over-reach** - a sample of the sentences the fold *adds* (ones nobody said) goes to the
   **classification model**, which must agree each one is the same request on the same target,
   speech-to-text mistakes allowed.

Anything that fails a check leaves the group exactly as it was. Applied folds are re-checked
against the live matcher after the reload and put back if it does not accept them.

Folding works on a group in place - its ``slots:`` and its ``response:`` stay with it - so a
binding to a built-in intent folds as safely as an intent the learner invented.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hassil import Intents, parse_sentence, recognize
from hassil.sample import sample_expression
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.util import yaml as yaml_util

from .const import FILE_PREFIX
from .intents import concrete
from .fileio import async_run
from .judge import JudgeError, Noul, async_ask, has_judge
from .learner import (
    LearnerError,
    _load,
    async_save,
    async_verify,
    language_of,
    normalize_sentence,
    sentence_for_storage,
    sentences_root,
)
from .models import async_generate_data

_LOGGER = logging.getLogger(__name__)

#: A group is worth folding once it holds this many plain phrasings, or one new phrasing
#: beside lines that are already folded.
FOLD_MIN_LINES = 3

#: The most templates a fold may produce for one group. More than this is not a fold.
MAX_TEMPLATES = 3

#: Bounds on what is enumerated: a template's expansions, the novel ones the judge sees.
MAX_EXPANSIONS = 400
MAX_JUDGED = 8

#: How sure the judge must be that each added sentence is the same request. Every sample must
#: clear it: one sentence that means something else is one sentence that does the wrong thing.
FOLD_SAMPLE_CLEAN = 0.75

_SYNTAX = ("(", ")", "[", "]", "|", "<", ">")

#: Content last attempted per group, so a group the model could not fold is not asked about
#: again until it changes. In memory: a restart retries once, which is fine.
_ATTEMPTED: dict[str, str] = {}

FOLD_INSTRUCTIONS = """You write Home Assistant custom sentence templates (hassil syntax).

These phrasings all trigger the same action on the same target. Some are speech-to-text \\
mistakes of the same words - keep those: they are how this house's microphone hears people.

PHRASINGS
{lines}

TARGET
{target}

Write at most {most} templates that together match EVERY phrasing above. Syntax:
- (a|b|c) one of several words or phrases
- [x] optional words; [a|b] optional choice
- keep any {{slot}} reference exactly as written; do not invent new ones
- no other syntax: no <rules>, no regular expressions, no punctuation

Rules:
- Fold the misheard variants of a word into one alternative: (toggle|togl|togle|toget)
- Never make the whole target optional unless a phrasing above has no target words at all
- Do not add words or synonyms nobody said; generalise only by combining what is above
- Prefer one template; use more only when the phrasings do not share a shape
"""

FOLD_STRUCTURE = {
    "templates": {
        "description": "the templates, one per line",
        "required": True,
        "selector": {"text": {"multiline": True}},
    },
    "reason": {"description": "one line on what was folded", "selector": {"text": {}}},
}


@dataclass
class Group:
    """One `data:` block of one intent in one learner-owned file."""

    path: Path
    intent: str
    index: int
    sentences: list[str]
    pin: dict[str, Any]
    lists: dict[str, Any]

    @property
    def key(self) -> str:
        return f"{self.path.name}:{self.intent}:{json.dumps(self.pin, sort_keys=True, default=str)}"

    @property
    def plain(self) -> list[str]:
        return [s for s in self.sentences if not any(c in s for c in _SYNTAX)]

    @property
    def templates(self) -> list[str]:
        return [s for s in self.sentences if any(c in s for c in _SYNTAX)]

    def ready(self) -> bool:
        plain = len(self.plain)
        return plain >= FOLD_MIN_LINES or (plain >= 1 and bool(self.templates))

    def fingerprint(self) -> str:
        return hashlib.sha1("\n".join(self.sentences).encode()).hexdigest()


@dataclass
class Outcome:
    group: Group
    folded: list[str] = field(default_factory=list)
    why: str = ""
    samples: dict[str, float] = field(default_factory=dict)
    applied: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "intent": self.group.intent,
            "target": self.group.pin or None,
            "from": list(self.group.sentences),
            "into": self.folded or None,
            "applied": self.applied,
            "why": self.why,
            "judged": self.samples or None,
        }


# --------------------------------------------------------------------------
# Reading the groups
# --------------------------------------------------------------------------


def groups(hass: HomeAssistant, language: str, intent: str | None = None) -> list[Group]:
    """Every group in the learner's own files, optionally for one intent."""
    found: list[Group] = []
    root = sentences_root(hass, language)
    if not root.is_dir():
        return found
    for path in sorted(root.glob(f"{FILE_PREFIX}*.yaml")):
        try:
            content = _load(path) or {}
        except LearnerError:
            continue
        lists = dict(content.get("lists") or {})
        for name, body in (content.get("intents") or {}).items():
            if intent is not None and name != intent:
                continue
            for index, item in enumerate((body or {}).get("data") or []):
                sentences = [
                    str(s) for s in (item or {}).get("sentences") or [] if str(s).strip()
                ]
                if sentences:
                    found.append(
                        Group(path, name, index, sentences, dict((item or {}).get("slots") or {}), lists)
                    )
    return found


# --------------------------------------------------------------------------
# The proofs
# --------------------------------------------------------------------------


def expansions(sentence: str, lists: dict[str, Any], limit: int = MAX_EXPANSIONS) -> list[str]:
    """Every sentence a template accepts (bounded), with a sample value in each slot."""
    text = concrete(sentence, lists)
    try:
        # `parse_sentence` wraps the expression in a Sentence, which the sampler does not take.
        expression = parse_sentence(text).expression
        found = itertools.islice(sample_expression(expression), limit)
        return sorted({" ".join(item.split()) for item in found})
    except Exception:  # noqa: BLE001 - an unparsable line expands to nothing
        return []


def added_sentences(group: Group, folded: list[str]) -> list[str]:
    """What the folded lines accept that the group did not - by the matcher's own measure.

    Compared case- and punctuation-free, the way the matcher compares: "Toggle D2 keyboard"
    was said, so "toggle D2 keyboard" is not new and is not a question for the judge.
    """
    said = {normalize_sentence(e) for s in group.sentences for e in expansions(s, group.lists)}
    found: dict[str, str] = {}
    for sentence in folded:
        for example in expansions(sentence, group.lists):
            key = normalize_sentence(example)
            if key not in said:
                found.setdefault(key, example)
    return sorted(found.values())


def _block(group: Group, sentences: list[str]) -> dict[str, Any]:
    block: dict[str, Any] = {"sentences": list(sentences)}
    if group.pin:
        block["slots"] = dict(group.pin)
    return block


def _intents(group: Group, sentences: list[str]) -> Intents | None:
    try:
        return Intents.from_dict(
            {
                "language": "en",
                "intents": {group.intent: {"data": [_block(group, sentences)]}},
                "lists": group.lists,
            }
        )
    except Exception:  # noqa: BLE001 - a line hassil will not parse is not a fold
        return None


def _answers(intents: Intents, text: str, pin: dict[str, Any]) -> bool:
    try:
        result = recognize(text, intents)
    except Exception:  # noqa: BLE001
        return False
    if result is None:
        return False
    return all(
        (entity := result.entities.get(name)) is not None and str(entity.value) == str(value)
        for name, value in pin.items()
    )


def covers(group: Group, folded: list[str]) -> str | None:
    """None when the folded lines answer every original; otherwise what they miss."""
    intents = _intents(group, folded)
    if intents is None:
        return "the matcher cannot parse the proposed lines"
    for sentence in group.sentences:
        examples = expansions(sentence, group.lists) if sentence in group.templates else [
            concrete(sentence, group.lists)
        ]
        for example in examples:
            if not _answers(intents, example, group.pin):
                return f"the folded lines do not answer {example!r}"
    return None


def _others(hass: HomeAssistant, language: str, group: Group) -> list[tuple[str, Intents, list[str]]]:
    """Every other group's sentences, as the matcher reads them, file by file."""
    found: list[tuple[str, Intents, list[str]]] = []
    root = sentences_root(hass, language)
    for path in sorted(root.glob("*.yaml")) if root.is_dir() else []:
        try:
            content = _load(path) or {}
        except LearnerError:
            continue
        lists = dict(content.get("lists") or {})
        intents_doc: dict[str, Any] = {}
        examples: list[str] = []
        for name, body in (content.get("intents") or {}).items():
            blocks = []
            for index, item in enumerate((body or {}).get("data") or []):
                if path == group.path and name == group.intent and index == group.index:
                    continue
                blocks.append(item)
                for sentence in (item or {}).get("sentences") or []:
                    examples.extend(expansions(str(sentence), lists, limit=30))
            if blocks:
                intents_doc[name] = {"data": blocks}
        if not intents_doc:
            continue
        try:
            intents = Intents.from_dict(
                {
                    "language": content.get("language", "en"),
                    "intents": intents_doc,
                    "lists": lists,
                    "expansion_rules": content.get("expansion_rules") or {},
                }
            )
        except Exception:  # noqa: BLE001 - a file the matcher cannot read takes nothing
            continue
        found.append((path.name, intents, examples))
    return found


def steals(hass: HomeAssistant, language: str, group: Group, folded: list[str], added: list[str]) -> str | None:
    """None when the fold takes nothing from anyone; otherwise what it would take."""
    mine = _intents(group, folded)
    if mine is None:
        return "the matcher cannot parse the proposed lines"
    for name, theirs, examples in _others(hass, language, group):
        for example in examples:
            try:
                if recognize(example, mine) is not None:
                    return f"it would answer {example!r}, which {name} already answers"
            except Exception:  # noqa: BLE001
                continue
        for example in added:
            try:
                if recognize(example, theirs) is not None:
                    return f"{example!r} is already answered by {name}"
            except Exception:  # noqa: BLE001
                continue
    return None


def _spread(items: list[str], count: int) -> list[str]:
    """`count` items spread evenly over a sorted list - a sample without randomness."""
    if len(items) <= count:
        return items
    step = len(items) / count
    return [items[int(i * step)] for i in range(count)]


async def _over_reach(
    hass: HomeAssistant, entry: ConfigEntry, group: Group, added: list[str]
) -> tuple[str | None, dict[str, float]]:
    """Ask the judge whether the sentences the fold adds are the same request."""
    sample = _spread(added, MAX_JUDGED)
    if not sample:
        return None, {}
    questions = {
        f"s{index}": Noul(
            f"`examples` are things a person said to a voice assistant - speech-to-text "
            f"mistakes included - and every one of them does the same thing to `target`. Would "
            f"a person who says `candidate_{index}` mean that same request on that same target? "
            "Allow for speech-to-text mistakes. Answer false if it could reasonably mean "
            "another device, another room or another action.",
            true="The same request on the same target",
            false="It could mean something else",
        )
        for index in range(len(sample))
    }
    state = {
        "examples": group.plain[:15] or group.sentences[:15],
        "target": group.pin or group.intent,
        **{f"candidate_{index}": text for index, text in enumerate(sample)},
    }
    try:
        answers = await async_ask(hass, entry, state, questions)
    except JudgeError as err:
        return f"the judge could not be asked: {err}", {}
    scores = {
        text: round(answers[f"s{index}"].noul or 0.0, 3) for index, text in enumerate(sample)
    }
    doubtful = [text for text, score in scores.items() if score < FOLD_SAMPLE_CLEAN]
    if doubtful:
        return f"the judge doubts {', '.join(repr(t) for t in doubtful)} means the same", scores
    return None, scores


# --------------------------------------------------------------------------
# Proposing and writing
# --------------------------------------------------------------------------


def _parse_templates(text: str) -> list[str]:
    lines: list[str] = []
    for raw in str(text or "").splitlines():
        line = raw.strip().lstrip("-*• ").strip().strip("`\"'")
        line = sentence_for_storage(line)
        if line and line not in lines:
            lines.append(line)
    return lines[:MAX_TEMPLATES]


async def _propose(hass: HomeAssistant, entry: ConfigEntry, group: Group) -> list[str] | str:
    from .autolearn import _ai_task_for  # noqa: PLC0415 - autolearn imports this module

    entity_id = _ai_task_for(hass, entry)
    if entity_id is None:
        return "no processing model"
    instructions = FOLD_INSTRUCTIONS.format(
        lines="\n".join(f"- {s}" for s in group.sentences),
        target=json.dumps(group.pin, default=str) if group.pin else group.intent,
        most=MAX_TEMPLATES,
    )
    try:
        response = await async_generate_data(
            hass,
            {
                "task_name": "fold phrasings into a template",
                "entity_id": entity_id,
                "instructions": instructions,
                "structure": FOLD_STRUCTURE,
            },
        )
    except Exception as err:  # noqa: BLE001 - a failed proposal changes nothing
        return f"the processing model could not propose: {err}"
    payload = (response or {}).get("data")
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except ValueError:
            payload = {"templates": payload}
    templates = _parse_templates((payload or {}).get("templates", ""))
    if not templates:
        return "the processing model proposed nothing"
    slots_before = {part for s in group.sentences for part in _slot_names(s)}
    if any(_slot_names(t) - slots_before for t in templates) or any("<" in t for t in templates):
        return "the proposal used syntax or slots the group does not have"
    return templates


def _slot_names(sentence: str) -> set[str]:
    return set(re.findall(r"\{([^}:]+)", sentence))


def _write(group: Group, sentences: list[str]) -> None:
    """Replace one group's sentences in place; its slots and response stay."""
    content = _load(group.path) or {}
    blocks = content["intents"][group.intent]["data"]
    blocks[group.index] = {**blocks[group.index], "sentences": list(sentences)}
    group.path.write_text(yaml_util.dump(content), encoding="utf-8")


async def async_fold_group(
    hass: HomeAssistant, entry: ConfigEntry, group: Group, *, apply: bool
) -> Outcome:
    """Propose, prove and (with ``apply``) write the fold of one group."""
    outcome = Outcome(group)
    language = language_of(entry)
    if not has_judge(entry):
        outcome.why = "folding needs a classification model to check what a fold adds"
        return outcome

    proposal = await _propose(hass, entry, group)
    if isinstance(proposal, str):
        outcome.why = proposal
        return outcome
    if {normalize_sentence(s) for s in proposal} == {normalize_sentence(s) for s in group.sentences}:
        outcome.why = "the proposal is the group as it is"
        return outcome
    outcome.folded = proposal

    if (missing := covers(group, proposal)) is not None:
        outcome.why = missing
        return outcome

    added = added_sentences(group, proposal)
    if (
        theft := await async_run(hass, steals, hass, language, group, proposal, added)
    ) is not None:
        outcome.why = theft
        return outcome

    doubt, outcome.samples = await _over_reach(hass, entry, group, added)
    if doubt is not None:
        outcome.why = doubt
        return outcome

    outcome.why = f"covers all {len(group.sentences)} phrasings and adds {len(added)} the judge accepts"
    if not apply:
        return outcome

    await async_run(hass, _write, group, proposal)
    await hass.services.async_call("conversation", "reload", {}, blocking=True)
    plain = [concrete(s, group.lists) for s in group.plain]
    if await async_verify(hass, language, plain, group.intent) == "unmatched":
        await async_run(hass, _write, group, group.sentences)
        await hass.services.async_call("conversation", "reload", {}, blocking=True)
        outcome.why = "the live matcher rejected the folded lines; the originals are back"
        return outcome
    outcome.applied = True
    _LOGGER.info(
        "Irabelle Home Learning folded %d phrasing(s) of %s into %s",
        len(group.sentences),
        group.intent,
        proposal,
    )
    return outcome


async def async_fold(
    hass: HomeAssistant,
    entry: ConfigEntry,
    intent: str | None = None,
    *,
    apply: bool = False,
    only_ready: bool = False,
) -> dict[str, Any]:
    """Fold every group of one intent, or all of them. A dry run unless ``apply``.

    ``only_ready`` is the automatic pass after a learning: only groups big enough to fold and
    changed since the last attempt are proposed. Writing a group replaces its sentences in
    place, so the other groups' positions stay valid through the loop.
    """
    language = language_of(entry)
    outcomes: list[Outcome] = []
    for group in await async_run(hass, groups, hass, language, intent):
        if only_ready:
            if not group.ready() or _ATTEMPTED.get(group.key) == group.fingerprint():
                continue
            _ATTEMPTED[group.key] = group.fingerprint()
        elif len(group.sentences) < 2:
            continue
        outcomes.append(await async_fold_group(hass, entry, group, apply=apply))
    applied = [o.as_dict() for o in outcomes if o.applied]
    if applied:
        entry.runtime_data.spend["last_fold"] = applied[-1]
        await async_save(hass, entry)
    return {"folded": [o.as_dict() for o in outcomes]}
