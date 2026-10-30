"""Folding phrasings together: the judge decides, the matcher confirms.

Two ways of asking the same thing are two lines in a sentence file, and the file is what a
person reads and an assistant searches. Home Assistant's template syntax says both in one line
- `what's the weather [like] (today | tomorrow)` - and this is the pass that gets there, which
is also what makes the wording *greedy*: one line covers the ways nobody has said yet.

The division of labour is the whole design:

* the **candidate** is arithmetic, and nothing else. Two sentences that differ in one span
  become `(this | that)`; one that is the other plus a word or two becomes `[the extra]`.
* whether the two wordings **mean the same thing** is a judgement, and it is exactly what the
  judge is for: one `noul` per candidate, every candidate in a single request. It is also the
  judgement that has to exist - `(is | was)` looks as symmetrical as `(today | tomorrow)` and
  is wrong, and no amount of string arithmetic can tell them apart.
* whether the **matcher accepts** the folded line is not a judgement at all. The real
  recogniser is asked, with a concrete value in every slot, and a merge it does not match for
  both originals is dropped. A model is never the reason a phrase stops working.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from hassil import Intents, recognize
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .intents import concrete, learner_sentences, replace_sentences, slot_sample
from .judge import JudgeError, Noul, async_ask, has_judge
from .learner import (
    LearnerError,
    _load,
    async_save,
    async_verify,
    language_of,
    normalize_sentence,
)

_LOGGER = logging.getLogger(__name__)

#: How sure the judge has to be that two wordings ask the same thing before they are folded.
#: A starting point to be replaced by measurement, like every other threshold here: the cost of
#: being wrong is one phrasing answering with its neighbour's intent, and the cost of being
#: cautious is a longer file.
MERGE_CLEAN = 0.80

#: Tokens that are already template syntax, so a sentence using them is left alone: folding a
#: folded line again is how `((a | b) | c)` appears, and nobody asked for that.
_STRUCTURED = ("[", "]", "(", ")", "<", ">")

#: The most tokens a folded span may hold, and the most candidates asked about per request.
#: Both bound the cost and the damage: a candidate that spans half a sentence is not a rewording
#: of it, it is a different sentence.
MAX_SPAN = 3
MAX_CANDIDATES = 40


@dataclass(frozen=True)
class Candidate:
    """One folded sentence that would replace two, and the two it replaces."""

    intent: str
    a: str
    b: str
    merged: str
    kind: str  # "alternative" | "optional"
    prefix: tuple[str, ...] = ()
    suffix: tuple[str, ...] = ()
    options: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        return self.merged

    @property
    def shape(self) -> tuple:
        return (self.intent, self.kind, self.prefix, self.suffix)


@dataclass(frozen=True)
class Fold:
    """One line for a whole family of phrasings, and every phrasing it replaces."""

    intent: str
    kind: str
    merged: str
    sources: tuple[str, ...]
    score: float = 0.0


def group(candidates: Iterable[Candidate], limit: int = 8) -> list[Fold]:
    """One line per family of alternative phrasings, so three do not become three lines.

    `is the TV on`, `is the telly on` and `is the tv turned on` differ at the same place, and
    the useful line is `is the (TV | telly | tv turned) on` - not one line per pair, which
    would leave the file exactly as long as it started. Every phrasing that shares a shape goes
    into that one line, up to `limit` of them: a template with twenty alternatives is a file
    nobody reads, so the phrasings that do not fit keep their own line instead of being lost.

    Optional words are not grouped: two different optional extras at the same place are two
    different lines, and a phrasing is only ever taken by one of them.
    """
    families: dict[tuple, list[Candidate]] = {}
    optional: list[Candidate] = []
    for candidate in candidates:
        if candidate.kind == "alternative":
            families.setdefault(candidate.shape, []).append(candidate)
        else:
            optional.append(candidate)

    folds: list[Fold] = []
    used: set[str] = set()
    # Biggest family first, and a phrasing already spoken for is not taken again: the same three
    # phrasings can be split at two different places (`is the living room (TV | telly) on` and
    # `is the (living room TV | lounge tv) on`), and writing both would leave two redundant
    # lines. The family that replaces the most phrasings wins; what is left over keeps its own
    # line, which is where it started.
    for (intent, kind, prefix, suffix), family in sorted(
        families.items(), key=lambda item: -len(item[1])
    ):
        options: list[str] = []
        sources: list[str] = []
        for candidate in family:
            for option, source in zip(candidate.options, (candidate.a, candidate.b), strict=False):
                if not option or source in used or option in options or len(options) >= limit:
                    continue
                options.append(option)
                sources.append(source)
        if len(options) < 2:
            continue
        merged = _join([*prefix, f"({' | '.join(options)})", *suffix])
        folds.append(Fold(intent=intent, kind=kind, merged=merged, sources=tuple(sources)))
        used.update(sources)
    for candidate in optional:
        if candidate.a in used or candidate.b in used:
            continue
        folds.append(
            Fold(
                intent=candidate.intent,
                kind="optional",
                merged=candidate.merged,
                sources=(candidate.a, candidate.b),
            )
        )
        used.update((candidate.a, candidate.b))
    return folds


def _tokens(sentence: str) -> list[str]:
    return sentence.split()


def _join(parts: Iterable[str]) -> str:
    return " ".join(part for part in parts if part).strip()


def _foldable(sentence: str) -> bool:
    return not any(mark in sentence for mark in _STRUCTURED)


def candidates(intent: str, sentences: Iterable[str]) -> list[Candidate]:
    """Every way these phrasings fold into one line, by arithmetic alone.

    Two shapes, both from the template syntax: a *span* that differs becomes an alternative
    (`(on | off)`), and words one phrasing has and the other does not become optional
    (`[like]`). The order is stable so a report reads the same twice.
    """
    usable = [sentence for sentence in dict.fromkeys(sentences) if _foldable(sentence)]
    found: dict[str, Candidate] = {}
    for index, a in enumerate(usable):
        for b in usable[index + 1 :]:
            for candidate in _span_candidate(intent, a, b):
                found.setdefault(candidate.key, candidate)
            for candidate in _optional_candidates(intent, a, b):
                found.setdefault(candidate.key, candidate)
    return list(found.values())[:MAX_CANDIDATES]


def _shared_ends(a: list[str], b: list[str]) -> tuple[list[str], list[str], int]:
    """The common prefix and suffix of two token lists, and the suffix's length."""
    prefix = 0
    while prefix < min(len(a), len(b)) and a[prefix] == b[prefix]:
        prefix += 1
    suffix = 0
    while (
        suffix < min(len(a), len(b)) - prefix
        and a[len(a) - 1 - suffix] == b[len(b) - 1 - suffix]
    ):
        suffix += 1
    return a[:prefix], a[len(a) - suffix :] if suffix else [], suffix


def _span_candidate(intent: str, a: str, b: str) -> list[Candidate]:
    tokens_a, tokens_b = _tokens(a), _tokens(b)
    if tokens_a == tokens_b:
        return []
    prefix, suffix, suffix_length = _shared_ends(tokens_a, tokens_b)
    span_a = tokens_a[len(prefix) : len(tokens_a) - suffix_length]
    span_b = tokens_b[len(prefix) : len(tokens_b) - suffix_length]
    if not span_a or not span_b:
        return []
    if len(span_a) > MAX_SPAN or len(span_b) > MAX_SPAN:
        return []
    merged = _join([*prefix, f"({' '.join(span_a)} | {' '.join(span_b)})", *suffix])
    return [
        Candidate(
            intent=intent,
            a=a,
            b=b,
            merged=merged,
            kind="alternative",
            prefix=tuple(prefix),
            suffix=tuple(suffix),
            options=(" ".join(span_a), " ".join(span_b)),
        )
    ]


def _optional_candidates(intent: str, a: str, b: str) -> list[Candidate]:
    """The shorter phrasing with the extra words of the longer one made optional."""
    short, long = sorted((_tokens(a), _tokens(b)), key=len)
    extra_length = len(long) - len(short)
    if not extra_length or extra_length > MAX_SPAN:
        return []
    for start in range(len(long) - extra_length + 1):
        if long[:start] + long[start + extra_length :] != short:
            continue
        extra = long[start : start + extra_length]
        merged = _join([*long[:start], f"[{' '.join(extra)}]", *long[start + extra_length :]])
        shorter, longer = (a, b) if len(_tokens(a)) == len(short) else (b, a)
        return [
            Candidate(
                intent=intent,
                a=shorter,
                b=longer,
                merged=merged,
                kind="optional",
                prefix=tuple(long[:start]),
                suffix=tuple(long[start + extra_length :]),
                options=(" ".join(extra),),
            )
        ]
    return []


def _pin_key(pins: dict[str, dict[str, Any]], sentence: str) -> tuple:
    """The pinned slot values of one sentence, as a comparable key."""
    return tuple(sorted(pins.get(normalize_sentence(sentence), {}).items()))


def pins_of(blocks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """What each sentence pins, as its group declares it (`slots:` under a list of sentences).

    `"what's the weather tomorrow"` is pinned `day: tomorrow`; the same words without the pin
    match and then answer about today. A fold has to carry the pin, or refuse.
    """
    pins: dict[str, dict[str, Any]] = {}
    for item in blocks:
        pin = dict((item or {}).get("slots") or {})
        for sentence in (item or {}).get("sentences") or []:
            pins[normalize_sentence(str(sentence))] = pin
    return pins


def _intent_data(hass: HomeAssistant, spec: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The `data:` blocks and the `lists:` of one learned intent, as its file has them."""
    path = Path(hass.config.path(spec.file))
    try:
        content = _load(path) or {}
    except LearnerError:
        return [], {}
    block = (content.get("intents") or {}).get(spec.name) or {}
    return list(block.get("data") or []), dict(content.get("lists") or {})


def covers(
    intent: str,
    merged: str,
    originals: list[str],
    lists: dict[str, Any],
    pin: dict[str, Any] | None = None,
) -> bool:
    """Whether the real matcher answers both originals with the folded line, slots and all.

    Offline, against the same hassil the house runs, with a concrete value in every slot: a
    slot left as `{day}` is not an utterance and the matcher is right to refuse it.

    ``pin`` is the `slots:` the folded sentence will carry, and the check is not only that the
    originals *match* but that the pinned values come back out of the match. Matching without
    them is how "what's the weather tomorrow" became a question about today: the sentence
    matched, nothing filled `day`, and the reply took its no-day branch.
    """
    block: dict[str, Any] = {"sentences": [merged]}
    if pin:
        block["slots"] = dict(pin)
    config = {
        "language": "en",
        "intents": {intent: {"data": [block]}},
        "lists": lists,
    }
    try:
        intents = Intents.from_dict(config)
    except Exception:  # noqa: BLE001 - a template hassil will not parse is not a merge
        return False
    sample = slot_sample(lists)
    for sentence in originals:
        try:
            result = recognize(concrete(sentence, lists), intents)
        except Exception:  # noqa: BLE001
            return False
        if result is None:
            return False
        for name, value in (pin or {}).items():
            entity = result.entities.get(name)
            if entity is None or str(entity.value) != str(value):
                return False
    return True


def _merge_question(folds: list[Fold]) -> dict[str, Any]:
    """The state: every family, and the line that would answer all of it."""
    return {
        "folds": [
            {"id": f"c{index}", "line": fold.merged, "covers": list(fold.sources)}
            for index, fold in enumerate(folds)
        ]
    }


async def _verdicts(
    hass: HomeAssistant, entry: ConfigEntry, folds: list[Fold]
) -> dict[str, float]:
    """One `noul` per fold, from a single judge request. A missing answer counts as 0."""
    answers = await async_ask(
        hass,
        entry,
        _merge_question(folds),
        {
            f"c{index}": Noul(
                f"One line of a smart home's sentence file is \"{fold.merged}\", and it would "
                "replace these wordings: "
                + " / ".join(f"\"{source}\"" for source in fold.sources)
                + ". Do they all ask for the same thing - the same answer, at the same moment, "
                "about the same thing - so that one template may answer every one of them? A "
                "change of tense, of who it is about, or of which device is meant makes them "
                "different requests, and a line that covers them would answer the wrong one.",
                true="Every wording asks for the same thing",
                false="They are different requests, or one is a different thing altogether",
            )
            for index, fold in enumerate(folds)
        },
    )
    scores: dict[str, float] = {}
    for index in range(len(folds)):
        answer = answers.get(f"c{index}")
        scores[f"c{index}"] = (answer.noul or 0.0) if answer is not None else 0.0
    return scores


async def async_combine(
    hass: HomeAssistant,
    entry: ConfigEntry,
    intent: str | None = None,
    *,
    apply: bool = False,
) -> dict[str, Any]:
    """Propose (and optionally write) folds for the phrasings of one intent, or all of them.

    Nothing is written unless ``apply`` is set, so the interesting half - what the judge said -
    can be read before it changes the house. Applying verifies the result against the *live*
    matcher and puts the originals back if it does not accept them.
    """
    if not has_judge(entry):
        raise LearnerError(
            "folding phrasings needs the judge: whether two wordings mean the same thing is a "
            "judgement, not a string operation. Configure a classification model, or leave the wording as "
            "it is"
        )
    language = language_of(entry)
    specs = [spec for spec in entry.runtime_data.intents if intent is None or spec.name == intent]
    if intent is not None and not specs:
        raise LearnerError(f"{intent!r} is not a learned intent")

    spoken = learner_sentences(hass, language, {spec.name for spec in specs})
    proposed: list[Candidate] = []
    pin_refused: list[dict[str, Any]] = []
    for spec in specs:
        pins = pins_of(_intent_data(hass, spec)[0])
        for candidate in candidates(spec.name, spoken.get(spec.name) or []):
            # A pair that pins different values cannot become one line, and it must not be
            # allowed to *claim* its phrasings either: `group` takes each phrasing for the
            # family that folds most of them, so a mixed pair left in would take one away
            # from the consistent family that could actually have used it.
            if _pin_key(pins, candidate.a) != _pin_key(pins, candidate.b):
                pin_refused.append(
                    {
                        "intent": candidate.intent,
                        "merged": candidate.merged,
                        "why": "the wordings pin different slot values, so one line cannot carry them",
                    }
                )
                continue
            proposed.append(candidate)
    # One line per family before anything is asked: the judge answers about the line that would
    # be written, not about a pair that is one of several ways to write it.
    folds = group(proposed)
    if not folds:
        # Still worth reporting what was looked at and refused: "nothing to fold" and "the
        # wordings pin different values" are different answers to the same question.
        return {
            "intents": [spec.name for spec in specs],
            "considered": 0,
            "merged": [],
            "refused": pin_refused,
            "applied": False,
        }

    try:
        verdicts = await _verdicts(hass, entry, folds)
    except JudgeError as err:
        raise LearnerError(f"the judge could not be asked: {err}") from err

    merged: list[dict[str, Any]] = []
    refused: list[dict[str, Any]] = []
    for index, fold in enumerate(folds):
        score = verdicts.get(f"c{index}", 0.0)
        spec = next(item for item in specs if item.name == fold.intent)
        blocks, lists = _intent_data(hass, spec)
        pins = pins_of(blocks)
        wanted = {
            tuple(sorted(pins.get(normalize_sentence(source), {}).items()))
            for source in fold.sources
        }
        if len(wanted) > 1:
            refused.append(
                {
                    "intent": fold.intent,
                    "merged": fold.merged,
                    "why": "the wordings pin different slot values, so one line cannot carry them",
                }
            )
            continue
        pin = dict(wanted.pop() or ())
        if score < MERGE_CLEAN:
            refused.append(
                {
                    "intent": fold.intent,
                    "merged": fold.merged,
                    "why": f"the judge scored it {score:.2f}",
                }
            )
            continue
        if not covers(fold.intent, fold.merged, list(fold.sources), lists, pin):
            refused.append(
                {
                    "intent": fold.intent,
                    "merged": fold.merged,
                    "why": "the matcher does not accept it for every wording it would replace",
                }
            )
            continue
        merged.append(
            {
                "intent": fold.intent,
                "kind": fold.kind,
                "from": list(fold.sources),
                "into": fold.merged,
                "pin": pin or None,
                "score": round(score, 3),
            }
        )

    report: dict[str, Any] = {
        "intents": [spec.name for spec in specs],
        "considered": len(folds),
        "merged": merged,
        "refused": pin_refused + refused,
        "applied": False,
    }
    if not apply or not merged:
        return report

    written: list[str] = []
    for spec in specs:
        originals = list(spoken.get(spec.name) or [])
        folded = [item for item in merged if item["intent"] == spec.name]
        if not folded:
            continue
        updated = list(originals)
        for item in folded:
            for original in item["from"]:
                if original in updated:
                    updated.remove(original)
            updated.append(item["into"])
        path = Path(hass.config.path(spec.file))
        remove = {original for item in folded for original in item["from"]}
        by_pin: dict[tuple, list[str]] = {}
        for item in folded:
            key = tuple(sorted((item.get("pin") or {}).items()))
            by_pin.setdefault(key, []).append(item["into"])
        for key, lines in by_pin.items():
            replace_sentences(path, spec.name, remove, lines, pin=dict(key))
        # The matcher caches what it read, so the fold is not in force until it re-reads the
        # directory - and the check below would be asking about the sentences that just left.
        await hass.services.async_call("conversation", "reload", {}, blocking=True)
        verdict = await async_verify(hass, language, sorted(remove), spec.name)
        if verdict == "unmatched":
            replace_sentences(path, spec.name, {item["into"] for item in folded}, sorted(remove))
            await hass.services.async_call("conversation", "reload", {}, blocking=True)
            report["refused"].extend(
                {"intent": spec.name, "merged": item["into"], "why": "the live matcher rejected it"}
                for item in folded
            )
            report["merged"] = [item for item in merged if item["intent"] != spec.name]
            continue
        written.append(spec.name)

    if written:
        report["applied"] = True
        report["written"] = written
        await async_save(hass, entry)
    return report
