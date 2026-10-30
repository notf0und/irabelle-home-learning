#!/usr/bin/env python3
"""Run the screen's questions over a set of hand-built cases and print pass or fail.

The screen asks a model for a snap judgement on every learning, so its *wording is code*:
change a question and its answers move. This replays a fixed set of cases through
`irabelle_home_learning.judge` - the same questions, the same state shape - so a wording change can be
checked instead of hoped about, and so a threshold can be moved on evidence.

    cp scripts/cases.example.json scripts/cases.json     # then edit for your house
    python3 scripts/probe.py                             # every case
    python3 scripts/probe.py clean flagged_scope         # named cases
    python3 scripts/probe.py --questions                 # also print the raw answers

The questions below are a *copy* of `async_screen_learning`'s. When you change one there,
change it here in the same commit, or this stops testing what the house actually runs. The
cases are examples: a correct learning, an audit with no request behind it, and four ways a
learning goes wrong. Build your own from real mistakes - that is where the value is.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import _ha  # noqa: E402

HERE = pathlib.Path(__file__).parent

#: The screen's thresholds, as constants in `autolearn.py`.
SCREEN_CLEAN = 0.60
SCREEN_MARGIN = 0.50

QUESTIONS = {
    "answers_the_request": {
        "type": "noul",
        "instructions": (
            "`what_the_template_says` is what `the_template_that_was_learned` answers, once "
            "for each person who lives here and once with nobody speaking. A template that "
            "branches on who is asking shows its fallback with nobody speaking, which is "
            "not by itself wrong. Does the answer it gives to a person answer `request`?"
        ),
        "criteria": {
            "true": "It answers what was asked, for whoever is asking",
            "false": "It answers a different question, or a different thing",
        },
    },
    "same_scope": {
        "type": "noul",
        "instructions": (
            "`the_phrasings_this_intent_now_answers` are all matched to this one intent and "
            "answered by `the_template_that_was_learned`. Is every one of them answered "
            "correctly by that template? A phrasing that names a different device, a "
            "different room, or a different property than the template gives is NOT "
            "answered correctly, however similar it sounds."
        ),
        "criteria": {
            "true": "Every phrasing is answered correctly by that template",
            "false": "At least one phrasing would be answered with the wrong device, room or property",
        },
    },
    "reads_a_real_entity": {
        "type": "noul",
        "instructions": (
            "`the_entities_this_template_reads` is every entity "
            "`the_template_that_was_learned` reads, with the state it reads right now. Does "
            "every one of them exist in this house? `(not in this house)` marks one that "
            "does not."
        ),
        "criteria": {
            "true": "Every entity it reads exists in this house",
            "false": "At least one is not in this house, so the answer is invented",
        },
    },
    "answers_the_speaker": {
        "type": "noul",
        "instructions": (
            "`what_the_template_says` is the same template answered once per person who "
            "lives here. If a phrasing asks about \"my\" something, every person should get "
            "their own answer - the lines should not be identical. Answer true when no "
            "phrasing is about the speaker."
        ),
        "criteria": {
            "true": "No phrasing is about the speaker, or the people get different answers",
            "false": "A phrasing is about the speaker and every person gets the same answer",
        },
    },
    "time_looks_right": {
        "type": "noul",
        "instructions": (
            "Is every time of day in `what_the_template_says` plausible for `request`, given "
            "`the_time_where_the_house_is`? A time twelve hours out - AM where it should be "
            "PM - is not."
        ),
        "criteria": {
            "true": "No time is present, or every time is plausible",
            "false": "A time is wrong, missing, or reads as a placeholder",
        },
    },
    "reads_like_a_sentence": {
        "type": "noul",
        "instructions": (
            "`what_the_template_says` is what the house will say out loud. Does it read like "
            "one sentence a person would say? Every word in place, nothing missing or "
            "repeated, and properly a sentence."
        ),
        "criteria": {
            "true": "It reads like something a person would say out loud",
            "false": "A word is missing, repeated or out of place",
        },
    },
    "what_is_wrong": {
        "type": "choice",
        "instructions": "What, if anything, is wrong with this learning?",
        "criteria": {
            "correct": "Every phrasing above is answered correctly by that template",
            "too_narrow": "A phrasing is answered too broadly for what it asks",
            "too_wide": "A phrasing is answered with less than it asks for",
            "different_device": "A phrasing names one thing and is answered with another",
            "wrong_property": "A phrasing asks one thing and is answered with another (asked WHEN, answered WHERE)",
            "missing_state": "The template reads something that is not there",
            "not_for_the_speaker": "The answer is the same for everyone when it is about the speaker",
            "none_of_these": "Something else is wrong",
        },
    },
}


def verdict(answers: dict) -> tuple[str, list[str]]:
    reasons: list[str] = []
    for name in (
        "answers_the_request",
        "same_scope",
        "reads_a_real_entity",
        "answers_the_speaker",
        "time_looks_right",
        "reads_like_a_sentence",
    ):
        value = (answers.get(name) or {}).get("noul")
        if value is None or value < SCREEN_CLEAN:
            reasons.append(f"{name}={value}")
    what = answers.get("what_is_wrong") or {}
    probabilities = sorted((what.get("probabilities") or {}).values(), reverse=True)
    margin = probabilities[0] - probabilities[1] if len(probabilities) > 1 else 1.0
    if what.get("choice") != "correct":
        reasons.append(f"what_is_wrong={what.get('choice')}")
    elif margin < SCREEN_MARGIN:
        reasons.append(f"what_is_wrong near-tie (margin {margin:.2f})")
    return ("CLEAN" if not reasons else "FLAGGED"), reasons


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="*", help="case names (default: all)")
    parser.add_argument("--file", default=None, help="cases file (default: cases.json)")
    parser.add_argument("--questions", action="store_true", help="print every answer")
    args = parser.parse_args()

    path = pathlib.Path(args.file) if args.file else HERE / "cases.json"
    if not path.exists():
        path = HERE / "cases.example.json"
    cases = json.loads(path.read_text())
    wanted = args.cases or sorted(cases)

    failures = 0
    async with _ha.connect() as ha:
        for name in wanted:
            if name not in cases:
                print(f"{name}: no such case in {path.name}")
                failures += 1
                continue
            case = cases[name]
            answers = await ha.call_service(
                "irabelle_home_learning.judge",
                {
                    "state": case["state"],
                    "questions": case.get("questions") or QUESTIONS,
                },
            )
            label, reasons = verdict(answers.get("answers") or {})
            expected = case.get("expect", "CLEAN")
            ok = "ok  " if label == expected else "FAIL"
            if label != expected:
                failures += 1
            print(f"{ok} {name:22} {label:8} {'; '.join(reasons)[:110]}")
            if args.questions:
                for question, answer in (answers.get("answers") or {}).items():
                    print(f"       {question:26} {json.dumps(answer, default=str)[:140]}")
    print(f"\n{len(wanted) - failures}/{len(wanted)} as expected ({path.name})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
