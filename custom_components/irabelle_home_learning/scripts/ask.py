#!/usr/bin/env python3
"""Drive one utterance through a real assist pipeline and print what came back.

This is the path a satellite takes: the local sentence matcher first, then the pipeline's
conversation agent, then whatever that agent falls back to. It is the only way to see
whether the house already answers something locally - `conversation.process` goes straight
to an agent and skips the matcher entirely, which makes a phrase look unlearned when it is
not.

    export HA_TOKEN=...            # Profile -> Long-lived access tokens
    python3 scripts/ask.py --list
    python3 scripts/ask.py "extinguish the office light"
    python3 scripts/ask.py --pipeline Jev "when was the office light last on"
    python3 scripts/ask.py --json "turn on the lamp"

Exit code is 0 if the pipeline answered. The time next to the answer is the whole turn,
which is what tells a local answer (single-digit milliseconds) from a model one.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import _ha  # noqa: E402


#: The agent this tool exists to exercise. A pipeline that uses it is the only one whose
#: traces say anything about the learner.
LEARNER = "conversation.irabelle_home_learning"


async def available() -> list[dict]:
    """The pipeline list, over the API rather than out of Home Assistant's storage."""
    async with _ha.connect() as ha:
        listing = await ha.command(type="assist_pipeline/pipeline/list")
    return listing["result"]["pipelines"]


def choose(wanted: str | None, pipelines: list[dict]) -> str:
    """Pick a pipeline.

    Defaulting to the *first* one is wrong and quietly so: Home Assistant ships a pipeline
    of its own, and a sentence sent through that never reaches this integration - the tool
    then reports what the default agent says and looks like it is testing the learner. So
    the default is the first pipeline that actually uses it.
    """
    if not pipelines:
        raise SystemExit("there are no pipelines to run")
    if wanted is not None:
        for item in pipelines:
            if wanted.lower() in ((item["name"] or "").lower(), item["id"]):
                return item["id"]
        raise SystemExit(f"no pipeline called {wanted!r}; try --list")
    for item in pipelines:
        if item.get("conversation_engine") == LEARNER:
            return item["id"]
    raise SystemExit(
        f"no pipeline uses {LEARNER}; pass --pipeline, or see --list"
    )


async def run(pipeline_id: str, text: str, *, raw: bool = False) -> int:
    """Run the pipeline and read its events.

    `assist_pipeline/run` is a subscription, not a request: it acknowledges nothing and
    then streams events, so the events are read here directly. Waiting for a `result` for
    this command swallows every event and looks exactly like a hang.
    """
    started = time.monotonic()
    async with _ha.connect() as ha:
        await ha.ws.send_json(
            {
                "id": 1,
                "type": "assist_pipeline/run",
                "start_stage": "intent",
                "end_stage": "intent",
                "pipeline": pipeline_id,
                "input": {"text": text},
            }
        )
        speech = None
        error = None

        async def read() -> None:
            nonlocal speech, error
            while True:
                message = await ha.ws.receive_json()
                kind = message.get("type")
                if kind == "result" and message.get("success") is False:
                    error = message.get("error")
                    return
                if kind != "event":
                    continue
                event = message["event"]
                if event["type"] == "intent-end":
                    output = (event.get("data") or {}).get("intent_output") or {}
                    speech = ((output.get("response") or {}).get("speech") or {}).get(
                        "plain", {}
                    ).get("speech")
                elif event["type"] in ("run-end", "error"):
                    if event["type"] == "error":
                        error = event.get("data")
                    return

        try:
            await asyncio.wait_for(read(), timeout=60)
        except asyncio.TimeoutError:
            error = {"code": "timeout", "message": "no run-end within 60s"}

    elapsed = (time.monotonic() - started) * 1000
    if raw:
        print(json.dumps({"speech": speech, "error": error}, indent=2))
    elif error:
        print(f"pipeline error: {error}")
        return 1
    else:
        print(f"{elapsed:8.0f} ms  {speech}")
    return 1 if error else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", help="what to say")
    parser.add_argument("--pipeline", help="pipeline name or id (default: the first)")
    parser.add_argument("--list", action="store_true", help="list the pipelines and exit")
    parser.add_argument("--json", action="store_true", help="print the raw result")
    args = parser.parse_args()

    async def go() -> int:
        pipelines = await available()
        if args.list:
            for item in pipelines:
                marker = " <- learner" if item.get("conversation_engine") == LEARNER else ""
                print(f"{item['id']}  {item['name']}{marker}")
            return 0
        return await run(choose(args.pipeline, pipelines), args.text, raw=args.json)

    if not args.text and not args.list:
        parser.error("give an utterance, or --list")
    return asyncio.run(go())


if __name__ == "__main__":
    sys.exit(main())
