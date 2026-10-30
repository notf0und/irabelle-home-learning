#!/usr/bin/env python3
"""Read the last few assist-pipeline traces - the Debug tab of a pipeline, from a shell.

This is the view that shows *why* a phrase took the path it did: whether the local matcher
answered, which agent it went to, and what each stage produced. Use it before guessing why
a phrase is not local: a trace that ends in milliseconds never reached an agent at all,
and one with `intent-progress` events went to a model.

    python3 scripts/traces.py                 # last 3 runs of every pipeline that has any
    python3 scripts/traces.py 6 Jev           # last 6 runs of the pipeline called "Jev"
    python3 scripts/traces.py --full <pipeline_id> <run_id>

Home Assistant keeps the debug data in memory (ten runs per pipeline), so it is lost on
restart. For what was *learned* rather than what was said, read the observations:
`python3 scripts/service.py irabelle_home_learning.observations`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import _ha  # noqa: E402

#: Events worth showing. `intent-progress` is a streamed token delta and there can be
#: hundreds of them, so it is counted rather than printed.
QUIET = ("intent-progress",)


async def runs_for(ha: _ha.Socket, pipeline: dict) -> list[dict]:
    listing = await ha.command(
        type="assist_pipeline/pipeline_debug/list", pipeline_id=pipeline["id"]
    )
    return (listing.get("result") or {}).get("pipeline_runs") or []


async def events_for(ha: _ha.Socket, pipeline_id: str, run_id: str) -> list[dict]:
    message = await ha.command(
        type="assist_pipeline/pipeline_debug/get",
        pipeline_id=pipeline_id,
        pipeline_run_id=run_id,
    )
    return (message.get("result") or {}).get("events") or []


def speech_of(event: dict) -> str:
    """What was said, or the closest thing the trace has to it.

    An intent that acted without speaking - a routed action, an intent_script that only
    ran - leaves `intent-end` with no speech at all, and printing `None` for those hides
    which of the two happened.
    """
    output = (event.get("data") or {}).get("intent_output") or {}
    response = output.get("response") or {}
    speech = ((response.get("speech") or {}).get("plain") or {}).get("speech")
    if speech:
        # Template whitespace goes out as written, and blank lines are long pauses.
        return " ".join(str(speech).split())
    name = (output.get("intent") or {}).get("name")
    return f"(no speech: {name or response.get('response_type') or 'nothing'})"


def intent_of(event: dict) -> str:
    """Whether the local matcher answered or an agent was called.

    The debug payload does **not** carry the intent's name in this version - it carries
    `processed_locally`, and an earlier version of this tool read the absence of a name as
    "an agent answered", which labelled five-millisecond local answers as model calls. To
    find *which* intent answered, read the sentence files and their handlers; the trace
    will not tell you.
    """
    data = event.get("data") or {}
    name = ((data.get("intent_output") or {}).get("intent") or {}).get("name")
    if data.get("processed_locally"):
        return f"local matcher{'' if name is None else f' ({name})'}"
    if name:
        return f"agent, intent {name}"
    return "a conversation agent (nothing local matched)"


async def show(limit: int, only: str | None) -> int:
    async with _ha.connect() as ha:
        listing = await ha.command(type="assist_pipeline/pipeline/list")
        for pipeline in listing["result"]["pipelines"]:
            if only and only.lower() not in (pipeline["name"] or "").lower() and (
                only != pipeline["id"]
            ):
                continue
            recent = await runs_for(ha, pipeline)
            if not recent:
                continue
            print(f"### {pipeline['name']} ({pipeline['id']}) - {len(recent)} run(s) kept")
            for run in recent[-limit:]:
                print(f"  --- {run.get('timestamp')}  {run['pipeline_run_id']}")
                for event in await events_for(ha, pipeline["id"], run["pipeline_run_id"]):
                    kind = event.get("type")
                    if kind in QUIET:
                        continue
                    if kind == "intent-end":
                        print(f"      intent:   {intent_of(event)}")
                        print(f"      answered: {speech_of(event)}")
                    elif kind == "error":
                        print(f"      ERROR: {json.dumps(event.get('data'))[:300]}")
                    else:
                        print(f"      {kind}")
    return 0


async def full(pipeline_id: str, run_id: str) -> int:
    async with _ha.connect() as ha:
        for event in await events_for(ha, pipeline_id, run_id):
            print(f"[{event.get('type')}] {json.dumps(event.get('data'), default=str)[:2000]}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("limit", nargs="?", type=int, default=3)
    parser.add_argument("pipeline", nargs="?", help="pipeline name or id")
    parser.add_argument(
        "--full", nargs=2, metavar=("PIPELINE_ID", "RUN_ID"), help="dump one run entirely"
    )
    args = parser.parse_args()
    if args.full:
        return asyncio.run(full(*args.full))
    return asyncio.run(show(args.limit, args.pipeline))


if __name__ == "__main__":
    sys.exit(main())
