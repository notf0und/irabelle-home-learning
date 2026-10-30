#!/usr/bin/env python3
"""Call a Home Assistant service and show its result, or its real error.

The REST API turns a service's validation error into a bare `HTTP 500 / Server got itself
in trouble`; the websocket API returns `{"code": ..., "message": ...}` - the same thing the
Dev Tools UI and the LLM tools see. That is the difference between debugging a component
and guessing at one.

    python3 scripts/service.py irabelle_home_learning.judge '{}'
    python3 scripts/service.py irabelle_home_learning.validate '{}'
    python3 scripts/service.py irabelle_home_learning.observations '{"state": "ignored"}'
    python3 scripts/service.py irabelle_home_learning.forget '{"utterance": "what time is it"}'

Any service works, not just this integration's. The body is JSON; `{}` if omitted. Services
that do not return a response are called with `return_response` off, which the script
detects from the error and retries - so `light.turn_on` works too.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import _ha  # noqa: E402


async def call(service: str, data: dict, keep_quiet: bool) -> int:
    async with _ha.connect() as ha:
        domain, name = service.split(".", 1)
        message = await ha.command(
            type="call_service",
            domain=domain,
            service=name,
            service_data=data,
            return_response=True,
        )
        if not message.get("success"):
            error = message.get("error") or {}
            if "does not support responses" in str(error.get("message", "")):
                message = await ha.command(
                    type="call_service",
                    domain=domain,
                    service=name,
                    service_data=data,
                )
            else:
                print(f"failed: {error.get('code')}: {error.get('message')}")
                return 1
        if message.get("success"):
            result = message.get("result")
            if not keep_quiet and result is not None:
                print(json.dumps(result, indent=2, default=str))
            return 0
        print(f"failed: {(message.get('error') or {}).get('message')}")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("service", help="domain.service, e.g. irabelle_home_learning.judge")
    parser.add_argument("data", nargs="?", default="{}", help="JSON body (default: {})")
    parser.add_argument("--quiet", action="store_true", help="print nothing on success")
    args = parser.parse_args()
    try:
        body = json.loads(args.data)
    except ValueError as err:
        return int(bool(sys.stderr.write(f"the body is not JSON: {err}\n")))
    return asyncio.run(call(args.service, body, args.quiet))


if __name__ == "__main__":
    sys.exit(main())
