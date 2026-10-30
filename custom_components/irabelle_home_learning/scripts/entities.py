#!/usr/bin/env python3
"""Set entity registry names and aliases, and show what the model-facing lists will say.

Two different things, and the difference matters:

* **`name` replaces** what the entity is called everywhere - dashboard, history,
  automations - and it takes away the composed name. Home Assistant builds the friendly
  name of an entity with `has_entity_name` as `"<device name> <entity name>"`, so a sensor
  named `Area Last Seen` on a device called `OnePlus 7 Pro` usually reads
  `OnePlus 7 Pro Area Last Seen`; overriding the registry name replaces all of that. A
  template that derives a label from the friendly name breaks quietly when it goes.
* **`aliases` are additional.** They are what the intent matcher and the LLM exposure list
  accept *in addition* to the name, and they are what the lists in this component show. An
  entity whose own name is hardware can still be reachable as "Alex's watch".

So the rule this tool exists to enforce: **alias, never rename** - unless the entity has no
usable name at all.

    python3 scripts/entities.py show sensor.mi_rc_area_last_seen
    python3 scripts/entities.py alias sensor.mi_rc_area_last_seen "Android TV remote"
    python3 scripts/entities.py name sensor.iphone_alex_area_last_seen --clear
    python3 scripts/entities.py name climate.ac_living_room "Air conditioner"

Setting `name` to the empty string clears the override, which restores the composed name.
"""

from __future__ import annotations

import argparse
import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import _ha  # noqa: E402


async def change(action: str, entity_id: str, value: str, clear: bool) -> int:
    async with _ha.connect() as ha:
        if action == "alias":
            got = await ha.command(type="config/entity_registry/get", entity_id=entity_id)
            if not got.get("success"):
                print(f"failed: {(got.get('error') or {}).get('message')}")
                return 1
            aliases = [
                alias
                for alias in ((got.get("result") or {}).get("aliases") or [])
                if isinstance(alias, str)
            ]
            if value in aliases:
                print(f"{value!r} is already an alias of {entity_id}")
                return 0
            changes = {"aliases": [*aliases, value]}
        else:
            changes = {"name": "" if clear else value}
        message = await ha.command(
            type="config/entity_registry/update", entity_id=entity_id, **changes
        )
        if not message.get("success"):
            print(f"failed: {(message.get('error') or {}).get('message')}")
            return 1
        entry = message["result"]["entity_entry"]
        print(
            f"{entity_id}\n"
            f"  name={entry.get('name')!r} original={entry.get('original_name')!r}\n"
            f"  aliases={entry.get('aliases')}"
        )
    for item in _ha.states():
        if item["entity_id"] == entity_id:
            print(f"  friendly name now: {item['attributes'].get('friendly_name')!r}")
            break
    return 0


async def _remove(entity_id: str) -> int:
    """Delete a registry entry - what a renamed entity's stale entry needs."""
    async with _ha.connect() as ha:
        message = await ha.command(
            type="config/entity_registry/remove", entity_id=entity_id
        )
        if not message.get("success"):
            print(f"failed: {(message.get('error') or {}).get('message')}")
            return 1
    print(f"removed {entity_id} from the registry")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["show", "name", "alias", "remove"])
    parser.add_argument("entity_id")
    parser.add_argument("value", nargs="?", default="")
    parser.add_argument("--clear", action="store_true", help="remove the name override")
    args = parser.parse_args()

    if args.action == "remove":
        return asyncio.run(_remove(args.entity_id))
    if args.action == "show":
        for item in _ha.states():
            if item["entity_id"] == args.entity_id:
                attributes = item["attributes"]
                print(
                    f"{args.entity_id}\n  state={item['state']!r}\n"
                    f"  friendly_name={attributes.get('friendly_name')!r}"
                )
                return 0
        print(f"{args.entity_id} has no state: disabled, or from a removed integration")
        return 1
    if args.action == "alias" and not args.value:
        parser.error("alias needs a value")
    if args.action == "name" and not args.value and not args.clear:
        parser.error("name needs a value, or --clear")

    return asyncio.run(change(args.action, args.entity_id, args.value, args.clear))


if __name__ == "__main__":
    sys.exit(main())
