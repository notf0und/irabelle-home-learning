#!/usr/bin/env python3
"""Run every `test_*.py` beside this file, inside the Home Assistant container.

    docker exec homeassistant python3 /config/custom_components/irabelle_home_learning/tests/run.py

Unittest cannot await `asyncSetUp`, so the loop is here instead of in a runner: each test is
constructed, its async hooks awaited, and anything raised is reported with the traceback.
Exit code is the number of failures, so this can gate anything that runs it.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import os
import pathlib
import sys
import tempfile
import traceback
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
#: Whatever holds `custom_components/` - /config in the container, the repository root
#: in a checkout.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

#: Run from a throwaway directory, so a test that passes a *relative* path cannot quietly
#: read or write the live house config that happens to be mounted at /config. Every path in a
#: test goes through `hass.config.path(...)` (the temp config dir) or `sentences_dir()`.
os.chdir(tempfile.mkdtemp(prefix="irabelle_home_learning_cwd_"))


def cases() -> list[type[unittest.TestCase]]:
    found: list[type[unittest.TestCase]] = []
    for path in sorted(HERE.glob("test_*.py")):
        module = importlib.import_module(path.stem)
        for name, obj in vars(module).items():
            if (
                inspect.isclass(obj)
                and issubclass(obj, unittest.TestCase)
                and obj is not unittest.TestCase
            ):
                obj._module_name = module.__name__  # type: ignore[attr-defined]
                found.append(obj)
    return found


async def run_one(cls: type[unittest.TestCase], name: str) -> str | None:
    """Run one test, returning a traceback string when it fails."""
    instance = cls(name)
    try:
        if hasattr(instance, "asyncSetUp"):
            await instance.asyncSetUp()
        elif hasattr(instance, "setUp"):
            instance.setUp()
        outcome = getattr(instance, name)
        if inspect.iscoroutinefunction(outcome):
            await outcome()
        else:
            outcome()
    except Exception:  # noqa: BLE001 - a failing test is the point
        return traceback.format_exc()
    finally:
        try:
            if hasattr(instance, "asyncTearDown"):
                await instance.asyncTearDown()
            elif hasattr(instance, "tearDown"):
                instance.tearDown()
        except Exception:  # noqa: BLE001 - teardown must not mask the failure
            pass
    return None


async def main() -> int:
    passed = 0
    failures: list[tuple[str, str, str]] = []
    for cls in cases():
        module = getattr(cls, "_module_name", cls.__module__)
        names = sorted(n for n in dir(cls) if n.startswith("test_"))
        if not names:
            continue
        print(f"\n{module}.{cls.__name__} ({len(names)})")
        for name in names:
            problem = await run_one(cls, name)
            if problem is None:
                passed += 1
                print(f"  ok   {name}")
            else:
                failures.append((cls.__name__, name, problem))
                print(f"  FAIL {name}")
    print(f"\n{passed} passed, {len(failures)} failed")
    for cls_name, name, problem in failures:
        print(f"\n=== {cls_name}.{name} ===\n{problem}")
    return len(failures)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
