"""What Home Assistant's own sentence files say, read from the bundled intents package.

The built-in intents' wordings ship with Home Assistant (`home_assistant_intents`): 41 intents,
30 lists and 30 expansion rules for English in the version this house runs. Reading them is
worth more than guessing at two things in particular:

* **the slots a built-in intent expects.** `HassLightSet`'s sentences use `{brightness}`, and
  its handler reads that name. A learned phrasing that says `{level}` matches the sentence and
  then hands the intent nothing: the words are understood and the value is dropped. The names
  are in the package, so nobody has to guess them - and the generaliser is shown them.
* **the vocabulary a sentence may use for free.** `<the>`, `<numeric_value_set>`, `{color}`,
  `{brightness_level}` and the rest are loaded into the same matcher as our sentences, so a
  learned phrasing can lean on them instead of spelling them out.

Read-only, offline, and versioned with Home Assistant: nothing here writes, and nothing
depends on a rule or list being present - a name that is missing is simply not offered.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

#: `{slot}` and `{list:slot}` - the slot is the part after a colon, if there is one.
_SLOT = re.compile(r"\{([^{}]+)\}")

#: Slots every sentence may use, whatever the intent: Home Assistant resolves these from the
#: areas, floors and exposed entities before the sentence is matched.
PROVIDED_SLOTS = frozenset(
    {"name", "area", "floor", "domain", "device_class", "preferred_area_id", "preferred_floor_id"}
)


@lru_cache(maxsize=4)
def bundled(language: str) -> dict[str, Any]:
    """The built-in intents for a language, or an empty dict if the package is not there."""
    try:
        from home_assistant_intents import get_intents  # noqa: PLC0415 - optional dependency
    except ImportError:  # pragma: no cover - the tests and the house both have it
        return {}
    try:
        return dict(get_intents(language) or {})
    except Exception:  # noqa: BLE001 - a language with no bundle is not an error here
        return {}


def slot_name(reference: str) -> str:
    """The slot a `{...}` reference fills: `{brightness_level:brightness}` is `brightness`."""
    return reference.split(":", 1)[1].strip() if ":" in reference else reference.strip()


def expected_slots(language: str, intent: str) -> set[str]:
    """Every slot a built-in intent's own sentences fill, plus the ones always provided.

    An empty result means "nothing known about this intent" - it is not a built-in with a
    bundle, or the bundle is not installed - so a caller must not read it as "no slots".
    """
    block = (bundled(language).get("intents") or {}).get(intent)
    if not block:
        return set()
    names: set[str] = set(PROVIDED_SLOTS)
    for group in [block, *(block.get("data") or [])]:
        names.update(str(key) for key in (group.get("slots") or {}))
        for sentence in group.get("sentences") or []:
            names.update(slot_name(match) for match in _SLOT.findall(str(sentence)))
    return {name for name in names if name}


def vocabulary(language: str) -> tuple[set[str], set[str]]:
    """The expansion rules and list names a learned sentence may use for free."""
    data = bundled(language)
    return (
        {str(name) for name in (data.get("expansion_rules") or {})},
        {str(name) for name in (data.get("lists") or {})},
    )


def reference(language: str, intent: str, limit: int = 6) -> str:
    """The slot names an intent expects, for a prompt: `area, brightness, name`.

    Sorted and capped, because this goes into a prompt that is read by a model: the point is
    that it copies the right names, not that it reads a dictionary.
    """
    names = sorted(expected_slots(language, intent) - {"preferred_area_id", "preferred_floor_id"})
    if not names:
        return ""
    shown = names[:limit]
    return ", ".join(shown) + (", ..." if len(names) > limit else "")
