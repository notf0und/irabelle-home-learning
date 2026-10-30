"""Where an exported `intent_script` registration goes.

A learned intent is a name that needs a handler, and `intent_script` is Home Assistant's only
YAML way to register one without a component. *Where* that lives is the house's business, so
this follows whatever the house already says rather than insisting on a layout of its own:

| `configuration.yaml` says | the stubs go |
|---|---|
| `intent_script:` with the entries inline under it | into that block, in `configuration.yaml` |
| `intent_script: !include foo.yaml` | into `foo.yaml`, beside whatever it already holds |
| `intent_script: !include_dir_merge_named <dir>` | `<dir>/irabelle_home_learning.yaml` |
| `intent_script: !include_dir_named <dir>` | one file per intent, `<dir>/<Intent>.yaml` |
| nothing at all | `intent_script.yaml` at the root, and the include line added |

Two things make writing into somebody else's file safe. Every entry this component writes
carries a comment naming the sentence file that holds the intent - the words and the reply live
there - and that comment is how the entry is found again to be updated or removed. A
hand-written entry has no such comment and is never touched, so a house that keeps its own
intents in the same file keeps them exactly as they are. And `configuration.yaml` is the one
file whose syntax error stops Home Assistant from starting, so after any edit it is re-loaded
the way Home Assistant loads it (includes, secrets and all) and put back if that fails.

The `export_path` option therefore only decides two things: the file name used inside a
directory the house declared, and where the file goes when the house has declared nothing.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.util import yaml as yaml_util

from .const import CONF_EXPORT_PATH, DEFAULT_EXPORT_PATH, INTENT_NAME_RE
from .learner import LearnedIntent, LearnerError, language_of

_LOGGER = logging.getLogger(__name__)

CONFIGURATION_FILE = "configuration.yaml"
#: What the comment on a written entry starts with, so it can be recognised again. The rest of
#: it is the sentence file that holds the intent.
MARK = "custom_sentences/"

#: The line that pulls our own mapping in, when the house has not said where it goes.
def _include_line(relative: Path) -> str:
    return f"intent_script: !include {relative.as_posix()}"


#: `intent_script:` at the start of a line, and whatever was written after the colon.
_DECLARATION = re.compile(r"^intent_script:[ \t]*(?P<rest>[^\n]*)$", re.MULTILINE)
#: A key line written by us, e.g. `WeatherForecast:  # custom_sentences/en/learned-….yaml`.
_WRITTEN_KEY = re.compile(
    r"^(?P<indent>[ \t]*)(?P<name>[A-Za-z][A-Za-z0-9_]{2,60}):[ \t]*#[ \t]*"
    + re.escape(MARK)
)


@dataclass
class Registration:
    """How this house declares `intent_script`, and where the stubs therefore belong."""

    kind: str  # inline | file | merge_dir | named_dir | none | list | unsupported
    path: Path | None  # the file, or the directory, the stubs live in
    declaration: str  # what `configuration.yaml` says, for the report
    include: str  # the line `configuration.yaml` needs to be whole


def _chosen_path(hass: HomeAssistant, entry: ConfigEntry) -> Path:
    """The file name this component uses when the house has not named one itself.

    The value comes from an option, so it is user input: an absolute path, or one that climbs
    out of the configuration folder, is refused rather than written somewhere else.
    """
    configured = str(entry.options.get(CONF_EXPORT_PATH) or "").strip()
    if not configured:
        # Left empty on purpose: the option is optional, and empty means the default.
        return Path(DEFAULT_EXPORT_PATH)
    candidate = Path(configured)
    if candidate.is_absolute() or ".." in candidate.parts:
        _LOGGER.warning(
            "Irabelle Home Learning: export_path %r is not a plain path inside the configuration folder; "
            "using %s instead",
            configured,
            DEFAULT_EXPORT_PATH,
        )
        candidate = Path(DEFAULT_EXPORT_PATH)
    return candidate


def _argument(declaration: str) -> str:
    """The path after an include tag, with any trailing comment removed."""
    return declaration.split("#")[0].split(None, 1)[1].strip().strip("\"'")


def detect(hass: HomeAssistant, entry: ConfigEntry) -> Registration:
    """Read `configuration.yaml` and work out where a registration belongs."""
    config_dir = Path(hass.config.config_dir)
    path = config_dir / CONFIGURATION_FILE
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    match = _DECLARATION.search(text)

    if match is None:
        chosen = _chosen_path(hass, entry)
        return Registration("none", config_dir / chosen, "nothing", _include_line(chosen))

    declaration = match.group("rest").strip()
    bare = declaration.split("#")[0].strip()
    if bare.startswith("!include_dir_merge_named"):
        directory = _argument(bare)
        return Registration("merge_dir", config_dir / directory, bare, f"intent_script: {bare}")
    if bare.startswith("!include_dir_named"):
        directory = _argument(bare)
        return Registration("named_dir", config_dir / directory, bare, f"intent_script: {bare}")
    if bare.startswith("!include") and not bare.startswith("!include_dir"):
        target = _argument(bare)
        return Registration("file", config_dir / target, bare, f"intent_script: {bare}")
    if bare.startswith("!include_dir"):
        # A list of intent configs has no names to register, so nothing can be added to it.
        return Registration("list", None, bare, f"intent_script: {bare}")
    if not bare:
        return Registration("inline", None, "an inline mapping", "intent_script:")
    return Registration("unsupported", None, bare, f"intent_script: {bare}")


def _entry_lines(spec: LearnedIntent, indent: str) -> str:
    """One written entry: the intent name, a pointer to its file, and the empty handler."""
    note = f"  # {spec.file}" if spec.file else ""
    return f"{indent}{spec.name}:{note}\n{indent}  action: []\n"


def _header(include: str) -> str:
    """The comment above a file this component owns outright."""
    return f"""\
# Written by the Irabelle Home Learning integration's `export` service. Do not edit by hand - run the
# service again instead.
#
# One entry per learned intent that needs a handler of its own, and the sentence file that
# holds it named beside the entry. The words people say and the reply are NOT here: they are
# in that file, under `custom_sentences/<language>/learned-*.yaml`, which is where Home
# Assistant looks for both. This file registers the intent, which is the one thing a sentence
# file cannot do - a sentence whose intent nothing answers returns an error for that phrase.
#
# Kept current on every change to what is installed, so it cannot go stale. Add this to
# `configuration.yaml` once if it is not there already:
#
#     {include}
#
# What is lost while this integration is not installed: the speaker variables. A reply that
# asks `person_name`/`user_name` is rendered with slots and state only, so it answers with its
# fallback for everyone.
"""


def _body(specs: list[LearnedIntent]) -> str:
    """Every entry, written by hand so each can carry its pointer comment."""
    body = ""
    for spec in sorted(specs, key=lambda item: item.name):
        if re.match(INTENT_NAME_RE, spec.name or ""):
            body += _entry_lines(spec, "")
        else:
            body += yaml_util.dump({spec.name: {"action": []}})
    # A file of nothing but comments loads as `None`, and the loader keeps only what is a dict,
    # so an empty registration says `{}` rather than reading as a mistake.
    return body or "{}\n"


def _written_blocks(text: str) -> list[tuple[str, int, int]]:
    """Every entry this component wrote, as (name, start, end) offsets into ``text``."""
    blocks: list[tuple[str, int, int]] = []
    lines = text.splitlines(keepends=True)
    offset = 0
    index = 0
    while index < len(lines):
        match = _WRITTEN_KEY.match(lines[index])
        if match is None:
            offset += len(lines[index])
            index += 1
            continue
        indent = match.group("indent")
        start = offset
        offset += len(lines[index])
        index += 1
        # The body of the entry: everything more indented than its key.
        while index < len(lines):
            stripped = lines[index].strip()
            if stripped and not lines[index].startswith(indent + " ") and not lines[
                index
            ].startswith(indent + "\t"):
                break
            offset += len(lines[index])
            index += 1
        blocks.append((match.group("name"), start, offset))
    return blocks


def _mapping_text(text: str, specs: list[LearnedIntent], indent: str, at: int) -> str:
    """The mapping with our entries brought up to date, leaving every other entry alone."""
    wanted = {spec.name: spec for spec in specs}
    written = {name: (start, end) for name, start, end in _written_blocks(text)}

    # Out with the entries we wrote that are no longer wanted, from the end so the offsets
    # before them stay valid.
    for name, (start, end) in sorted(written.items(), key=lambda item: -item[1][0]):
        if name not in wanted:
            text = text[:start] + text[end:]
            at -= end - start
            written.pop(name)

    # In with the ones that are missing.
    additions = ""
    for name, spec in sorted(wanted.items()):
        if name in written or not re.match(INTENT_NAME_RE, name or ""):
            continue
        additions += _entry_lines(spec, indent)
    if additions:
        if at and not text[:at].endswith("\n"):
            additions = "\n" + additions
        text = text[:at] + additions + text[at:]
    return text


def _inline_block(text: str) -> tuple[int, str]:
    """Where an inline `intent_script:` block ends, and the indentation its entries use."""
    match = _DECLARATION.search(text)
    assert match is not None
    lines = text.splitlines(keepends=True)
    offset = 0
    seen = False
    indent: str | None = None
    for line in lines:
        if not seen:
            offset += len(line)
            if offset >= match.end():
                seen = True
            continue
        if not line.strip() or line.lstrip().startswith("#"):
            offset += len(line)
            continue
        # The next top-level key ends the block. Anything indented is an entry of it, and the
        # *first* one sets the indentation the new entries are written with - taking the last
        # would follow the deepest line in the block and nest the entry inside another one.
        if not line[0].isspace():
            return offset, indent or "  "
        if indent is None:
            indent = line[: len(line) - len(line.lstrip())]
        offset += len(line)
    return len(text), indent or "  "


def _write_checked(hass: HomeAssistant, path: Path, text: str) -> None:
    """Write a file, and put it back if Home Assistant could not load the result.

    Only used for `configuration.yaml`, where a syntax error is a house that will not start.
    The previous text is kept under `irabelle_home_learning/backups/` as well, so a person can see what
    changed without digging through the store.
    """
    original = path.read_text(encoding="utf-8") if path.exists() else ""
    if original == text:
        return
    backups = Path(hass.config.path("irabelle_home_learning", "backups"))
    backups.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")
    (backups / f"{path.name}-{stamp}").write_text(original, encoding="utf-8")
    path.write_text(text, encoding="utf-8")
    try:
        yaml_util.load_yaml(path)
    except Exception as err:  # noqa: BLE001 - any loader complaint means put it back
        path.write_text(original, encoding="utf-8")
        raise LearnerError(
            f"the change to {path.name} did not load, so it was put back unchanged: {err}"
        ) from err


def _add_include(hass: HomeAssistant, line: str) -> bool:
    """Add the include line to `configuration.yaml`, if it is not there already."""
    path = Path(hass.config.path(CONFIGURATION_FILE))
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if _DECLARATION.search(text):
        return False
    addition = (
        "\n# Written by the Irabelle Home Learning integration: registers the intents the house learned\n"
        "# for itself, so their phrases keep answering from their sentence files if it is\n"
        "# removed. One empty handler each; the words and the replies are in custom_sentences/.\n"
        f"{line}\n"
    )
    _write_checked(hass, path, text.rstrip("\n") + "\n" + addition)
    return True


def sync(hass: HomeAssistant, entry: ConfigEntry) -> dict[str, Any]:
    """Bring the registration in line with what is installed, wherever this house keeps it."""
    registration = detect(hass, entry)
    specs = [spec for spec in entry.runtime_data.intents]
    names = sorted(spec.name for spec in specs)
    language = language_of(entry)
    report: dict[str, Any] = {
        "declaration": registration.declaration,
        "where": registration.kind,
        "intents": names,
        "count": len(names),
        "written": [],
        "removed": [],
    }

    if registration.kind == "list":
        raise LearnerError(
            "`intent_script` is a list in configuration.yaml, and a list of intents has no "
            "names to register; use `!include <file>`, `!include_dir_named <dir>` or "
            "`!include_dir_merge_named <dir>` instead"
        )
    if registration.kind == "unsupported":
        raise LearnerError(
            f"`intent_script` is declared as {registration.declaration!r}, which this service "
            "cannot add to; use `!include`, `!include_dir_named` or `!include_dir_merge_named`"
        )

    changed = False
    if registration.kind == "merge_dir":
        target = registration.path / _chosen_path(hass, entry).name
        text = _header(registration.include) + _body(specs)
        changed = _write_if_different(target, text)
        report["file"] = str(target)
        report["include"] = registration.include
        report["written"] = names

    elif registration.kind == "named_dir":
        directory = registration.path
        directory.mkdir(parents=True, exist_ok=True)
        wanted = {spec.name for spec in specs if re.match(INTENT_NAME_RE, spec.name or "")}
        for existing in sorted(directory.glob("*.yaml")):
            if existing.stem in wanted:
                continue
            if MARK in existing.read_text(encoding="utf-8", errors="ignore"):
                existing.unlink()
                report["removed"].append(existing.stem)
                changed = True
        for spec in sorted(specs, key=lambda item: item.name):
            if not re.match(INTENT_NAME_RE, spec.name or ""):
                continue
            text = f"# {spec.file}\naction: []\n"
            changed |= _write_if_different(directory / f"{spec.name}.yaml", text)
            report["written"].append(spec.name)
        report["file"] = str(directory)
        report["include"] = registration.include

    elif registration.kind == "none":
        target = registration.path
        text = _header(registration.include) + _body(specs)
        changed = _write_if_different(target, text)
        if _add_include(hass, registration.include):
            changed = True
        report["file"] = str(target)
        report["include"] = registration.include

    else:  # inline, or a file the house already includes
        if registration.kind == "inline":
            path = Path(hass.config.path(CONFIGURATION_FILE))
            text = path.read_text(encoding="utf-8")
            at, indent = _inline_block(text)
            updated = _mapping_text(text, specs, indent, at)
            before = _written_blocks(text)
            _write_checked(hass, path, updated)
            after = _written_blocks(updated)
            report["file"] = CONFIGURATION_FILE
            report["written"] = sorted({name for name, _, _ in after})
            report["removed"] = sorted({name for name, _, _ in before} - set(report["written"]))
            changed = updated != text
        else:
            path = registration.path
            text = path.read_text(encoding="utf-8") if path.exists() else ""
            before = _written_blocks(text)
            updated = _mapping_text(text, specs, "", len(text.rstrip("\n")) + 1)
            if text and not text.endswith("\n"):
                updated = updated.rstrip("\n") + "\n"
            if not text:
                updated = _header(registration.include) + _body(specs)
            _write_checked(hass, path, updated)
            after = _written_blocks(updated)
            report["file"] = str(path)
            report["written"] = sorted({name for name, _, _ in after})
            report["removed"] = sorted({name for name, _, _ in before} - set(report["written"]))
            changed = updated != text
        report["include"] = registration.include

    report["changed"] = changed
    report["applies_on_next_restart"] = True
    report["note"] = (
        "While the integration is installed its own handler is registered after the stub and "
        "answers, so nothing changes; the stubs are for the day it is gone. Bindings to "
        "built-in intents need no stub - Home Assistant registers those itself. What is lost "
        "without the integration is the speaker variables: a reply that asks person_name or "
        "user_name renders with slots and state only, so it answers with its fallback."
    )
    return report


def _write_if_different(path: Path, text: str) -> bool:
    """Write a file this component owns, and say whether anything actually changed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    path.write_text(text, encoding="utf-8")
    return True


def export_intent_script(hass: HomeAssistant, entry: ConfigEntry) -> dict[str, Any]:
    """Write the registration for every learned intent, wherever this house keeps it."""
    report = sync(hass, entry)
    _LOGGER.warning(
        "Irabelle Home Learning exported %d intent_script stub(s) into %s (%s); it is kept current from "
        "now on",
        report["count"],
        report["file"],
        report["where"],
    )
    return report


def _asked_for(hass: HomeAssistant, registration: Registration) -> bool:
    """Whether this house has a registration to keep current.

    A house that declares `intent_script` itself has already said where its registrations go,
    so ours is kept in step with it. One that says nothing is left alone until `export` is run
    once: insurance is asked for, and a learning must not edit `configuration.yaml` as a side
    effect of itself.
    """
    if registration.kind != "none":
        return True
    config = Path(hass.config.path(CONFIGURATION_FILE))
    text = config.read_text(encoding="utf-8") if config.exists() else ""
    return bool(registration.path.exists() and _DECLARATION.search(text))


def refresh_export(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Keep the registration current, if this house has one. Says whether it changed.

    A registration is only worth anything if it names the intents that exist: a stale one
    registers last month's set, so the phrases it misses are exactly the ones learned since -
    the failure it was written to prevent. Nothing here may raise into a learning, because a
    house that cannot learn is worse off than one whose registration is out of date.
    """
    try:
        registration = detect(hass, entry)
        if not _asked_for(hass, registration):
            return False
        report = sync(hass, entry)
    except Exception:  # noqa: BLE001 - a refresh must never break the write that asked for it
        _LOGGER.error("Irabelle Home Learning could not refresh the intent_script registration", exc_info=True)
        return False
    if not report["changed"]:
        return False
    _LOGGER.info(
        "Irabelle Home Learning kept the intent_script registration at %s current (%d stub(s))",
        report["file"],
        report["count"],
    )
    return True
