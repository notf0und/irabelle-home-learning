#!/usr/bin/env python3
"""Move hand-written replies out of intent_script.yaml and into the sentence files.

A reply belongs beside the sentences that trigger it. Home Assistant keeps it there
already - `responses: intents: <name>: default:` in the same YAML - and that is where the
built-in intents' wording lives. An `intent_script` keeps it in a second file, so reading
one question means opening two.

It also fixes the whitespace, for free: a response template is rendered by the default
agent, which ends with

    speech = " ".join(speech.strip().split())

while an `intent_script`'s `speech:` is spoken exactly as written, blank lines and all -
and a blank line is a long pause.

An `intent_script` entry stays behind as `action: []` - a handler that does nothing and
says nothing. That is not decoration: a `responses:` block supplies *wording*, it does not
register an intent, so deleting the entry leaves a sentence whose intent nothing answers
and the whole pipeline fails with "Unexpected error during intent recognition" for that
phrase. `action: []` keeps the intent registered and silent, which is what lets the default
agent fill in the reply from the sentence file.

Only entries with **no `action:`** can move. A script that computes a variable and speaks
it cannot: a response template is rendered with `slots`, `state` and `query` only, so
`action_response` does not exist there. Those are left alone and reported.

    python3 scripts/move_replies.py                  # dry run, says what it would do
    python3 scripts/move_replies.py --apply          # do it, after backing both files up
    python3 scripts/move_replies.py --apply --config /config

Run it inside the container (the config files are root-owned), then reload:

    docker exec homeassistant python3 /config/custom_components/irabelle_home_learning/scripts/move_replies.py --apply
    docker exec homeassistant python3 -c "..."   # or restart Home Assistant, which is simpler
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys

import yaml

HEAD = (
    "# Replies moved here from intent_script.yaml: a reply belongs beside the sentences\n"
    "# that trigger it, and a response template has its whitespace normalised before it is\n"
    "# spoken - an intent_script's speech is read out exactly as written, blank lines and\n"
    "# all. The entries that compute a value first (FindMyPhone, FindPartnerPhone, WhoAmI)\n"
    "# stay in intent_script.yaml: a response template cannot see their action variables.\n"
)


class Dumper(yaml.SafeDumper):
    """Dump multi-line strings as `|` blocks, so a 6,000-character template stays readable."""


def _represent_str(dumper: yaml.SafeDumper, value: str):
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


Dumper.add_representer(str, _represent_str)


def sentence_file(root: pathlib.Path, name: str) -> pathlib.Path | None:
    """The sentence file that declares an intent, preferring a hand-written one."""
    candidates = []
    for path in sorted(root.glob("*/")):
        for yaml_path in sorted(path.glob("*.yaml")):
            try:
                data = yaml.safe_load(yaml_path.read_text()) or {}
            except yaml.YAMLError:
                continue
            if name in (data.get("intents") or {}):
                candidates.append(yaml_path)
    for path in candidates:
        if not path.name.startswith("learned-"):
            return path
    return candidates[0] if candidates else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="/config")
    parser.add_argument("--apply", action="store_true", help="write the changes")
    args = parser.parse_args()

    config = pathlib.Path(args.config)
    script_path = config / "intent_script.yaml"
    sentences = config / "custom_sentences"
    if not script_path.exists():
        raise SystemExit(f"{script_path} does not exist")

    scripts = yaml.safe_load(script_path.read_text()) or {}
    movable: dict[str, tuple[str, pathlib.Path]] = {}
    stuck: list[str] = []
    for name, body in scripts.items():
        body = body or {}
        speech = body.get("speech")
        text = (speech or {}).get("text") if isinstance(speech, dict) else speech
        if not isinstance(text, str) or not text.strip():
            continue
        if "action" in body:
            stuck.append(name)
            continue
        target = sentence_file(sentences, name)
        if target is None:
            stuck.append(f"{name} (no sentence file declares it)")
            continue
        movable[name] = (text, target)

    for name, (text, target) in movable.items():
        print(f"{name}: {len(text)} chars -> {target.relative_to(config)}")
    for name in stuck:
        print(f"left alone: {name}")

    if not args.apply:
        print(f"\n{len(movable)} to move, {len(stuck)} left alone. Nothing written (dry run).")
        return 0

    backup = config / "intent_script.yaml.before-replies-moved"
    if not backup.exists():
        shutil.copy2(script_path, backup)
        print(f"backed up {script_path.name} -> {backup.name}")
    for name, (text, target) in movable.items():
        shutil.copy2(target, target.with_suffix(".yaml.before-replies-moved"))
        data = yaml.safe_load(target.read_text()) or {}
        data.setdefault("responses", {}).setdefault("intents", {})[name] = {"default": text}
        header = "" if target.read_text().startswith("#") else HEAD
        target.write_text(header + yaml.dump(data, Dumper=Dumper, sort_keys=False, allow_unicode=True))
        scripts[name] = {"action": []}
        print(f"moved {name} into {target.name}, left a silent handler behind")

    script_path.write_text(
        "# Replies live with their sentences now: see `responses:` in the file under\n"
        "# custom_sentences/<language>/ that declares each intent.\n"
        "#\n"
        "# An `action: []` below is a handler that does nothing and says nothing, which is\n"
        "# what lets the default agent fill in the reply from the sentence file. It is not\n"
        "# optional: a `responses:` block supplies wording, not an intent, and a sentence\n"
        "# whose intent nothing answers fails the whole pipeline.\n"
        "#\n"
        "# The entries that compute a value first cannot move at all - a response template is\n"
        "# rendered with slots, state and query, so their action variables do not exist there.\n"
        + yaml.dump(scripts, Dumper=Dumper, sort_keys=False, allow_unicode=True)
    )
    print(f"rewrote {script_path.name} with {len(scripts)} entry/entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
