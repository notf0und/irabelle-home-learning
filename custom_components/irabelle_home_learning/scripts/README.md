# Tools

Verification tools for this component. They are here rather than in a scratch directory
because every claim about this integration should be checkable the same way twice, and
because the checks that found the bugs in it were all of this shape: drive the real thing,
read what actually happened.

Nothing in here is imported by the integration. Home Assistant ignores this directory.

These are the *live* checks - they drive the real pipeline and ask the real judge. The rules
they check are pinned offline in `tests/` (`scripts/test.sh`), and the reasoning behind them is
in [DESIGN.md](../DESIGN.md).

## Setup

```bash
export HA_TOKEN=...      # Profile -> Long-lived access tokens
export HA_URL=http://localhost:8123   # optional; the default
python3 scripts/ask.py --list
```

Only `aiohttp` is needed, which Home Assistant already provides - run them from the host or
from inside the container, whichever can reach the API.

## The tools

| Tool | What it answers |
|---|---|
| `ask.py` | What does the house actually say to this sentence? Goes through a **real pipeline**, so the local matcher runs first. A 1 ms answer is local; 2 s is a model. |
| `traces.py` | Why did that phrase take the path it did? Reads the last pipeline runs - the Debug tab of the pipeline, from a shell. |
| `service.py` | Call any service and see the **real error**. The REST API turns a validation error into a bare 500; this returns the message. |
| `probe.py` | Re-runs the screen's questions over fixed cases. Use it after changing question wording or a threshold. |
| `entities.py` | Set an entity's registry name or aliases, and see the composed friendly name. Exists to enforce *alias, never rename*. |

## What each one is for

**`ask.py`** - the only way to tell "learned" from "answered by a model". A phrase that
never reaches the agent is invisible to everything else here, which is exactly what you
want to confirm after a learning.

```bash
python3 scripts/ask.py "where is my phone"
python3 scripts/ask.py --pipeline Jev "when was the office light last on"
```

**`traces.py`** - the pipeline keeps ten runs per pipeline, in memory (lost on restart). For
what was *learned* rather than what was said, read the observations:

```bash
python3 scripts/traces.py 6 Jev
python3 scripts/service.py irabelle_home_learning.observations '{"state": "ignored"}'
```

**`service.py`** - any service, any integration:

```bash
python3 scripts/service.py irabelle_home_learning.judge '{}'
python3 scripts/service.py irabelle_home_learning.validate '{}'
python3 scripts/service.py irabelle_home_learning.duplicates '{}'
python3 scripts/service.py irabelle_home_learning.merge '{"into": "WeatherWeekday", "others": ["WeatherTomorrow"], "apply": true}'
python3 scripts/service.py irabelle_home_learning.rename_intent '{"name": "WeatherWeekday", "to": "WeatherForecast"}'
```

`duplicates` is the one to run when the same question comes back in two different lengths:
it lists every entity read by more than one learned intent, with what each one says. `merge`
reports its plan unless you pass `"apply": true`.

**`probe.py`** - copy `cases.example.json` to `cases.json` and replace the cases with real
mistakes from your own house; the examples are a correct learning, an audit with no request
behind it, and five ways a learning goes wrong. Each case expects `CLEAN` or `FLAGGED`.

```bash
cp scripts/cases.example.json scripts/cases.json
python3 scripts/probe.py
python3 scripts/probe.py flagged_scope --questions
```

The questions in `probe.py` are a **copy** of `async_screen_learning`'s. Change a question
there and change it here in the same commit, or this stops testing what the house runs.

**`entities.py`** - a registry `name` replaces the composed friendly name
(`"<device> <entity>"`) that templates and automations may be reading; an alias adds to it.
Prefer the alias.

```bash
python3 scripts/entities.py alias sensor.mi_rc_area_last_seen "Android TV remote"
python3 scripts/entities.py name sensor.iphone_alex_area_last_seen --clear
python3 scripts/entities.py show switch.living_room_tv_virtual_switch
```

## A reasonable loop while changing something

1. `python3 scripts/service.py irabelle_home_learning.validate '{}'` - is anything broken at all?
2. `python3 scripts/probe.py` - does the screen still separate good from bad?
3. `python3 scripts/ask.py "<the phrase you care about>"` - does it answer locally?
4. `python3 scripts/traces.py 3` - and if not, where did it go instead?
