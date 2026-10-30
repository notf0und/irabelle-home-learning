# Irabelle Home Learning

A conversation agent that watches what your voice assistant could not answer locally, and
turns it into something the *local* sentence matcher answers from then on. A phrase the house
has learned costs milliseconds and no API call, for ever after.

```
assist pipeline  ->  Irabelle Home Learning  ->  your conversation agent  ->  its fallback
                         (watches)                (answers, unchanged)
```

<p align="center"><img src="custom_components/irabelle_home_learning/brand/icon@2x.png" width="128" alt=""></p>

## Install

[![Open your Home Assistant instance and open a repository inside the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=notf0und&repository=irabelle-home-learning&category=integration)

1. Click the button above, or in HACS add `https://github.com/notf0und/irabelle-home-learning`
   as a custom repository (type *Integration*), then download **Irabelle Home Learning**.
2. Restart Home Assistant.

Or manually: copy `custom_components/irabelle_home_learning/` into your `config/custom_components/`.

Then *Settings → Devices & services → Add integration → Irabelle Home Learning*. Setup asks for
the language learned sentences are written in, then creates the first **agent**.

### Agents

An agent is what you pick as the **Conversation agent** of a voice assistant. Each has:

| Field | What it is |
|---|---|
| **Name** | how it is listed in the voice assistant's agent dropdown |
| **Answering agent** | the agent that really answers; its answer is returned unchanged |
| **Classification model** | answers the quick typed judgements on almost every miss |
| **Processing model** | writes and reviews the templates behind new answers |

Models are picked by name from every AI Task entity and conversation agent you have, plus
**Jev** when the Jev integration is installed. Add as many agents as you like from the
integration page (**Add agent**) — for example *Jev → DeepSeek* and *ChatGPT → DeepSeek* — and
change one with **⋮ → Reconfigure**. They share one store: whatever any agent learns, every
agent and the local matcher answer from then on.

At a glance:

* it learns **wording**, not behaviour — an action is bound to the intent that performs it, and
  only a question gets a template;
* it **binds** wording to an intent that already answered it, and only **generalises** (writes a
  template) when a text model has to;
* a missed turn is decided by the cheapest thing that can decide it — deterministic first, then
  an optional cheap judge, then a text model — and never twice for the same decision;
* it writes only `custom_sentences/<lang>/learned-*.yaml`.

Nothing here names an entity, a person, a room or a vendor. Point it at your own assistant and
it learns your house's own wording.

**How it works, and why it is built this way, is in [DESIGN.md](custom_components/irabelle_home_learning/DESIGN.md)** — the stages, the
judge and its measured thresholds, slots, and every rule this component holds itself to. This
file is what you need to run it.

## Options

Shared by every agent (**Configure** on the integration):

| Option | Default | Meaning |
|---|---|---|
| Language | `en` | which `custom_sentences/<lang>/` files are written |
| Learn automatically | on | learn from conversations at all |
| Times heard before learning | 2 | sightings before a request is generalised |
| intent_script export file | `configurations/intent_script/irabelle_home_learning.yaml` | where `export` writes when `configuration.yaml` does not say; leave empty for the default |

### Choosing the classification model

It answers typed questions — *pick one of these intents*, *how likely is this true* — and every
choice answers them in the same shape, so the learner behaves the same with or without Jev:

| Choice | Speed and cost | When |
|---|---|---|
| **Jev** | ~0.25 s, a fraction of a cent | the [Jev integration](https://github.com/AboveColin/HA-Jev) is installed: one key, one budget |
| **any AI Task entity** | one structured-output call | a small or local model is enough for classification |
| **any conversation agent** | one call, answered as JSON | a model that has no AI Task entity |
| **Jev API (direct)** | as Jev | Jev without its integration |
| **None** | free | learn by binding only; routing, screening and combining are skipped |

The thresholds were measured on Jev's probabilities; a text model reports its own, so treat them
as a starting point there and check with the `judge` action.

Learned *intents* are exposed to every agent (routing to a local intent is what should happen).
The teaching *tools* reach only a conversation agent whose LLM API setting includes them, and
automatic learning only sees a turn because an agent was put in a pipeline — both are opt-ins. Generalisations, reviews and
judge calls are each capped per hour per entry, so a chatty afternoon cannot turn into a bill.

## Services

Everything the learner decides can be inspected and undone by hand. All return a response, so
call them with `return_response`.

| Service | Use |
|---|---|
| `observations` | what has been heard, what was decided, and the judge's verdict with probabilities |
| `route` | what the router would decide for one utterance, **without learning anything** |
| `judge` | ask the classification model directly; with no questions it reports which model it is and its spend |
| `review` | check what a learned intent will say. **Reports by default**; `apply: true` lets it fix |
| `audit` | re-judge **one** installed intent now, the next in line, changing nothing |
| `sweep` | mark recorded requests that are about the assistant rather than for the house |
| `consolidate` | move every intent's sentences and reply into one file, its own |
| `duplicates` | every entity read by more than one learned intent, with what each says |
| `merge` | fold chosen intents into one. **Reports unless `apply: true`** |
| `reconsider` | re-run the generaliser on one utterance, or on everything not learned |
| `learn` / `learn_intent` | install a phrase, or a whole intent, by hand |
| `set_response` | change what a learned intent says, without re-learning it |
| `rename_intent` | give a learned intent another name, and its file the matching one |
| `forget` / `forget_intent` | remove a learned phrasing, or an intent and its wording |
| `forget_all` | amnesia: every learned intent and every `learned-*.yaml` |
| `fold` | have the processing model write templates like `(press\|toggle) [the] D2 keyboard` for each group of learned phrasings — speech-to-text mishearings included. The matcher must answer every original phrasing with them, they may take nothing another group answers, and the classification model must agree that every sentence they add is the same request. Runs by itself after each learning; dry run unless `apply: true` |
| `combine` | fold the phrasings of an intent into one template per wording (`what's the weather [like] (today | tomorrow)`): the judge decides whether two ways of asking are the same, the matcher confirms both are still matched. Dry run unless `apply: true` |
| `export` | write `intent_script` stubs for every learned intent, so they keep answering after the integration is removed (writes to the `export_path` option; see [DESIGN.md](custom_components/irabelle_home_learning/DESIGN.md#removing-the-integration)) |
| `validate` | re-run the safety pass (parse, quarantine anything broken, reload) |
| `tools` | diagnostics: which teaching tools a given platform can see |

`review` reporting rather than acting is deliberate: asking the house to look at itself must not
be able to delete the thing you only wanted inspected.

## Entities

All sit on one device, so the integration page shows what it is doing without a service call.

| Entity | What it says |
|---|---|
| `sensor.irabelle_home_learning_learned_phrases` | how many phrasings are answered locally because they were learned, with the phrasings, intents, files and anything quarantined |
| `sensor.irabelle_home_learning_last_learned` | **what was learned most recently**, as the phrase itself, so the Activity log reads as a list |
| `sensor.irabelle_home_learning_heard_once` | phrases heard once in the last seven days, not yet decided — not a queue: after the window they are history |
| `sensor.irabelle_home_learning_judge_calls_today` | how many questions the judge has been asked today, with tokens and the last audit |
| `sensor.irabelle_home_learning_judge_cost_today` | what that has cost, estimated |
| `conversation.irabelle_home_learning` | the agent itself; its state is the time of the last turn it handled |

**Download diagnostics** returns everything at once: the exact templates and replies, every
phrasing, the last fifty requests with what was decided about each, what it cost, what drift it
found, and the judge's configuration with the API key redacted. Attach that to a bug report.

## Where the state lives

| What | Where |
|---|---|
| the sentences, the slot lists, the reply and the template behind it | `custom_sentences/<language>/learned-*.yaml` |
| observations, provenance, repeats, spend | `.storage/<domain>.<entry_id>` |
| anything refused at load | `irabelle_home_learning/quarantine/` — outside `custom_sentences/`, so the matcher never reads it |

**A learned intent is one file: the words and the reply.** Nothing else defines it — there is no
`intent_script` to add, no registry to update, and no second file to keep in step. The sentences,
the slot lists and the reply live together in that file, so "why does it say that" and "what makes
it trigger" are answered by opening it. Edit a reply there and that is what the house says.

Removing the integration keeps them — on unload each file is renamed to `learned-*.yaml.off` so its
phrases fall through to your agent instead of erroring (immediately: the matcher is reloaded, you do
not have to restart), and setting the integration up again adopts them back from the files
themselves. The store holds what a file cannot (the audit trail and the spend counter), so deleting
it costs you those and not a single answer. A reply that uses a slot
(`{day}`, the words you chose for a list of values) carries a one-line binding in front of it for
the same reason: the file is rendered both by this integration and by Home Assistant alone, and
both have to say the same thing.

Deleting the *folder* is the one case that is not self-healing: no unload runs, so the sentences
stay in front of the matcher with nothing to answer them, and each learned phrase returns an error
until the integration is put back. `irabelle_home_learning.export` covers it — one `action: []` per invented
intent, the one thing a sentence file cannot express, named beside the file that holds its words and
reply — and returns the include line to add:

```yaml
intent_script: !include_dir_merge_named configurations/intent_script/
```

It goes wherever your `configuration.yaml` already says registrations live — an inline
`intent_script:` block, the file an `!include` points at, `irabelle_home_learning.yaml` in a
`!include_dir_merge_named` directory, or one file per intent in a `!include_dir_named` one. With
nothing declared, it writes `intent_script.yaml` at the root and adds the include line for you.
Entries it writes name the sentence file that holds the intent, which is how they are updated and
removed again without touching yours. Once it exists it is kept current on every change, so it
never names last month's intents. Nothing changes while the integration is installed: its own
handler takes the name back at startup and answers.
Details in [DESIGN.md](custom_components/irabelle_home_learning/DESIGN.md#removing-the-integration).

Deleting the `learned-*.yaml` files and the store is a complete reset; `forget_all` does both.

## Troubleshooting

**A phrase should have been learned but was not.** Read `irabelle_home_learning.observations`: every row
carries `hits`, `state`, a human-readable `note`, and — when the judge ran — the verdict with
probabilities. `irabelle_home_learning.route` replays the decision for one utterance without changing
anything.

**A learned phrase is answered wrongly.** `irabelle_home_learning.review` reports what the reviewer would
change; `apply: true` lets it. `forget` removes one phrasing, `forget_intent` the whole thing,
and nothing is ever a ban — say the phrase again and it can be learned again.

**Nothing is being learned at all.** Check that `conversation.irabelle_home_learning` is the pipeline's agent
and its delegate exists, that `auto_learn` is on, and that `min_occurrences` has been reached.
Check the sensor's attributes for a quarantine.

**The house says something odd about a value it cannot find.** A template reads an entity that is
missing or has no usable state; `irabelle_home_learning.review` names it.

**Editing the Python needs a restart.** Home Assistant imports a custom component's modules only
at startup — and delete `__pycache__` alongside it if a change seems not to have taken.

**The house says a sentence, never an exception.** Whatever a delegate does — raises, goes out of
quota, has no network, or *returns* an error response instead of raising, which is what an agent
does when a tool API it needs has not finished registering — this agent asks Home Assistant's own
agent instead, and says one short sentence if even that fails. An internal error read aloud
("Error preparing context: Error getting LLM API [...]") is the one thing this agent must never
produce. Only `FAILED_TO_HANDLE` counts as a failure: `NO_INTENT_MATCH` and `NO_VALID_TARGETS`
are legitimate spoken answers ("I could not find a device called that"), and replacing those
would swap a useful answer for a shrug. A turn only the backup answered teaches nothing — it is
not a turn the real chain handled.

## Tests

```bash
custom_components/irabelle_home_learning/scripts/test.sh   # inside the Home Assistant container
python3 custom_components/irabelle_home_learning/tests/run.py  # or anywhere Home Assistant is installed
```

No pytest and no test harness: the suite builds a real `HomeAssistant` object and calls this
component's own code, which is the only way to ask the questions that matter here — what hassil
matches, what Jinja scopes, what the template engine renders. 219 tests over the sentence rules,
the slot rules, the template refusals, what is actually said, the model-facing tools and the file
lifecycle that keeps voice working.

Every test creates the entities it needs in its own registries and a throwaway config directory,
so the suite never reads your house and never has to clean up after itself. What each file pins
down — and the two things it deliberately does not — is in [DESIGN.md](custom_components/irabelle_home_learning/DESIGN.md#tests).

## Limits worth knowing

* **Learned templates read live state**, so they go stale if an entity is renamed or removed.
  Drift reports it; nothing repairs it.
* **No review queue** — a proposal is installed as soon as it validates. Observations are the
  audit trail; `review`, `forget` and `reconsider` are the undo.
* **The classification model is never on the local answer path**; a deployment
  with no internet still learns by binding and reviews through its `ai_task`.
* **A delegate can be unusable for the first minute after a restart**, while an asynchronously
  connecting integration finishes registering its tool APIs. Requests in that window get the
  backup answer.

The full list, with the reasoning, is in [DESIGN.md](custom_components/irabelle_home_learning/DESIGN.md#known-gaps).
