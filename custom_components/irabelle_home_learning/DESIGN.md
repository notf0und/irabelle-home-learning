# Irabelle Home Learning — design

Why this component is built the way it is, and the rules it holds itself to. For installing and
running it, read [README.md](README.md).

Most of what is here was paid for: every threshold is a number someone measured, and every rule
exists because breaking it broke local voice, which is worse than not learning at all — a wrong
local answer is instant, confident and permanent.

## How it works

**A turn that arrives here was missed.** Home Assistant asks the local sentence matcher before it
calls any agent, and a locally answered phrase never arrives at all — so the learner cannot
relearn what already works.

**The chain shares one chat log.** `async_get_chat_log` hands the *active* log back when the
conversation id matches, so every agent below writes into the log this agent was given and nothing
has to report anything. From a finished turn the learner reads the intent that ran and its slots,
every tool call with its arguments, the spoken answer, and what it acted on.

**Two flags are advertised on the chain's behalf**, or standing in front would change behaviour:
`supports_streaming` (the pipeline only streams into TTS when its own agent claims it) and
`CONTROL` (so state questions still reach the agent rather than being answered by the fallback
matcher).

**Failure is not spoken.** Whatever the delegate does — raises, goes out of quota, has no network,
or returns an error response instead of raising — the agent asks Home Assistant's own agent
instead, and says one short sentence if even that fails. Only `FAILED_TO_HANDLE` counts as a
failure: `NO_INTENT_MATCH` and `NO_VALID_TARGETS` are legitimate answers and replacing them would
swap a useful reply for a shrug. The delegate's result is returned *before* any bookkeeping runs,
and the bookkeeping is wrapped, so learning can never fail an answer.

Five stages, cheapest first. Money is only spent where a decision genuinely needs a model, and
never twice for the same decision:

| Stage | Cost | What it decides |
|---|---|---|
| 1. **Notice** | free | record the request, count sightings |
| 2. **Deterministic bind** | free | the turn ran an intent with reproducible slots → bind the words now |
| 3. **Route** *(optional judge)* | ~1 cheap call | which existing intent is this; nothing fits but it is answerable; or nothing at all |
| 4. **Screen** *(optional judge)* | ~1 cheap call | is this learning actually right |
| 5. **Review** | 1 text-model call | for a learned template only: read it back, fix it |

A learning that clears the screen never reaches the text model. What passes everything is written
to `custom_sentences/<lang>/learned-*.yaml` and registered as an intent.

**The screen is deliberately over-eager, and the thresholds are tight on purpose.** A false
positive costs one text-model call, off the hot path, after the answer was already given — a
fraction of a cent. A false negative leaves a wrong answer installed. The asymmetry runs one way,
so an unsure verdict escalates to the reviewer. Loosening a threshold to save calls is a bad
trade: on real templates the same questions scored anywhere from 0.05 to 0.98 between runs on a
plainly correct learning, because the state carries the live time and the entities' current
states.

## What it learns

**Binding** is deterministic: the words are attached to an intent that demonstrably answered them.
Slots are pinned only when the sentence itself names them — "turn off the office light" pins the
office light; "turn it off" is refused, because pinning what context supplied would make those
words mean that light for ever. Bindings are refused outright for intents an assist pipeline never
answers locally (`HassGetState`, `HassMediaSearchAndPlay`): the sentence would be written, then
ignored, and the phrase would come straight back. That list mirrors Home Assistant's own
local-fallback filter; keep it in step when you upgrade.

**Generalisation** creates a new intent with a template. It needs a text model, because only a
text model writes templates, and it waits for `min_occurrences` sightings (default 2) — the first
time a phrase is heard it may well be a one-off.

### Slots: one intent per word-class, not one per value

A request that only swaps one word for another value of the same class — any weekday, any room,
any person — must become **one** intent with a list slot. Otherwise the house learns "the weather
on Wednesday" today, "… on Tuesday" tomorrow, and still cannot answer Saturday; and the first
mention of a value is always the one that was never taught.

A slot is declared where Home Assistant's own sentence files declare one, and written into the
learned intent's file beside its sentences:

```yaml
intents:
  WeatherForecast:
    data:
    - sentences:
      - what's the weather on {day}
lists:
  day:
    values: [today, tomorrow, monday, tuesday, wednesday, thursday, friday, saturday, sunday]
```

Both paths support it. The **teaching tool** takes `slots` alongside the template and the
phrasings, e.g. `{"day": ["today", "tomorrow", ...]}`: every example phrasing containing one of a
slot's values is rewritten to use `{day}`, so seven weekday examples collapse onto one sentence
and the template may read `day`. The **generaliser** asks for the same thing in a plain line
(`day = today, tomorrow, ...`), and its prompt says to list *every* value the class can take,
including ones nobody has said yet.

Rules the learner enforces, because each one is a broken sentence file otherwise:

* a list of fewer than two values is refused — a slot matching only the literal word "day" is
  worse than being told the list is short;
* a `{name}` that no `lists:` block declares is refused: hassil raises `MissingListError` when the
  sentence is matched, which takes recognition down for **every** agent, not only this one;
* slots are per file, so consolidation carries a file's `lists:` block along with the sentences
  that use it;
* the render checks fill each slot with one of its values, so a slot template is proved and
  displayed on a real answer instead of on the blanks an undefined name produces.

A slot is a small closed word-class, **not** a free-form capture ("remind me to X"). A wildcard
list is the tool for swallowing arbitrary words, and it makes a sentence match very broadly — as
long as it is declared with `wildcard: true` and **no** `values` key, which is the shape
`normalise_slots` writes and `test_hassil_rules` pins down.

### One question, one answer

Wording should vary; the answer should not. Two intents for one question is how the house starts
answering the same thing two different ways depending on the words used.

* **At create time**, an installed intent whose template reads the same entity is offered to the
  judge, which is asked whether it already answers the question. If it does, the create is refused
  and the refusal names the intent to bind to. One judge question, only when an entity is shared,
  and no refusal at all when there is no judge — a guard against waste, not a reason to fail a
  learning.
* **Afterwards**, `irabelle_home_learning.duplicates` lists every entity read by more than one learned intent
  with what each says (no model involved), and `irabelle_home_learning.merge` folds chosen intents into one:
  phrasings move across, parameterised where the target has a slot for the word; the target's
  template, reply and slot list can be replaced in the same call; the rest are forgotten. `apply`
  defaults to **false**, so asking what would be merged cannot delete anything. `phrasings` covers
  wording *bound* to a built-in intent, which has no learned intent to merge and whose reply may
  not even accept the slot the words imply.

**Names follow what the intent does.** A file is named after its intent
(`learned-<intent-name>.yaml`), a binding's too, so a phrase bound to `HassTurnOn` lands in
`learned-hassturnon.yaml` beside every other phrasing for that intent. Never after the note that
happened to come with the write: a note is a sentence, and `learned-migrated.yaml` says nothing
about what is inside. When an intent stops describing what it answers, `rename_intent` moves the
spec, the file, the sentences (pinned slot values and all), the reply, the `lists:` block and
every phrase record together.

## The two models

Every **agent** (a config subentry, the thing a pipeline picks) names two models, and either can
be any model the house has — an AI Task entity or a conversation agent, picked by name:

* the **processing model** — writes and reviews templates (stage 5 and the generaliser). Rare,
  larger calls;
* the **classification model** — answers stages 3 and 4 and every other snap judgement
  (duplicates, combining, compound requests, the audit). Small, frequent calls.

The classification model is one of:

| Setting | What answers | Notes |
|---|---|---|
| Jev | the `jev.ask` action of the Jev integration | its key, daily budget and spend sensors |
| any `ai_task.*` entity | that model, through structured output | a small local model is plenty |
| any `conversation.*` agent | that agent, asked for a JSON reply | for models with no AI Task entity |
| Jev API | a direct POST to the endpoint below | URL, model and key in the agent |
| None | nothing | every judgement-backed step is skipped |

The agents share one hub — the sentence files, the store, the counters — because the files are
the house's: two hubs would write the same `custom_sentences/` and fight. A turn is learned with
the models of the agent that handled it (`agents.AgentView`); anything a person starts by hand
(a service, the audit) uses the first agent's.

**Every backend answers in the same shape**, so nothing downstream knows which one answered. A
text model is asked the same questions in one `ai_task.generate_data` call whose structure has,
per question, the chosen option and its probability plus the runner-up and its probability (a
`choice`), the probability the statement is true (a `noul`), or the level and its probability (a
`score`); those are read back into the same `choice` / `probabilities` / `confidence` / `noul`
fields. A runner-up is asked for because the margin, not the top answer, is what tells a decision
from a coin toss. The difference is calibration: a typed model's probabilities are measured, a
text model's are self-reported, and the thresholds below were set on Jev. With a text model they
are a starting point — `irabelle_home_learning.judge` asks the same question by hand, so they can
be re-measured on your own data.

A *typed decision model* takes one request carrying the state plus several typed questions,
returning answers your code branches on, for roughly a hundredth of a cent and a few hundred
milliseconds. Any endpoint that speaks this shape works — it is configured by URL, model name and
key, not by vendor:

```json
POST <endpoint>
{ "state": <anything>,
  "model": "<model>",
  "questions": {
    "which_intent": { "type": "choice", "instructions": "...", "criteria": {"<option>": "<what it means>"} },
    "repeatable":   { "type": "noul",   "instructions": "...", "criteria": {"true": "...", "false": "..."} },
    "scores":       { "type": "score",  "instructions": "...", "criteria": ["low", "high"] }
  } }
```

`choice` returns the chosen option plus `probabilities` and `confidence`; `noul` returns a 0–1
probability that the statement is true; `score` returns a weighted value. Every stage degrades to
its old behaviour when the judge is off or unreachable.

Three lessons, all measured:

* **Ask every question in one request.** Questions are evaluated in parallel against the same
  state, so one more costs only its own tokens and almost no latency. One question must be one
  snap judgement; a broad question hides several judgements behind one answer.
* **Question wording is code.** The `true`/`false` descriptions are part of the prompt, so they
  change the answer. A question scored 0.55 with its labels and 0.86 without, reproducibly on
  identical state, and was replaced by one phrased around the failure it was looking for.
* **A near-tie is not a decision.** `probabilities` is kept whole precisely so a choice made 0.40
  against 0.35 can be treated as unknown. Every threshold here is a starting point to be replaced
  by measured accuracy — plot confidence against accuracy on your own data.

### What the screen asks

Seven judgements in one request, on every learning including bindings to built-in intents (which
nothing else checks at all):

| Question | Type | Fault it catches |
|---|---|---|
| `answers_the_request` | noul | the answer answers a different question |
| `same_scope` | noul | a phrasing is answered with the wrong device, room, property or scope |
| `reads_a_real_entity` | noul | the template reads something with no usable state |
| `reads_the_right_entity` | noul | it reads a *real* entity that is about something else entirely |
| `answers_the_speaker` | noul | a "my …" phrasing answered identically for everyone |
| `time_looks_right` | noul | a time twelve hours out, or a placeholder |
| `what_is_wrong` | choice | which of the above it is, including "nothing" |

**State only what the questions need.** Sending the whole entity catalogue made the judge answer
*0.43* to "does this answer the request" for a template that answered it exactly. The screen sends
the request, the answer, the template's own render and only the entities it reads — plus, for the
entity question, the *other* entities whose name also matches the request, and the attribute names
each entity has. Without the attributes, a template reading an entity's `temperature` attribute is
judged from `home = rainy` and the judge prefers whichever entity is literally *named*
temperature; without the alternatives, a real but wrong entity cannot be noticed at all. Two
entities holding the same quantity are both correct, so the question asks whether the entity is
the right *kind* of thing.

**Render a template the way the house answers it.** A template branching on who is asking shows
its fallback when nobody is speaking, so a single unrendered judgement calls every speaker-aware
intent broken. Each is rendered once per `person` entity and once for nobody; a slotted template
is rendered **both** with the slot filled and without it, because those are two different runs of
one intent ("where is Robin's phone" carries the slot, "where is my phone" does not) — showing
only the filled one made every person's answer the sample person's.

### When the judge says a learning is wrong

The learning goes to the text model with the judge's reasons attached. A `fix` verdict can replace
the template and/or remove phrasings; when the last phrasing goes the intent goes with it, and
**the request is handed back to the generaliser with the reason written down**, because deleting a
wrong local answer without replacing it sends the request to the cloud model for ever. One attempt
only, so a replacement that is itself removed does not spawn another.

**A replacement is installed only when it is no worse than what it replaces.** The screen is noisy
at the margin — a correct answer scored `reads_like_a_sentence=0.52` on one pass and `ok` on the
next — and an auto-applied rewrite is how a working forecast was once replaced by a template that
always said "tomorrow". The candidate is screened exactly as the current one was; if it does not
come back cleaner, it is reported with its own screen and left alone.

### The teaching tools

What a model may call, in the order a change usually needs them:

| Tool | For |
|---|---|
| `ListIntents` | what exists, and the exact phrasings already learned for it |
| `GetIntent` | **read one intent in full** — its template, the reply to edit, what it renders now, its slots, its phrasings, the entities it reads. A request like "improve this template" needs this first, and a learned template is in this integration's store, not in a configuration file a model can open |
| `ListEntities` | the real entity ids, names, states and attribute names — copy one instead of inventing it |
| `SetResponse` | replace a reply: wording *and* formatting (units, rounding, a bearing code turned into a word, joining two sentences). Send the whole text back |
| `LearnIntent` / `LearnUtterance` | a new intent, or new wording for an existing one |
| `ForgetUtterance` / `ForgetIntent` | undo |

## Who decides what

Three different things decide what a learned intent becomes, and keeping them apart is what
makes the result trustworthy:

| decides | who | why not the others |
|---|---|---|
| the **wording** and the **reply** | a text model | it is writing, and no score can produce it |
| whether two ways of asking are **the same request**, whether a request is one the house answers, which intent it is | the judge, typed (`choice`, `noul`) | a judgement over a fixed set of options, which is what a typed model is for, and it comes back with a number that can be recorded and argued with |
| whether the **matcher accepts** a sentence | the matcher | not a judgement at all: `hassil` is asked, with a concrete value in every slot, and what it refuses is never written |

The middle row is the one that gets skipped by accident. `irabelle_home_learning.combine` is where it earns
its keep: folding `what's the weather today` and `what's the weather tomorrow` into
`what's the weather (today | tomorrow)` is arithmetic, and folding `what's the weather` and
`what's the weather like` into `what's the weather [like]` is arithmetic too - but `(is | was)`
is just as symmetrical and means two different things. So the *candidate* is arithmetic, the
judge scores each one ("do these ask for the same thing?"), and the matcher has the veto on the
result. One judge request covers every candidate in a file, and the answer to each is a number
that lands in the report, so a wrong fold can be read back and argued with rather than guessed
at.

That is also why the component never asks a model to decide whether a phrase *works*: a
`sentence` that the recogniser does not accept fails the whole pipeline for that phrase, and a
model's opinion is not evidence about a parser.

## Speaker awareness

A hand-written `intent_script` template is rendered with **slots only**, so it cannot tell whether
a request came from a signed-in user in the chat or from a satellite. A learned intent gets the
`Intent` object and is handed:

| Variable | Meaning |
|---|---|
| `user_name`, `user_id` | the signed-in user (chat in Home Assistant, or the app) |
| `person_entity_id`, `person_name`, `person_state` | the matching `person` entity |
| `device_id`, `satellite_id`, `user_area`, `user_floor` | where the request came from |
| `speaker_certain`, `speaker_confidence` | whether the speaker is known, and the voice match's own score |

All are always defined (empty when unknown), and anything else that names the speaker — a
voice-biometrics sensor, say — can be used the same way. Two guards keep it honest:

* a template that follows the speaker must not be claimed by a phrasing naming a *specific*
  device: "where is my phone" and "where is my <brand> phone" are different requests, and the
  second is wrong for everyone who does not own that brand;
* a template that reads an entity named after one person and never consults the speaker answers
  the wrong person half the time, so `speaker_scope_problem()` refuses that combination outright.

## Entity ids

**A friendly name is not an entity id.** A model shown `Home Home Forecast` writes
`weather.home_home_forecast`, which does not exist. Every write is checked against the entity
registry and refused when an id is not there — `states()` renders the word *unknown* and the house
would say it out loud — and three things make that a correction rather than a wall:

* `irabelle_home_learning__ListEntities` returns the real ids, with friendly names, current states and
  attribute names, for the entities a request's words point at, so a model copies one instead of
  inventing it;
* the refusal names the nearest real id (`weather.home_home_forecast -> weather.home`, which is
  what a friendly name becomes when it is mistaken for an id);
* every successful write returns `reads`: the ids it ended up reading, with their states, in the
  same result the model sees.

**An attribute the entity does not have is reported, not refused.** `state_attr(x, 'forecast')`
where the attribute is `forecast_daily` returns nothing **without raising**, so a reply written
around it degrades silently — a model asked for a more detailed weather answer read the wrong name
and the house quietly answered with the current conditions. Reading an optional attribute is
legitimate (a guarded `forecast_hourly | or []`), so the finding is a warning on the result, in
the reviewer's evidence, and in drift.

## Making a hardware-named entity legible

Entities that carry state worth answering from are usually named after hardware: a tracker's area
sensor is `<mac>_area_last_seen`, and four of them produce four identical lines in any list a model
reads, so it writes a template that reads the wrong one.

**Add an alias; do not rename.** A registry `name` replaces what the entity is called everywhere —
dashboard, history, automations — and takes away the name its owner recognises. An alias is
additional, it is already what the matcher and the exposure list accept, and it is what this
component's lists show:

```
sensor.<mac>_area_last_seen (Area Last Seen aka "Person's watch" = Office, Person's)
```

A registry name also **deletes the device prefix, and templates can depend on it**. Home Assistant
composes the friendly name of an entity with `has_entity_name` as `"<device name> <entity name>"`;
overriding the registry name replaces the whole string. A sensor whose entity name is
`Area Last Seen` normally reads `Bedroom Tracker Area Last Seen`, which is where the device words
come from for anything that derives a label from the name. One template did exactly that
(`name | replace(' Area Last Seen', '')`) and after a rename answered *"your Area Last Seen was
last seen in the Bedroom"*. An alias adds words without taking the composed name away.

Nothing in the component depends on a registry name: ownership is derived from the `person` →
`device_tracker` → device chain, and aliases are read from the registry on the fly.

## Where a reply is written

A `custom_sentences/` file carries **triggers**, and Home Assistant keeps the **reply** for an
intent in the same file, under `responses:`:

```yaml
intents:
  ExampleIntent:
    data:
      - sentences: ["where is my phone at"]
responses:
  intents:
    ExampleIntent:
      default: "Your phone was last seen in the {{ states('sensor.phone_area_last_seen') }}."
```

That dictionary is merged from every sentence file, and the user's own files are merged *after*
the built-in ones, so a `responses:` block can both supply a reply and override one. The key is
chosen from what the handler produced — `default`, `one`, `any`, `no_match` — and the template is
rendered with `slots`, `state` and `query` only. Two consequences:

* **A reply template is only used when the handler says nothing.** A handler that sets its own
  speech wins, and a `set`/`math` response type is ignored entirely. A learned intent renders its
  own reply, so it reads `responses:` from its file, which is authoritative, and falls back to the
  stored template for intents learned before replies lived there.
* **A reply template gets no speaker.** `user_name`, `person_name` and the rest are handler
  context, not template context — so a learned intent that branches on who is asking *must* render
  its own reply, which is why this component does.

### One reply, two renderers

The two bullets above are the whole reason a reply needs care. The same text in the same file is
rendered by two different things: this component's handler, which defines every slot as a variable
of its own *and* as `slots`; and, once this component is gone, Home Assistant itself, which defines
`slots` and nothing else. A learned reply is inside the file precisely so it survives the second
case — that is what the exported `intent_script` stubs lean on — so one of those renderers has to
bend, and it must be the one still installed.

Written into the reply is therefore a binding per slot the reply reads bare:

```jinja
{% set day = slots.day or '' %}{% set w = (day | lower) if day is defined else 'today' %}…
```

`slots.day or ''` is the value the handler was using all along, so nothing about the answer
changes where the component is installed, and after it is gone the day is still the day that was
asked about. This is not cosmetic: **an unbindable name fails silently in the other renderer.** A
weather reply asked about tomorrow rendered *today's* forecast, took the wrong branch, and answered
confidently — the failure a reader would never attribute to a missing variable. Hence:

* the binding is written at the single write path (`write_response`), so nothing can install a
  reply without it;
* only the slots the file itself declares are bound — the sentence's `{day}`, a pinned
  `slots: {day: tomorrow}`, or a `lists:` key — so a typo is still reported as reading nothing
  rather than papered over;
* the readers strip it (`file_responses`, `adopt_disarmed`), so the handler, the prompts, the
  screen and a person editing the file all work on the same readable text;
* `backfill_responses` rewrites an unbound reply once at startup, which is how the replies
  installed before this existed were migrated;
* and the test asserts the property directly — the stored text rendered with `slots` and *nothing
  else* must say what the handler says.

This component writes every learned intent's reply into its file (`backfill_responses` fills in the
ones learned before that existed), and `irabelle_home_learning.set_response` — or the `irabelle_home_learning__SetResponse`
teaching tool, in conversation — changes it. A reply set that way survives a template correction; a
reply that is just the template moves with it.

**A sentence whose intent nothing answers breaks *that phrase*.** Measured: the matcher reports a
match, no handler runs, and the pipeline answers `intent-failed: Unexpected error during intent
recognition` for it — the person hears an error instead of an answer, while every other local
phrase keeps answering in milliseconds. They appear when an intent is forgotten while a phrasing
bound to it lives in another file, and they are why an unloaded integration must not leave its
sentences in place (see *Removing the integration* below). At startup, after every integration has
registered its handlers, this component removes sentences for intents nothing answers from the
files it writes, and *reports* the ones in files it does not own, because reading those is allowed
and rewriting them is not.

### What is checked about the wording, mechanically

A reply is rendered and read back before it is installed. `reply_problems` is the part that
needs no model, and each rule in it came from a real answer it should have caught: a doubled
space (a value did not render), an unrendered `{{ … }}`, a thing said twice, a word repeated, and
two rules about the joins a rewrite produces — two punctuation marks in a row ("…in the
afternoon., the high…") and a word capitalised after the joining comma *when the text itself
also writes that word in lowercase* (so "Showers" beside "showers" is caught, and "Sam" beside
nothing is left alone, because the evidence has to be in the text rather than in a name list).

The punctuation join is not merely reported: `tidy_punctuation` fixes it on write. It is a typo,
not a wording choice — asking the same model that had just made the mistake to polish it
answered "no rewrite offered" often enough that the sloppy text was installed anyway.

## What it checks about itself

A learning is checked when it is made; none of that helps a week later, when an entity has been
renamed and the template renders the word *unknown* into a spoken sentence. Four things watch what
is already installed, and all of them report rather than repair: a component that rewrites the
house on a timer is worse than one that names what is wrong.

| | When | What it does |
|---|---|---|
| **Drift** | at startup, after every integration has loaded | runs the checks a new learning gets over every installed intent: entities that exist, a template that renders, attributes the entity does not have, a reply that reads like a sentence. `irabelle_home_learning.validate` lists it. |
| **The house, not the assistant** | before anything is recorded | one question on sentences that look like an instruction *about* the assistant ("next time add the device name"), which are never said twice. A sentence that does not look like one is never asked about, so the gate cannot lose a real request. |
| **Repeats** | when a phrase the house already answers locally arrives at the agent anyway | counts it against that intent, and from the second time asks the reviewer to look. That is the only negative signal a person gives without being asked for one. |
| **Audit** | every few hours, one intent at a time, rotating | screens it again and logs anything that no longer passes. `irabelle_home_learning.audit` does one now. |

Deferring drift to the end of startup is not a detail: run during setup, entities from
integrations that load later are not in `hass.states` yet, so every template that reads one looks
like it reads something that does not exist and the check reports a working house as broken.

## Cost, in the order you pay it

1. **Nothing** for a phrase the matcher already answers.
2. **Nothing** for a turn seen fewer than `min_occurrences` times.
3. **One cheap judge call** to route, and one more to screen a new learning, when the judge is on.
   Both bounded per hour.
4. **One text-model call** to write a template and one to review one, both bounded, and the review
   is skipped entirely when the screen says the learning is clean.
5. **Nothing, for ever**, once the phrase is local — which is the whole point.

As a sense of scale from a real installation: a clean learning was screened in 0.33 s with no
text-model call, a wrong one cost one text-model call and was corrected, and a learned phrase
answers in single-digit milliseconds where the cloud model took 2–3 seconds.

## Tests

```bash
scripts/test.sh
```

No pytest and no test harness: the suite builds a real `HomeAssistant` object and calls this
component's own code, which is the only way to ask the questions that matter here — what hassil
matches, what Jinja scopes, what the template engine renders. A mock would answer the wrong
question, and every fact below was learned the hard way.

Each test owns its world: its own `HomeAssistant`, a throwaway config directory, and entities it
creates in the registries (`tests/fixtures.py`). Nothing is cleaned up afterwards because
nothing outlives the test — better than a teardown step that can be forgotten, and it means the
suite can run against a live installation without touching it.

| File | Pins down |
|---|---|
| `test_sentences.py` | punctuation stripped on write, apostrophes kept, follow-up words, filenames, the wording checks |
| `test_hassil_rules.py` | against hassil itself: a stored `?` never matches, an undeclared `{slot}` raises at match time, a wildcard list dies if it carries a `values` key |
| `test_slots.py` | what a model may declare, and how example phrasings fold onto `{day}` |
| `test_templates.py` | every refusal a template gets, the missing-attribute warning, the id suggestion |
| `test_handler.py` | what is actually *said*: a matched slot reaching the template, an explicit reply winning, the speaker variables, a reply edited in the file |
| `test_intent_files.py` | where sentences and replies are written, consolidation, rename, merge, and the pipelines that break when a file is left behind |
| `test_hygiene.py` | the startup repair of an unmatchable sentence, orphan pruning, and that forgetting removes the handler, the sentences and the record together |
| `test_tools.py` | the model-facing surface: aliases and ownership as shown, `ListEntities` filtering and the teaching gate |
| `test_screen.py` | the state the judge is shown — both slot renders, the alternatives, the attribute names — and the reasons derived from its answers |

Two things are deliberately *not* unit-tested, and `scripts/` covers them against a live house
instead: the screen's actual verdicts (which need a real judge, and are calibration rather than
behaviour) and the device-name half of ownership derivation (`device_registry` insists on a real
`ConfigEntry`, which does not exist outside a bootstrapped Home Assistant — faking enough of it
to satisfy HA is more brittle than the code it would test). The person → tracker link *is*
tested.


## Invariants

The rest of the rules this component holds itself to; the per-feature ones live with their feature
above (entity ids, attributes, slots, duplicates, replies). Most exist because breaking them broke
local voice.

* **A bad write must never break local voice.** One unparseable file in `custom_sentences/` takes
  down sentence matching for every agent, so every file is parsed and a broken one moved to
  `irabelle_home_learning/quarantine/` *before* any reload — at the top of the configuration folder, not inside
  `custom_sentences/`, because the loader reads that tree recursively (`rglob("*.yaml")`) and would
  read the quarantined file straight back in.
* **A sentence stored with sentence punctuation matches nothing** — hassil treats `, ; ? !` as
  separators and Home Assistant strips them from the *input* but not from the stored sentence, so
  the two can never meet. Stripped on write; existing files repaired at startup.
  (`test_hassil_rules.Punctuation` is this rule, run against hassil.)
* **Writing a sentence is not the same as the matcher accepting it.** The real recogniser is asked
  afterwards and the result recorded (`verified: ok | unmatched | unknown`), with a warning for any
  phrase that will keep reaching the agent.
* **A follow-up is not a phrasing.** "when?" means whatever the previous turn said, so a phrase
  made *only* of follow-up words is refused. Person words are exempt, so "who am i" is learnable.
* **Never re-target.** A phrase bound to a different intent is refused, and a phrase the matcher
  already answers is skipped even from a hand-written file; colliding variants are dropped rather
  than aborting the write.
* **Prompts use `string.Template`, never `.format()`** — prompts contain Jinja, and `.format` reads
  `{% set x = ... %}` as a field.
* **Only files prefixed `learned-` are ever written or deleted.** Hand-written sentence files are
  read, never modified. Forgetting compares normalised forms, or the sentence stays behind while
  its only record is dropped.
* **Removal must not leave a sentence nothing answers.** An unload disarms its files
  (`learned-*.yaml.off`) unless something else still registers their intents, and the deletion that
  runs no code is covered by `export`'s `intent_script` stubs. In both cases the reply stays in the
  sentence file, so that is the only copy that has to survive.
* **`export_path` is user input, so it is checked.** A path that is absolute or climbs out of the
  configuration folder falls back to the default rather than writing outside `/config`.

## Removing the integration

A learned intent is a sentence plus a handler, and the two live in different places. The sentence
lives in `custom_sentences/<language>/learned-*.yaml` — a **documented Home Assistant location**,
read by the matcher whether or not any component wrote it — and the handler only exists while this
component is set up. That asymmetry is the whole removal problem: sentences without a handler make
their phrases answer with an error (see *A sentence whose intent nothing answers*).

Two mechanisms answer for it, because there are two ways to leave:

* **Unload** — removing the integration, disabling it, reloading it: `async_unload_entry` renames
  every file it owns to `learned-*.yaml.off`, which the matcher ignores, so its phrases fall through
  to the agent instead of erroring. Nothing is lost but the registration: setting the integration up
  again adopts the files back *from the files themselves* and not from the store, so it also
  survives deleting `.storage`. The rename is skipped for a file whose intent something else still
  answers — an installed `intent_script` stub, say — because that sentence is not orphaned and its
  reply is still wanted. **The rename is followed by `conversation.reload`, and that is not
  optional**: the matcher caches what it read, so without it the sentence still matches from memory
  while the handler that answered it has just been released, and the phrase answers
  `intent-failed` — measured on this house, before the reload was added. The same reload follows
  arming at setup, which is why a re-enabled integration answers locally again without a restart.
* **Deleting the folder** — uninstalling, so no unload runs: no code of ours runs at all and the
  sentences stay in front of the matcher with nothing behind them. `irabelle_home_learning.export` is the
  preparation: one `action: []` stub per learned intent, which registers the same intent through
  Home Assistant's own `intent_script` integration, with the reply coming from the file's  `responses:` block exactly as it does now. Run it again when the set of intents changes.

### Why the registration cannot live in the sentence file

A learned intent is one file: the phrasings and the reply together, which is what makes the folder
the thing a person reads. The registration is a second file for a reason that is Home Assistant's
and not this component's — **the two loaders want different files.** A sentence file's `language:`
and `intents:` keys are read by the conversation integration, and the same file handed to
`intent_script` is rejected outright (measured: *"expected a mapping at
`intent_script.language`"*), because `intent_script` is intent name to config and has no place for
sentences. The reverse is true as well: `custom_sentences` cannot register a handler, and a phrase
whose intent nothing answers is an error for that phrase.

Two consequences worth keeping:

* The stub carries a comment naming the sentence file that holds the intent, so the exported file
  is also its index — the one thing it can do for whoever reads it, a person or an assistant with
  access to the files, is say where to look. The body is written by hand for that reason, and
  parsed back before it is written: it is included from `configuration.yaml`, where a syntax error
  would stop Home Assistant from starting.
* Co-locating the stubs *inside* `custom_sentences/<language>/` does work — measured, the matcher
  tolerates a file with no `intents:` key, and `export_path` can point there — but it puts a
  non-sentence file in the one tree that must never fail to load. The invariant is worth more than
  the convenience: a registration file outside that tree can break only the phrases it registers.

The stubs are **opt-in, and this house includes them** (`configurations/intent_script/irabelle_home_learning.yaml`,
from the `export_path` option, included with `!include_dir_merge_named`). It is worth being clear
about what they are, because it is the part that looks redundant and is not: an intent is one file —
the words and the reply — and a `responses:` block supplies *wording*, not an intent. A phrase is
answered by a handler looked up by intent name, so an invented name needs something to register it,
and `intent_script` with an empty `action` is exactly that: a handler that runs, says nothing, and
lets Home Assistant speak the reply from the sentence file. This is the arrangement a house that
learns its own question types has *always* needed — before this component existed it was a
hand-written `intent_script.yaml` with one entry per invented intent — and `export` writes that same
file, with a pointer to each intent's own file beside its entry. An install that never runs `export`
has no such file and gets no writes; one that does gets it kept current, so it cannot go stale into
registering last month's set of intents.

**Where the registration goes is the house's choice, so it is read, not imposed.** `export`
parses the `intent_script:` line in `configuration.yaml` and writes to whatever that already says:

| `configuration.yaml` says | the stubs go |
|---|---|
| `intent_script:` with entries under it | into that block, in `configuration.yaml` itself |
| `intent_script: !include foo.yaml` | into `foo.yaml`, beside whatever it already holds |
| `intent_script: !include_dir_merge_named <dir>` | `<dir>/irabelle_home_learning.yaml` |
| `intent_script: !include_dir_named <dir>` | one file per intent, `<dir>/<Intent>.yaml` |
| nothing at all | `intent_script.yaml` at the root, and the include line added |

This repository declares a merge directory (`configurations/intent_script/`), so its option only
names the file inside it. The `export_path` option is the fallback: the file name used inside a
directory the house named, and the path used when the house has said nothing. It refuses a path
outside the configuration folder, because it is user input.

**An entry this component writes carries a comment naming the sentence file that holds it, and
that comment is its signature.** Updating an entry, and removing the one for an intent that was
forgotten, works by finding that comment again — so a hand-written entry in the same file is
never touched, and never deleted: a house that keeps its own intents beside ours keeps them
exactly as they are. For the same reason an existing `!include_dir_named` gets one file per
intent (the loader names the intent after the file), and only marked files are ever removed.

Two rules keep writing into somebody else's file honest. `configuration.yaml` is the one file
whose syntax error stops Home Assistant from starting, so after any edit it is re-loaded the way
Home Assistant loads it — includes, secrets and all — and put back unchanged if that fails, with
the previous text kept under `irabelle_home_learning/backups/`. And the whole thing is **opt-in**: a house
that declares `intent_script` has already said where registrations go, so ours is kept in step
with it, but a house that says nothing gets nothing until `export` is run once — a learning must
not edit the main configuration file as a side effect of itself.

Installed stubs are registrations, so while the file is included a sentence bound to an intent that
no longer exists counts as answered and the orphan sweep leaves it alone. Forgetting an intent
removes its own sentences and its reply, so what is left is a stub for a phrase nothing says; run
`export` again and the file matches the house.

What the stubs cannot bring back is the speaker: a reply that asks `person_name`/`user_name` is
rendered with slots and state only, so it answers with its fallback for everyone. A reply that asks
about a **slot** is a different case and is *not* lost — the binding written into it (see *One
reply, two renderers*) is what makes "what's the weather tomorrow" answer about tomorrow rather
than today once only Home Assistant is left to render it. Everything else — the words, the reply,
the entity ids inside it — is in the files and keeps working.

Measured, with the component's folder deleted and Home Assistant restarted (so no unload ran and
the sentence files stayed armed): the four learned phrases tried answered locally in 7–236 ms from
their own files, the two weather replies matched the answers the component gives word for word, and
only the `who am i` reply fell back — which is the one thing that is supposed to.

## Notes for whoever works on this next

* **Run the tests before and after a change**: `scripts/test.sh`. They pin the traps written down
  here, so breaking one is breaking something that was paid for. Where a rule here has no test, add
  the test rather than another paragraph.
* **Measure before tuning.** Every threshold is a number someone picked, with the measurement that
  produced it written beside it; change the comment with the number. A judge answer is not a
  verdict — record `probabilities` and `margin` with the observation and move thresholds on
  evidence, and treat a change to a screen question as a behaviour change: re-run
  `scripts/probe.py` over the fixed cases first.
* **The two model paths fail differently.** A text model with a JSON structure bolted on will
  happily return a plausible verdict with no basis; a typed decision model returns a number and a
  distribution, wrong but auditable after the fact. Neither replaces the deterministic checks,
  which cost nothing and never lie.
* **`scripts/` is how you check a claim against the live house.** `ask.py` drives a real pipeline
  (the local matcher runs first, so it is the only way to tell "learned" from "answered by a
  model"), `traces.py` reads the last runs, `probe.py` replays the screen's cases, `service.py`
  calls a service and shows the real error. See `scripts/README.md` — they found several of the
  bugs written down here.

## Known gaps

1. **Learned templates read live state**, so they go stale if an entity is renamed or removed:
   drift reports it, nothing repairs it.
2. **No review queue** — a proposal is installed as soon as it validates. Observations are the
   audit trail; `review`, `forget` and `reconsider` are the undo.
3. **A deletion gets one replacement attempt**, not a guarantee: if the generaliser cannot do
   better, the request goes back to the cloud model.
4. **Action intents may come back without speech** — some assistants route actions by running the
   handler directly and only fill in speech for state questions. Not caused by this integration,
   but binding the wording fixes it for that phrasing.
5. **`forget_all` is total**: everything this integration ever learned, including bindings for
   hand-written intents. It never touches hand-written files.
6. **The classification model is never on the local answer path**: it is asked after the answer was
   given. A deployment with no internet can classify with a local `ai_task` model, or with none
   and still learn by binding.
7. **A slot can only be declared when the intent is made** — an intent learned as one value per
   phrasing keeps those fixed sentences, so re-teach it or forget it and learn it again.
8. **A duplicate is refused, not merged.** `duplicates` finds pairs that predate the check, and
   `merge` folds them, but only when asked: which intent owns the question and which reply wins is
   a judgement about your house. Two intents answering the same question while reading *different*
   entities are not refused either — there is no shared entity to offer the judge.
9. **A delegate can be unusable for the first minute after a restart**, while an asynchronously
   connecting integration finishes registering its tool APIs; requests in that window get the
   backup answer. Nothing this component can fix from below — but it no longer speaks the failure
   out loud, which is the part that made it look broken.
