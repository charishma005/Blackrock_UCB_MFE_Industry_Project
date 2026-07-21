# The analyst layer — design record

Status: **built and run.** Validated on `inflation` across Haiku/Sonnet/Opus (§8–12),
then generalized: the config-driven path now carries **seven personas** (inflation,
labor_tightness, curve_slope, term_premium, balance_sheet, financial_conditions,
inflation_expectations), and the repo has been cut down to just this upstream
(data → agent → report) for handoff — the legacy deterministic analyst stack and the
old end-to-end demo were removed, `DriverView` in `contracts.py` is the seam.

This documents the data → analyst layer as rebuilt on 2026-07-19: the decisions and
why they were taken, the module map, every class and function, and the exact
pipeline a run executes. The layers above it (PM, ensemble, risk) are downstream and
owned by other teammates.

---

## 1. Why this exists

The previous analyst path could not test the thing the layered design is for.
Rendering the prompts the analysts actually sent showed two structural defects:

- **The numeric arm contained its own answer.** `form_view()` passed the finished
  `DriverView` — `direction`, `conviction`, and the deterministic `reasoning` — into
  the prompt and asked the model to "refine" it. The only evidence present was a
  single scalar; no time series ever reached the model. Measured agreement with the
  formula was `own_corr` 0.965 on inflation. "The LLM added nothing" was what that
  prompt was built to produce.
- **The text arm sent every analyst identical bytes.** The prompt was the FOMC
  statement alone, the analyst's own series dropped entirely, and the only
  difference across seven prompts was the driver name on line one. Average pairwise
  correlation rose 0.221 → 0.339 and faithfulness went negative for inflation
  (−0.171) and term premium (−0.050): those analysts tracked other drivers more
  closely than their own.

Both are fixed by changing what crosses the first boundary. An analyst now receives
**two channels about its own driver and nothing else** — engineered measurements
(*what moved*) and driver-specific policy language (*why it moved*) — and writes a
**report**. Because PMs are LLMs too, the report is what crosses the layer boundary.

---

## 2. Decisions taken, and why

| # | Decision | Rationale |
|---|---|---|
| 1 | **Real data only; no synthetic path in this pipeline** | The structure is being proven against history a person would actually have read. Synthetic regimes can't surface the problems that matter (templated text, release lags, irregular cadence). |
| 2 | **FRED from local CSVs, not the API** | `FRED_API_KEY` isn't required — the sibling FOMC corpus ships the raw series. Removes a setup step and makes runs reproducible. Series are shifted to their release date on load, reusing the same lag table as the API path so the two agree. |
| 3 | **Core PCE alongside headline CPI** | Headline moves with energy and food; core reads persistence. More importantly the Committee's 2% objective *is* stated in PCE terms, so the feature channel and the text channel refer to the same measure. Previously the analyst saw headline CPI only and could not make the distinction at all. |
| 4 | **Closed vocabulary of feature operations** | Makes "measurements, never signals" structural rather than a convention. No operation fits a parameter, standardizes on a full sample, or scores a direction, so a spec *cannot express* a forecast. A feature named `inflation_momentum_signal` is the old anchoring bug wearing a feature's clothes; this is what prevents it. |
| 5 | **Feature spec as config, not code** | An analyst is now fully described by its persona file. The seven hand-written analyst classes exist only to implement `read()`; once measurement is spec'd and judgment is the model's, there is nothing left to subclass. Adding an analyst becomes a YAML file. |
| 6 | **Date scrubbing in the selector base class** | An FOMC statement opens with its release date and closes with an implementation note carrying it again. Scrubbing lives in the base so no arm — including the control — can forget it. This removes the cheapest tell; it does **not** make the input date-blind (a 9.1% CPI print identifies its quarter regardless). |
| 7 | **Boundary-aware cue matching** | Substring matching routed `"4-1/2 percent"` into the inflation analyst via the `"2 percent"` cue — another driver's data, and a phrase that pins the meeting to Dec 2022. Cues now compile to regex with a negative lookbehind. |
| 8 | **Strict partition in v1, whole-document control retained** | Cross-driver causal language is excluded for now. The control arm is kept precisely so the cost of that strictness stays measurable rather than assumed. |
| 9 | **Report plus a structured header** | The report is the contract to an LLM PM. `direction`/`conviction`/`level` survive because every existing diagnostic operates on those numeric streams — dropping them would cost the whole measurement apparatus. All new fields are optional, so no existing caller breaks. |
| 10 | **Failures abstain; they never fall back to a benchmark** | Substituting a rule's answer on failure would mix the comparison into the thing being compared. A failure emits `degraded=True` so it is visible and excludable. |
| 11 | **`key_evidence` validated against the feature names** | Because the exact set handed over is known, grounding becomes a mechanical check rather than a lexicon guess. |
| 12 | **Report written before direction** | Field order in the output contract makes the call follow from the analysis, rather than the analysis rationalising a call already made. |
| 13 | **Benchmarks deliberately out of scope** | This step builds the LLM pipeline only. Benchmarks enter as a separate `ViewSource` implementation later, consuming the same `FeatureSet` so the comparison is apples-to-apples by type. |
| 14 | **Nothing removed** | `SingleDriverAnalyst` and the seven subclasses still run, so `run_diagnostics`, `test_layered`, and the existing reports stay reproducible while the new path is proven. |

---

## 3. Module map

```
src/data/
  markets.py         FRED API + PUBLICATION_LAG_DAYS (the release-lag table)
  fred_local.py      real FRED series from vendored CSVs (data/fred/), release-dated
  fomc_text.py       point-in-time FOMC corpus  (pair_as_of)

src/layered/
  contracts.py       DriverView (the seam) + StrategyTrade + FundAllocation, FeatureSet
  timeline.py        AsOf — the single no-lookahead choke point
  features/
    ops.py           the closed vocabulary of measurement operations
    spec.py          FeatureSpec / FeatureDef, parsed from persona YAML
    engine.py        FeatureEngine — evaluates a spec through the AsOf gate
  text/
    selector.py      TextSelector interface, TextContext, date scrubbing
    cue.py           CueSelector — driver-partitioned passages + diff
    whole.py         WholeDocumentSelector — un-partitioned control
  analysts/
    llm_analyst.py   LLMAnalyst — the specialist that reads evidence, writes a report
    carry_forward.py CarryForward — re-emit unchanged views, call only on new evidence
    build.py         shared runner harness (build analyst, preflight, audit)
    personas/
      _TEMPLATE.yaml           the field reference for a new analyst
      inflation.yaml           + labor_tightness, curve_slope, term_premium,
                               balance_sheet, financial_conditions, inflation_expectations
  evaluation/
    ic.py            ICEvaluator — release-clock IC, the bar, calibration split
    panel.py         FeaturePanel + release_dates (with market-clock resampling)
    report_quality.py deterministic prose checks over saved runs

src/run_analyst.py     single-analyst pilot runner (--dry-run to inspect a prompt)
src/run_analyst_ic.py  scored LLM run on the release clock
src/run_feature_ic.py  the free, no-LLM feature check
src/compare_sweep.py   tabulate IC across model runs
scripts/fetch_fred.py  add a FRED series to data/fred/ (needs a key)
tests/                 focused suite, no LLM calls
```

---

## 4. Every class and function

### `src/data/fred_local.py`
| Symbol | Description |
|---|---|
| `csv_dir()` | Directory holding the raw FRED CSVs; `FRED_CSV_DIR` overrides the default sibling-repo path. |
| `available()` | Series ids present on disk. |
| `load_series(series_id, start, end)` | One series indexed by approximate **release** date, not observation date. Applies `PUBLICATION_LAG_DAYS` so `.loc[:asof]` cannot see a print before it existed. |
| `load_bundle(series_ids, start, end)` | `load_series` over a list. |

### `src/data/fomc_text.py` (addition)
| Symbol | Description |
|---|---|
| `FomcCorpus.pair_as_of(asof)` | `(current, previous)` documents available at `asof`. Consecutive statements are ~0.80 similar, so the information is in the edit; serving the predecessor is what lets a selector show the change rather than the boilerplate. |

### `src/layered/contracts.py`
| Symbol | Description |
|---|---|
| `SeriesFeature` | One measured quantity as a trajectory: `name`, `values` (oldest→newest), `unit`. |
| `ScalarFeature` | One measured quantity as of now: `name`, `value`, `unit`. |
| `FeatureSet` | Everything measurable an analyst may see. Also the input contract for benchmarks, so "did it see more data?" becomes a property of the type. |
| `FeatureSet.names` | All feature names — the valid vocabulary for `key_evidence`. |
| `FeatureSet.level` | The driver's headline measurement, resolved from `level_feature`. What scoring grades against. |
| `FeatureSet.render()` | The measurement block as the analyst sees it. Relative time labels only, never dates. |
| `DriverView.report` | The analyst's prose analysis — the contract to an LLM PM. |
| `DriverView.key_evidence` | Feature names cited, already validated against what was supplied. |
| `DriverView.falsifier` | What would change the view. |
| `DriverView.source` | Provenance, e.g. `llm:inflation`, `benchmark:persistence`. Makes a stream self-describing. |
| `DriverView.degraded` | Emitted after a failure; exclude from grading. |

### `src/layered/features/ops.py`
Every operation maps one or two `pd.Series` to a `pd.Series`, so a spec can take
either the trajectory or the current reading from one definition.

| Symbol | Description |
|---|---|
| `level(s)` | The series itself — for drivers whose level is the measurement. |
| `diff(s, window)` | Absolute change over `window` observations. |
| `pct_change(s, window)` | Percent change over `window`. |
| `yoy(s, periods)` | Year-over-year percent change. |
| `pct_change_annualized(s, window, periods_per_year)` | Change over `window` at an annual rate. Turns before year-over-year does. |
| `moving_average(s, window)` | Rolling mean. |
| `spread(a, b)` | `a − b`, aligned and forward-filled (carries values forward only, so no look-ahead). |
| `distance_from_reference(s, reference)` | Distance from a stated policy constant, never a fitted one. |
| `rolling_min(s, window)` / `rolling_max(s, window)` | Range position. |
| `REGISTRY` | `name → (function, arity, allowed parameters)`. The closed vocabulary. |
| `apply(op, inputs, params)` | Looks up and runs one operation, validating arity and parameter names. |

### `src/layered/features/spec.py`
| Symbol | Description |
|---|---|
| `FeatureDef` | One definition: `name`, `op`, `sources`, `params`, `unit`, `history`. `history > 1` renders as a trajectory. |
| `FeatureDef.raw_sources` | Sources that are raw series ids rather than `@references`. |
| `FeatureSpec` | A driver's whole spec: `series`, `scalars`, `level_feature`. |
| `FeatureSpec.definitions` | Series first — scalars may reference them by `@name`. |
| `FeatureSpec.declared_inputs` | Every raw series the spec may read. The isolation contract. |
| `_parse_def(raw, default_history)` | Parses one YAML entry; unrecognised keys become op parameters. |
| `from_persona(driver, persona)` | Builds and validates a spec: rejects duplicate names and an undefined `level_feature`. |

### `src/layered/features/engine.py`
| Symbol | Description |
|---|---|
| `FeatureEngine(spec)` | Computes one driver's `FeatureSet`. |
| `FeatureEngine.inputs` | Raw series this engine may read. |
| `FeatureEngine.compute(world)` | Evaluates every definition through the `AsOf` gate, resolving `@references` from a cache and recording which raw series were touched. Features without enough history yet are omitted rather than invented. **The single choke point for input isolation.** |

### `src/layered/text/selector.py`
| Symbol | Description |
|---|---|
| `scrub_dates(text)` | Replaces absolute dates and release times with `[date]` / `[time]`. Applied to every arm. |
| `sentences(text)` | Sentence split; FOMC prose is clean enough for a boundary split. |
| `TextContext` | The text an analyst receives: `added`, `removed`, `unchanged`, plus availability. |
| `TextContext.render()` | Renders the change block and the recurring-context block separately. |
| `TextSelector` | Abstract interface. Swappable so the open preprocessing question becomes a measurable arm rather than a blocker. |

### `src/layered/text/cue.py`
| Symbol | Description |
|---|---|
| `strip_chrome(text)` | Removes the release header and the media/implementation trailers. |
| `compile_cues(cues)` | Compiles cues to boundary-aware regex. The negative lookbehind stops `"2 percent"` matching inside `"4-1/2 percent"`; the trailing `\w*` keeps `price → prices`. |
| `CueSelector._passages(text, patterns)` | Ordered `{key: sentence}` for sentences matching any cue, after scrubbing and chrome removal. |
| `CueSelector.select(asof, cues, driver)` | Selects the current document's passages and diffs them against the previous document's. |

### `src/layered/text/whole.py`
| Symbol | Description |
|---|---|
| `WholeDocumentSelector.select(...)` | Serves the entire scrubbed document, ignoring cues. The un-partitioned control — varies the partition only, not the leak surface. |

### `src/layered/analysts/llm_analyst.py`
| Symbol | Description |
|---|---|
| `LLMAnalyst` | One driver, one specialist, one report. No `read()` and no per-driver subclass. |
| `from_persona(driver, llm, text_selector)` | Builds an analyst entirely from its YAML file. |
| `inputs` / `cues` | The isolation contract and the text cues. |
| `build_inputs(world)` | Returns `(FeatureSet, TextContext)` — everything the analyst may see. Exposed so a prompt can be inspected without spending a call. |
| `_system_prompt()` | Role, mandate, what the evidence is, the instruction not to infer the period, the conviction ladder, that abstention is legitimate, and the output contract. |
| `_user_prompt(features, text)` | Driver name, the rendered measurement block, the rendered text block. Nothing else. |
| `form_view(world)` | Builds the evidence, then delegates to `form_view_from`. |
| `form_view_from(features, text)` | One call on evidence already assembled. Validates direction and conviction, filters `key_evidence` to real feature names, returns a `DriverView`. Split out so `CarryForward` does not pay to build the evidence twice. |
| `_degraded(features, why)` | An explicit abstention on failure — never a benchmark's answer. |

### `src/layered/analysts/carry_forward.py`
| Symbol | Description |
|---|---|
| `CarryForward(analyst)` | Wraps an analyst so it forms a view only when its evidence changes. Implements the same interface (`driver`, `inputs`, `build_inputs`, `form_view`), so it is a drop-in `ViewSource`. |
| `CarryForward._evidence_key(features, text)` | SHA-256 of both rendered prompts. The system prompt is included so a persona edit invalidates the cache. |
| `CarryForward.form_view(world)` | Returns the cached view stamped with the current meeting and `carried=True` when the key matches; otherwise calls and caches. Degraded views are never cached. |
| `CarryForward.stats` | `meetings`, `calls_made`, `carried`, `carried_share`. |

### `src/run_analyst.py`
| Symbol | Description |
|---|---|
| `build_analyst(args, llm)` | Wires the selector (`cue` / `whole` / `none`) and loads the persona. |
| `main()` | CLI: single date or a schedule, `--dry-run` to print the prompt without spending, `--out` to write views as JSONL, run audit of calls/tokens/cost. |

---

## 5. The pipeline, end to end

### Input

Two channels, both restricted to this driver, both point-in-time.

1. **Measurements.** `FeatureEngine.compute(world)` reads only the spec's declared
   series (`CPIAUCSL`, `PCEPILFE` for inflation) through `AsOf`, which slices to
   `<= asof`. The series were already shifted to their release dates on load, so the
   two guarantees compose: the right vintage, sliced at the right moment. It applies
   the spec's operations and returns a `FeatureSet` of two trajectories (13 monthly
   observations each) and eight derived scalars.
2. **Text.** `CueSelector.select(asof, cues)` takes the latest statement with
   `release_date <= asof` and its predecessor, strips publication chrome, scrubs
   dates and times, splits into sentences, keeps those matching this driver's cues,
   and diffs current against previous into `added` / `removed` / `unchanged`.

### What the model reads

System prompt: the mandate, a statement that it is being shown measurements and
policy language with no computed direction and no later information, an explicit
instruction not to identify the calendar period, a conviction calibration ladder, a
statement that abstention is legitimate, and the output contract.

User prompt: `Driver: <name>`, then `FeatureSet.render()`, then
`TextContext.render()`. There is no date, no direction, no conviction, and no prior
view anywhere in it.

### Output

A single JSON object, in this field order:

```json
{
  "report": "<120-250 words of prose>",
  "key_evidence": ["<feature names relied on>"],
  "falsifier": "<what would change this view>",
  "direction": "up" | "down" | "flat",
  "conviction": 0.0
}
```

`form_view` validates the direction, clamps conviction to `[0, 1]`, filters
`key_evidence` down to names actually supplied, attaches `level` from the feature
set for scoring, stamps `source`, and returns a `DriverView`. Any failure — a bad
response, unparseable JSON, an invalid direction — returns a flat, zero-conviction
view with `degraded=True`.

---

## 6. The evaluation layer

`src/layered/evaluation/` is deliberately separate from both the analyst and the
legacy `diagnostics.py`, because it must grade three different kinds of thing over
this project's life — raw features now, analyst views next, benchmark rules later —
and none should have to know about the others. Everything consumes a signal indexed
by release date, so a feature, a rule, and an LLM's signed conviction are scored by
identical code.

### Why the release clock

The prediction is fixed at **the next release**. That decision does real statistical
work. A fixed calendar horizon graded on a weekly schedule makes consecutive
observations share almost their whole outcome window — a quarterly horizon sampled
weekly overlaps by about twelve weeks in thirteen — and errors that autocorrelated
make a naive t-statistic badly overstated. Release-to-release changes are
**non-overlapping by construction**, so the t-statistic is honest with no
Newey-West, no block bootstrap, and no discarding of the sample. The sample is then
simply what it always was, about twelve a year; the weekly schedule only made it
look larger.

### Why IC leads, and what "good" means

Rank IC measures ordering skill and is indifferent to how conviction is scaled —
which matters because the calibration ladder in the prompt is itself untested. A
Sharpe-style number conflates "was the direction right" with "was the size right",
and when it comes back weak you cannot tell which failed.

The bar is set by breadth, not by taste. Under `IR ≈ IC · √breadth`, and with
breadth fixed by design at one bet per release (~11.8/yr):

| Target IR | Required IC |
|---|---|
| 0.5 | 0.15 |
| 1.0 | 0.29 |

A cross-sectional book making hundreds of bets is content with IC 0.05. A single
driver making twelve is not. This is the argument for the multi-analyst extension
being where the value is: breadth comes from many weakly-skilled *independent*
analysts, which makes the correlation diagnostic load-bearing rather than
decorative.

### Classes and functions

| Symbol | Description |
|---|---|
| `release_dates(macro, series_id, start, end)` | The clock a driver moves on — its target series' release dates. Series are release-dated on load, so these are when the number became known. |
| `FeaturePanel(engine)` | Replays a spec across history. |
| `FeaturePanel.clock_series` | Default clock — the first declared input. |
| `FeaturePanel.build(macro, dates)` | `(date × feature)` matrix; every row computed through `AsOf`, so a feature at *t* uses only data at *t*. A series feature contributes its latest value. |
| `FeaturePanel.level(panel)` | The driver's level column — what outcomes are measured against. |
| `ICEvaluator(level, steps)` | Scores any signal against the move over the next `steps` releases. |
| `ICEvaluator.outcome` | Change in level over the horizon. Positive = the driver rose. |
| `ICEvaluator.releases_per_year` | Inferred from the clock's own spacing, never assumed. |
| `ICEvaluator.breadth` | Independent bets per year. |
| `ICEvaluator.evaluate(signal, name)` | One `ICResult`. Spearman computed as Pearson-on-ranks to avoid pandas' scipy-backed `method="spearman"` (scipy is not a dependency). |
| `ICEvaluator.evaluate_frame(frame)` | Every column, strongest \|IC\| first. |
| `ICEvaluator.calibration_split(signed)` | IC of direction alone vs IC of signed conviction. A cleaner read on calibration than Sharpe, and it needs no annualization of a quantity that is not a return: if the two are close, conviction carries no ordering information. |
| `ICEvaluator.signal_sharpe(signed, ppy)` | Secondary, sizing-sensitive, explicitly not tradable. Annualized from the clock's real spacing — the legacy `diagnostics.signal_sharpe` hardcodes 52, which overstates a release-clock figure by `√(52/12) ≈ 2.08`. |
| `ICResult` | `name, n, ic, t_stat, p_approx, hit_rate`. `p_approx` is a normal approximation (no scipy); read the t-statistic. `hit_rate` is meaningless for features without a natural zero. |
| `required_ic(target_ir, breadth)` | The bar, statable in advance. |
| `src/run_feature_ic.py` | CLI for the free pre-LLM check. |

**The honesty rule for this module**: these ICs diagnose the *problem*. Feeding them
back to the analyst, or selecting features because they scored well here, would turn
a measurement into a fitted signal and reintroduce exactly the anchoring the
redesign removed. It informs the researcher; it must never inform the prompt.

---

## 7. Baseline result — is inflation predictable at all?

Run before any model call. Headline CPI year-over-year, next-release change, 10
features, non-overlapping observations.

**Full sample, 2005-01 → 2026-05 (255 obs)**

| signal | IC | t | hit |
|---|---|---|---|
| headline_yoy_12m_low | −0.193 | −3.13 | 0.500 |
| headline_yoy_12m_high | −0.182 | −2.94 | 0.520 |
| headline_yoy_change_3m | **+0.158** | 2.55 | 0.571 |
| headline_cpi_yoy | −0.105 | −1.68 | 0.512 |

Two opposing effects, both economically sensible: the *level* of inflation
negatively predicts its next change (mean reversion), while recent *acceleration*
positively predicts more acceleration (momentum).

**Pre-COVID, 2005-01 → 2019-12 (179 obs)**

| signal | IC | t | vs full sample |
|---|---|---|---|
| headline_yoy_12m_high | **−0.229** | −3.12 | holds, stronger |
| headline_cpi_yoy | −0.134 | −1.80 | weaker |
| headline_yoy_12m_low | −0.118 | −1.58 | **collapses** from −0.193 |
| headline_yoy_change_3m | +0.069 | 0.92 | **collapses** from +0.158 |

This is the finding that matters. **The momentum effect does not survive excluding
COVID** — it drops from t = 2.55 to t = 0.92, meaning the full-sample result was
driven by the 2021–22 run-up, a single long monotonic acceleration. `12m_low`
collapses the same way. Only **`headline_yoy_12m_high` is stable across both
windows**, at IC ≈ −0.18 to −0.23: mean reversion from an elevated range.

Reading it honestly: one of ten features survives a subsample split, which is about
what one real effect plus noise looks like, and picking the best of ten biases the
maximum upward. The stable effect implies IR ≈ 0.23 × √11.8 ≈ **0.79** — real,
below an IR of 1, and achieved by a single measurement.

**What this sets up for the analyst.** The driver is somewhat predictable, so a weak
model result would be about the model rather than the problem. And the interesting
question is now sharper than "does it beat a coin flip": the two effects point in
opposite directions and only one is robust, so the test is whether an analyst
reasoning over both — with the policy text as context — weighs them better than the
single stable measurement does. That is a harder and more meaningful bar than the
old deterministic rule ever posed.

---

## 8. Agent result — first run, 2026-07-19

Inflation analyst, next-release horizon, cue-partitioned statement text,
`claude-haiku-4-5`, 256 releases 2005-01 → 2026-05. 260 calls, 4 retries, $0.99.
Full audit in `reports/inflation_ic.jsonl` + `.meta.json`.

| | IC | t | n |
|---|---|---|---|
| **Agent (signed conviction)** | **+0.122** | 1.96 | 255 |
| direction only | +0.073 | 1.16 | 255 |
| best single feature, same window | −0.193 | −3.13 | 255 |

Implied IR 0.42 against a bar of 0.29 IC for IR 1.0. Hit rate 0.55. Direction mix
balanced (106 up / 95 down / 55 flat).

**The agent does not beat the best single measurement.** 0.122 against 0.193 on the
identical window, clock and outcome.

### Where the skill actually sits

Split on the same pre/post-COVID boundary used for the features *before* this run,
so it is a pre-specified split rather than fishing:

| window | n | IC | t |
|---|---|---|---|
| pre-COVID 2005–2019 | 179 | **+0.043** | 0.57 |
| COVID onward 2020–2026 | 75 | **+0.316** | 2.84 |
| post-training-cutoff 2025–2026 | 15 | −0.181 | −0.66 |

All of the measured skill is in the COVID window. Pre-COVID the agent has
essentially nothing.

### Why — it copied the fragile feature

Correlation of the agent's signed conviction against each feature:

| feature | corr |
|---|---|
| headline_yoy_change_3m (momentum) | **+0.362** |
| headline_yoy_12m_low | −0.276 |
| core_yoy_change_3m | +0.238 |
| headline_yoy_12m_high (**the stable one**) | −0.127 |

The agent tracks **momentum** most closely — and momentum is precisely the feature
that collapsed out of sample (+0.158 full sample → +0.069 pre-COVID, t 2.55 → 0.92).
Meanwhile it barely tracks `headline_yoy_12m_high`, the one effect that *did* survive
the subsample split at IC −0.23.

So the analyst latched onto the salient but regime-dependent signal and largely
missed the stable one. That is a specific, fixable diagnosis rather than "the LLM
does not work".

### What did work

Conviction carries real information: IC rises from 0.073 (direction alone) to 0.122
(signed), so the calibration ladder is doing something. But the distribution is
compressed — mean 0.565, range 0.25–0.75, never once using the ladder's 0.0–0.2 or
0.9–1.0 bands. There is headroom in sizing.

The post-cutoff slice is negative rather than positive, which at least does not
suggest a training-cutoff leak — but n=15 is noise and settles nothing.

### Was the COVID-period skill a training-cutoff leak?

Tested, and the answer is no — or at least, not large enough to matter. The features
are pure pandas and **cannot** have lookahead, so they set a no-leak ceiling in each
regime:

| window | agent | best feature | momentum feature |
|---|---|---|---|
| pre-COVID 2005–2019 (n=179) | +0.043 | −0.229 (`12m_high`) | +0.069 |
| COVID onward 2020–2026 (n=75) | +0.316 | **−0.472** (`12m_low`) | +0.336 |

In the COVID window the agent's 0.316 is *below* both the momentum feature (0.336)
and the range feature (0.472). A leak would have to show up as the agent **exceeding**
what a lookahead-free computation achieves where memorization helps most. It does the
opposite. The regime difference is real: that window is simply more predictable, for
arithmetic and agent alike, and the agent's regime pattern is inherited from the
features rather than from recall.

This does not prove zero leak; it shows leak is not large enough to lift the agent
above the no-lookahead benchmark. The n=15 post-cutoff slice still settles nothing.

**But the leak surface is genuinely open and should be closed anyway.** Date scrubbing
removes "February 01, 2023" and leaves *"supply and demand imbalances related to the
**pandemic**"* and *"**The war** and related events are contributing to upward pressure
on inflation"* — both verbatim in the rendered prompts, both pinning the era to within
months. Scrubbing calendar tokens is not the same as removing period identity.

### The sharper diagnosis

The agent underperforms the best lookahead-free feature **in both regimes** — 0.043
vs 0.229 pre-COVID, 0.316 vs 0.472 during COVID. Combined with its correlation
profile (momentum +0.36, `12m_high` only −0.13), the reading is:

> The analyst is approximating the momentum feature, imperfectly, and systematically
> under-weighting the level/range information — which is the strongest signal in
> *both* regimes.

Note the mandate already instructs it to "weigh both the level relative to the 2%
target and the recent momentum". It is being told; it is not doing it. That points at
the feature presentation rather than the instruction: `headline_yoy_12m_low` and
`_12m_high` are handed over as raw levels, so "we are at the top of the range" is
arithmetic the model must perform in its head. An explicit range-position measurement
would surface it directly.

### Consequences

1. **Base effects are the obvious missing feature.** The next print's YoY change is
   the incoming month minus the month rolling out of the window, and that second term
   is known today. The agent is being asked to predict something whose mechanical
   half it is never shown. Pure arithmetic, no fitting — it passes the
   measurement-only test.
2. **Carry-forward was never exercised here** (0 carried): on the release clock the
   evidence always changes. It matters for the weekly meeting loop, where it was
   verified offline at ~50%.
3. **Inflation caps out at ~12 bets/yr.** Even a strong IC gives IR ≈ 1. The
   architecture should not be judged on this driver alone.

---

## 9. Second run — base effects added (2026-07-19)

Same config, plus six features: the monthly path (`headline_mom`), the base-effect
terms (`outgoing_mom`, `headline_mom_3m_avg`, `mom_gap_vs_outgoing`), and explicit
range position (`yoy_above_12m_low`, `yoy_range_position`). 257 calls, 1 retry, $1.06.
Audit in `reports/inflation_ic_v2.jsonl`.

### The features are hugely informative — and stable

| feature | full IC | pre-COVID IC | hit |
|---|---|---|---|
| `outgoing_mom` | **−0.680** (t −14.8) | **−0.738** (t −14.5) | 0.36 |
| `mom_gap_vs_outgoing` | **+0.628** (t 12.8) | **+0.622** (t 10.6) | 0.75 |
| everything else | ≤ 0.21 | ≤ 0.23 | ~0.55 |

Unlike momentum, these do **not** decay out of the COVID window. That is expected:
a year-over-year rate is a rolling 12-month window, so the observation leaving it is
known today and mechanically determines part of the next reading. Legitimate (it is
data from *t*−11, gated by `AsOf`) but closer to an accounting identity than a
forecast, and it reframes the target: predicting ΔYoY is largely arithmetic plus a
genuine call on the incoming month.

### The agent barely used them

| | v1 | v2 |
|---|---|---|
| agent IC (full) | +0.122 | **+0.156** |
| pre-COVID | +0.043 | **+0.138** |
| COVID onward | +0.316 | +0.190 |
| hit rate | 0.550 | 0.577 |
| IC added by conviction | +0.049 | **+0.004** |

Genuine progress on robustness — the pre-COVID hole largely closed, and performance
is now consistent across regimes rather than concentrated in COVID. But:

| what it tracks (corr with signed conviction) | |
|---|---|
| `headline_mom` | +0.285 |
| `headline_yoy_change_3m` | +0.268 |
| `mom_gap_vs_outgoing` | +0.178 |
| `outgoing_mom` | < 0.10 |

It is still tracking **momentum**, and largely ignoring the two features that carry
almost all the signal. Direct comparison against the one-line mechanical rule:

- agreement with `sign(mom_gap_vs_outgoing)`: **58.6%** — barely above chance
- rule IC **+0.524** vs agent IC **+0.156**

**A one-line sign rule beats the LLM analyst by more than three times.**

### Why — the features have no semantics in the prompt

The base-effect logic is documented in `inflation.yaml` comments and in this file.
Neither reaches the model. What the analyst actually receives is `outgoing_mom
+0.42 % m/m` — a name and a number, with nothing to say that this quantity
mechanically *subtracts* from the next reading. The mandate never mentions base
effects at all.

So the binding constraint is not the feature space and not the reasoner. It is that
**a measurement was supplied without its economic meaning.** The analyst was given the
right number and no reason to care about it, and fell back on the pattern it could
read directly off the trajectory.

That is a fixable and rather interesting result: it says the mandate — the *knowledge*
half of the persona — carries more weight than the feature block, and that
measurements alone do not make an analyst. Explaining base effects in the mandate is
instruction, not signal, so it stays inside the measurement-only rule.

Also note conviction stopped doing any work (+0.049 → +0.004): direction-only and
signed conviction now score the same, and the distribution stayed compressed
(0.35–0.75, never using the ladder's tails).

---

## 10. Model sweep — Haiku / Sonnet / Opus (2026-07-20)

Identical prompt, features (the §9 base-effect set), clock and horizon; only the
model varies. Tool-forced structured output on all three (see §11). ~1 degraded
call each, excluded. Files `reports/sweep_{haiku,sonnet,opus}.jsonl`.

| model | IC | t | pre-COVID | COVID+ | conv_adds | hit |
|---|---|---|---|---|---|---|
| Haiku 4.5 | 0.187 | 3.0 | 0.142 | 0.291 | +0.052 | 0.561 |
| Sonnet 5 | 0.340 | 5.7 | 0.253 | 0.512 | −0.005 | 0.673 |
| Opus 4.8 | **0.492** | 9.0 | **0.447** | 0.549 | +0.055 | 0.710 |

**IC rises steeply and monotonically with model capability, and holds pre-COVID
for the strong models** (Opus 0.447, t≈6 on 179 pre-COVID obs). Implied IR at Opus
is ~1.7. This confirms we were **reasoner-limited**: the features and the design
were adequate; the weak model was the bottleneck. It is direct support for the
thesis — supply measurements, let the model's own knowledge connect them.

### The gain is at least partly demonstrable reasoning

Correlation of each model's signed conviction with the base-effect features it was
given:

| model | outgoing_mom | mom_gap_vs_outgoing | headline_mom (trend) |
|---|---|---|---|
| Haiku | −0.06 | +0.21 | +0.34 |
| Sonnet | −0.20 | +0.53 | +0.66 |
| Opus | **−0.42** | **+0.67** | +0.53 |

We can *watch* the reasoning improve. Haiku ignores `outgoing_mom` (−0.06) and
anchors on the salient trend — the §9 finding, now shown to be **model-specific,
not a design flaw**. Opus engages the base-effect measurement strongly and in the
**economically correct direction** (a hot outgoing month → predict down, −0.42),
approximating `mom_gap_vs_outgoing`, which was the single most predictive feature
in the free check. The strong model discovers and uses the genuinely predictive
measurement; the weak one does not. That is the cleanest evidence available that
the model-strength gain is reasoning over the provided features, not an artifact.

### Why 0.492 cannot be taken at face value

The window 2005–2024 sits **inside every model's training data**, and a stronger
model has memorized more of the actual inflation path. "IC rises with capability"
is therefore *also* the signature of a training-cutoff leak, and this sweep cannot
by itself separate the two.

The one discriminating slice — data after the models' cutoff — points toward
caution:

| model | 2005–2019 | 2020–2024 | **2025+** |
|---|---|---|---|
| Haiku | 0.142 | 0.318 | 0.162 |
| Sonnet | 0.253 | 0.541 | 0.163 |
| Opus | 0.447 | 0.563 | **0.181** |

On the most recent slice the enormous model spread **collapses to a
model-independent ~0.16–0.18.** If the big-model advantage were transferable skill
it should survive; it does not. But this is *not* conclusive — n=15 is tiny, and
2025-on is an intrinsically low-signal period (inflation becalmed near target), so
convergence could be the period rather than the cutoff.

**Net.** Reasoner-limited is confirmed, and the reasoning is partly visible in
correct feature usage — but the absolute in-sample IC is an **upper bound**, not a
clean out-of-sample skill estimate. Separating skill from recall needs genuine
post-cutoff data (only time supplies it at volume) or a driver/window outside
training. The date-scrub gap (§8: "pandemic"/"war" survive) makes closing the leak
surface worthwhile regardless.

The base-effect usage also revises §9: "the agent ignored the features" was true of
**Haiku**, not of the architecture. A capable model uses them.

---

## 11. Feature descriptions — the A/B (2026-07-20)

Step 2: does telling the model *what each feature is* (construction only — see the
`--describe-features` arm and the persona `description:` fields, lint-checked free of
any direction/implication word) help it use features it ignores? Arm A is the §10
sweep; arm B is the described re-run, same models, one variable changed.

| model | arm | IC | t | preCOVID | outgoing_mom | mom_gap | trend |
|---|---|---|---|---|---|---|---|
| Haiku | undescribed | 0.188 | 3.0 | 0.142 | −0.06 | +0.21 | +0.34 |
| Haiku | **described** | 0.163 | 2.6 | 0.142 | −0.11 | +0.23 | +0.24 |
| Sonnet | undescribed | 0.340 | 5.7 | 0.253 | −0.20 | +0.53 | +0.66 |
| Sonnet | **described** | 0.343 | 5.8 | 0.260 | −0.22 | +0.54 | +0.63 |
| Opus | undescribed | 0.490 | 8.9 | 0.447 | −0.42 | +0.67 | +0.53 |
| Opus | **described** | 0.493 | 9.0 | 0.465 | −0.43 | +0.68 | +0.53 |

**Clean null.** IC flat within noise for every model. The leak-robust readout is
decisive: Haiku's `outgoing_mom` usage did not shift toward Opus's (−0.06 → −0.11,
against Opus's −0.42), and its IC did not improve. Construction-only descriptions do
not rescue a weak reasoner.

### What it means, and the tension it exposes

The labels were **not** the bottleneck — the Haiku↔Opus gap is reasoning capacity,
not presentation. §10 is reinforced: reasoner-limited, full stop.

There is a real tension here worth stating. The only descriptions that might lift a
weak model are the ones that explain what a feature *means* — "outgoing_mom is the
month rolling out of the window, so a hot one pulls next year-over-year down." But
that is precisely the injected rule the design forbids; it hands over the conclusion
instead of the measurement. So within the measurement-only invariant, **you cannot
cheaply buy reasoning with better labels** — a capable model already infers what the
bare name affords, and a weak one is not helped by construction notes. The lever is
model strength, not description.

Practical consequence for step 3 (more features): adding features will help only a
model strong enough to use them (Opus did; Haiku did not use the ones it already has).
Run step 3 on the strong model, and judge it by feature-usage and out-of-sample /
post-cutoff behaviour, not in-sample IC.

---

## 12. Report quality — grading the prose, not the number (2026-07-20)

Deterministic, offline checks over the saved sweep runs
(`evaluation/report_quality.py`). Until now only the header (direction/conviction)
was ever graded; the report is the actual contract to the PM, so this checks it.

| run | names_trade | hallucinated | cites_text | cross_driver | dir_consistent | falsifier | words |
|---|---|---|---|---|---|---|---|
| haiku | 0.00 | 0.00 | 0.00 | 0.95 | 0.75 | 1.00 | 258 |
| haiku_desc | 0.00 | 0.01 | 0.00 | 0.93 | 0.79 | 1.00 | 257 |
| sonnet | 0.00 | 0.00 | 0.18 | 0.69 | 0.88 | 1.00 | 205 |
| opus | 0.00 | 0.00 | 0.00 | 0.66 | 0.84 | 1.00 | 201 |
| opus_desc | 0.00 | 0.00 | 0.00 | 0.63 | 0.85 | 1.00 | 200 |

Reports are well-formed: **no trade is ever named** (the core mandate rule holds),
evidence is grounded (~0 hallucinated feature names), a falsifier is always present,
and the prose leans the same way as the header 75–88% of the time — rising with model
capability. `cross_driver` is high but is the soft lexical signal (reports legitimately
discuss the Committee's *inflation* stance, and the FOMC text is in the prompt); it too
falls as the model gets stronger (0.95 → 0.63), i.e. better models stay more on their
own driver. Only Sonnet cites the FOMC text as prose evidence.

### Methodology caution (the metrics bit back twice before they were trusted)
Three separate false signals had to be caught before these numbers were believable,
and all three were artifacts of the checks, not the reports:
1. A "trade-naming" rate rising to 48% — entirely the word *position*, matching the
   feature `yoy_range_position` and ordinary English. Fixed by a report-specific trade
   lexicon (no "position"/"curve").
2. Sonnet "hallucinating" features 57% of the time — actually citing the FOMC *text*
   channel in prose ("policy language change"), legitimate evidence. Separated as
   `cites_text`.
3. Residual Sonnet "hallucination" of single characters (`_`, `e`, `o`) — Sonnet
   sometimes fills the `key_evidence` array with one comma-joined **string**, which the
   parser shredded into characters. This exposed a real (minor) bug: `LLMAnalyst` now
   coerces a string `key_evidence` into a list before validating. Affects only the
   audit field, not direction/conviction, so no scored result changes.

The lesson for anyone extending these checks: report every lexical rate with the
underlying hits visible, and verify a surprising number before believing it.

---

## 13. Open questions and known gaps, in priority order

1. ~~The prompt never states the horizon.~~ **Done.** The persona now carries a
   `horizon` block (`kind: next_release`, `clock: CPIAUCSL`), and the system prompt
   states it explicitly, naming the graded measurement so the call is unambiguous:
   *"will `headline_cpi_yoy` be HIGHER, LOWER, or essentially unchanged at the next
   monthly CPI release, compared with the most recent value you have been shown."*
   That is exactly what `ICEvaluator.outcome` measures. `approx_days: 31` populates
   the `DriverView` contract, which predates the release clock and requires a day
   count.
2. ~~Carry-forward on unchanged evidence is not implemented.~~ **Done.**
   `analysts/carry_forward.py` wraps an analyst, hashes both rendered prompts, and
   re-emits the previous view marked `carried=True` when nothing has moved. Measured
   on real data: **26 calls for 52 weekly meetings in 2023, 159 for 313 meetings over
   2019–2024 — 49–50% carried.** The ~26 evidence-change dates a year are CPI (12)
   plus core PCE (12) plus statements (8), less the ones that coincide.
   The saving in money is trivial (cents on Haiku); the point is statistical.
   Counting carried views as fresh ones is what would make a monthly driver look like
   it produced 52 independent opinions a year when it produced twelve. Degraded views
   are deliberately not cached, so a failed call is retried rather than frozen.
3. **Temperature is an open question.** `AnthropicClient` never sets it, so it runs
   at the API default of 1.0, not 0. Under carry-forward the determinism argument is
   weaker than it first appears, since identical prompts would not be re-sent anyway.
   Deliberately left undecided.
4. **Sentence-level cross-driver bleed — two distinct kinds observed.** The
   boundary-aware cue fix removed the explicit rate *level* (`"4-1/2 percent"` no
   longer matches the `"2 percent"` cue, verified against the Feb-2023 statement).
   Two categories survive, and neither is a regex problem:
   - *Genuinely multi-driver sentences.* "readings on labor market conditions,
     inflation pressures and inflation expectations, and financial and international
     developments" — one sentence naming three drivers.
   - *Policy-path sentences that mention inflation incidentally.* "In determining the
     extent of future increases in the target range, the Committee will take into
     account … economic activity and inflation" — primarily about the rate path,
     which the inflation mandate explicitly forbids reasoning about, but a legitimate
     cue match.
   These are the open preprocessing decision, now with concrete instances.
5. ~~The cue fix is unverified.~~ **Verified** against the Feb-2023 statement.
6. **No benchmark arm yet.** Deferred by decision; enters as a `ViewSource` consuming
   the same `FeatureSet`.
7. **`diagnostics.signal_sharpe` is mis-annualized** for a release clock (hardcoded
   52). The existing published figures are inflated by roughly 2.08×.
