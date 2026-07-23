# The PM layer — results record

The measured state of the PM layer, in the same form as `analyst-layer.md` records the
analyst layer: dated result sections, each naming the exact run files so any number here
can be recomputed. The *reasoning* behind each change lives in `decisions.md`; this file
is what the changes produced.

The PM reads a meeting (seven analyst reports as of one date, replayed from disk) and
emits an `ArbitratedView`: a signed conviction per driver, prose notes, and — for a pod
that declares a `trade:` block — a `StrategyTrade` that crosses the PM→fund seam. Two
questions are scored: does the PM call each driver better than that driver's own analyst
(`pm_bench`, driver space), and does its trade make money in yield space (`trade_pnl`).

---

## 1. First duration run — the PM→fund seam populated (2026-07-22)

Duration pod, `claude-sonnet-5`, month-end clock, 120 monthly meetings 2016-01 → 2025-12.
Board replayed from `reports/ab/*_on.jsonl` (the seven-analyst board; the pod listens to
five of them). Offline vendored data throughout. 120 calls, 0 degraded, **$5.40**. Run in
`reports/pm/duration_on.jsonl` + `.meta.json`.

This is the first run in which `StrategyTrade` exists as data rather than as a type. The
trade path is clean: 90% of meetings produced a trade, every one grounded into the
declared `{DGS2, DGS10}` universe, 0% sign violations, 0 dropped legs, 0 rejected. The
remaining 10% are genuine abstentions (no trade emitted), not grounding failures.

**Two defects surfaced, neither of which is a judgment failure:**

*The driver block scored 0/5 against its analysts.* Split by polarity, the cause was a
contract ambiguity, not bad judgment. The pod mandate speaks in **rate space** ("judge
the net direction of nominal Treasury yields") while the `conviction` field spoke in
**driver space** ("positive = the driver's headline measurement rises"). For a +1-polarity
driver these coincide; for a −1-polarity driver they are opposite — a *shrinking* balance
sheet is an *upward* force on yields. On 55 of 120 meetings the PM followed the mandate and
`pm_bench` graded it as driver space:

| driver | polarity | corr(pm, analyst) | IC (analyst → pm) |
|---|---|---|---|
| balance_sheet | −1 | **−0.168** | 0.714 → **−0.167** |
| labor_tightness | −1 | −0.050 | 0.156 → 0.050 |
| financial_conditions | +1 | 0.818 | 0.442 → 0.346 |
| term_premium | +1 | 0.960 | 0.023 → 0.022 |
| inflation | +1 | 0.960 | 0.426 → 0.418 |

The two negative-polarity drivers decouple (sign agreement ~44–51%, coin-flip); the three
positive-polarity ones track almost perfectly. The PM said so in its own words —
"a modest upward yield force despite the analyst's own 'down' framing on asset levels".

*The PM was stateless.* It never saw its own previous arbitration or the position it was
carrying, so it re-struck the book from scratch every month: sign flips on **45.8%** of
months, mean |Δnet| 0.896 against mean |net| 0.904, and a **+0.52** correlation between the
previous month's 10y move and the new position (it chased). None of this was avoidable by
the model — with no incumbent position in the prompt, "do not reverse without cause" was
not an instruction it could act on.

Trade P&L, for the record: n=108, mean +0.0017 pp of yield, t=+0.08 — inside the noise
band. Direction hit rate 44.4%, weak, with the whole loss in the 2024-25 cutting cycle.

---

## 2. The two fixes (2026-07-22)

Both are design changes, not attempts to make the model predict better. Reasoning in
`decisions.md`; mechanism here.

**`answer_space: driver | rate`, declared per pod.** Binds *both* halves of the prompt from
one key — the calibration ladder in the system prompt and the `conviction` field
description are generated from it, so they can never disagree again. `pm_bench.benchmark`
takes it as an argument and re-orients a `rate` run through polarity before grading. A
misspelled value raises rather than defaulting, because a silent default would flip the
grader's reading of every number in the run. All four shipped pods declare `driver` — the
only setting under which `ic_pm` and `ic_analyst` compare like for like.

**PM memory (`--memory`), mirroring `LLMAnalyst._render_memory`.** The previous
arbitration comes back in — commitments only (convictions, the carried position, the
falsifier), never the previous notes. Off by default so the memory-less arm reproduces
byte-for-byte.

**The `flat` trade flag, forced out by the memory fix.** With memory on, the first pilot's
emission rate *fell* to 62%: the PM began writing "I am reversing my prior position" and
"the honest move is to flatten to near-neutral" — real decisions — while `_parse_trade`
drops zero-weight legs and returned `None`, recording a deliberate flat identically to
never answering, which then cascaded into the next meeting being told it held nothing.
Fixed with an explicit `flat: true` (empty legs → a real `StrategyTrade`, gross 0) and
contract wording that the trade block is *the position you want to carry after this
meeting, not the change you are making*. `trade_pnl` scores a chosen flat as a real zero.

---

## 3. A/B result — memory-off vs memory-on (2026-07-22)

Same pod, model, board, window. **OFF** = `duration_on.jsonl` (§1, memory off,
pre-`answer_space`). **ON** = `duration_mem_on.jsonl` (memory on, `answer_space: driver`;
120 calls, 1 degraded, **$5.73**). **MECH** = `duration_mech.jsonl`, the deterministic
control (no model, no spend) — the polarity-weighted rule the LLM must beat. The two fixes
are entangled in the ON arm (it turns both on at once); the polarity recovery is squarely
the `answer_space` fix, since re-orientation is what flips those signs.

The full A/B with plots and executed outputs is `notebooks/pm_trade_evaluation.ipynb` — all
three arms, driver recovery, turnover, trade P&L, and the layer-boundary coherence check.

### Driver block — does the PM track its analyst now?

| driver | polarity | corr OFF → ON | sign-agree OFF → ON |
|---|---|---|---|
| balance_sheet | −1 | −0.168 → **+0.996** | 0.444 → 1.000 |
| labor_tightness | −1 | −0.050 → **+0.865** | 0.510 → 0.981 |
| financial_conditions | +1 | 0.818 → 0.970 | 0.921 → 1.000 |
| term_premium | +1 | 0.960 → 0.985 | 0.989 → 1.000 |
| inflation | +1 | 0.960 → 0.983 | 1.000 → 1.000 |

### Driver IC — PM vs its analyst, graded in the pod's declared space

| driver | ic_analyst | ic_pm OFF | ic_pm ON | ic_mech |
|---|---|---|---|---|
| inflation | 0.418 | 0.418 | 0.425 | 0.338 |
| labor_tightness | 0.157 | 0.050 | 0.173 | 0.159 |
| term_premium | 0.023 | 0.022 | 0.031 | 0.004 |
| financial_conditions | 0.431 | 0.346 | 0.391 | 0.444 |
| balance_sheet | 0.710 | **−0.167** | **+0.713** | 0.661 |

PM beat its analyst on **0/5 (mean Δic −0.218) → 4/5 (mean Δic −0.001)**. Balance sheet
recovered from −0.167 to +0.713, essentially matching its analyst.

### Behaviour — turnover, holds, chase

| metric | OFF | ON | MECH |
|---|---|---|---|
| emission rate | 90% | 100% | 100% |
| mean gross | 0.904 | 0.238 | 0.975 |
| mean \|Δnet\| | 0.896 | **0.122** | — |
| sign flips (m/m) | 45.8% | **11.0%** | 33.6% |
| unchanged holds | 6.5% | 21.2% | — |
| chase corr (prev move → net) | +0.52 | +0.45 | — |
| sign violations | 0% | 0% | 0% |

Turnover collapsed: mean position change fell 86%, the book is now held or nudged rather
than re-struck monthly. The chase correlation eased but did not vanish — some is genuine
trend-following on the evidence, not the stateless artifact. `flat_rate` was 0% (the PM
always held *some* position; the affordance exists and is tested, it just was not chosen).

### Trade — yield-space P&L and direction

| arm | n | mean (pp) | t | hit | Sharpe (ann) |
|---|---|---|---|---|---|
| OFF | 108 | +0.0017 | +0.08 | 0.463 | +0.03 |
| ON | 119 | +0.0101 | **+1.73** | 0.546 | +0.54 |
| MECH | 120 | +0.0056 | +0.26 | 0.442 | +0.08 |

Direction hit rate (net sign vs next 10y move), by regime:

| arm | all | 2016-19 | 2020-21 | 2022-23 | 2024-25 |
|---|---|---|---|---|---|
| OFF | 0.444 | 0.439 | 0.545 | 0.522 | 0.273 |
| ON | 0.513 | 0.468 | 0.583 | 0.667 | 0.375 |

The trade improved on most axes: mean P&L moved from inside the noise band to leaning
positive (t +0.08 → +1.73), and the memory arm clears the mechanical control (t +0.26) on
the trade for the first time. Direction was up in the three earlier regime buckets — but
**the 2024-25 bucket stayed the worst (0.375), and it is the concern §4 is about, not a
success.** An earlier draft of this line read "direction up in every regime including the
2024-25 cutting cycle"; that reading is withdrawn (see §4).

**Read the P&L honestly.** With turnover this low the 119 observations are no longer
independent — the P&L is carried by a handful of persistent positions, so the effective n
is smaller than it looks and t=+1.73 is suggestive, not decisive. And the pipeline has no
predictive features by construction; "not negative, leaning positive" is the finding, not
"an edge". What is unambiguous is the driver-block recovery (0/5 → 4/5) and the turnover
collapse — both are structural, not statistical.

---

## 4. Post-cutoff degradation — is it a memorization leak? (2026-07-22, UNRESOLVED)

The regime table in §3 shows the trade's direction and P&L collapsing in the most recent
period. Good in-sample, bad out-of-sample is the signature of a training-cutoff
memorization leak — the model "recalling" what rates did in years inside its training data
rather than forecasting them — and it is the single biggest validity threat to an LLM run
over the past. This section records what was measured and why the question is **not yet
settled**.

Split the memory-on trade at 2024 (post = 2024-2025, n=24; pre = 2016-2023, n=95), against
the mechanical control on the same board:

| | hit PRE → POST | mean P&L PRE → POST |
|---|---|---|
| LLM (memory-on) | 0.55 → **0.38** | +0.0134 → −0.0030 |
| mechanical control | 0.45 → **0.29** | +0.0120 → −0.0202 |
| LLM − mechanical (excess) | +0.10 → +0.08 | — |

**First read — looks like regime difficulty, not a leak.** The deterministic control cannot
memorize the future, yet it degrades by essentially the same amount (hit −0.16 vs the LLM's
−0.17), and the LLM's *excess over it* is stable across the boundary (+0.10 → +0.08). A leak
in the arbitration step would show that excess collapsing post-cutoff. It does not.

**Why that read is wrong on its own — the mechanical PM is not a leak-free control.** It is
deterministic in the *arbitration* (`mechanical_pm.py:181`, `own = e.view.signed_conviction`),
but its inputs are the LLM *analysts'* convictions off the same board. If the memorization
lives in the analyst layer — an analyst recalling 2021 CPI or the 2021 curve — then BOTH the
LLM-PM and the mechanical-PM inherit it, and "they degrade together" is exactly what a shared
upstream leak produces. So this test rules out a leak in the PM step only, and says nothing
about the analyst step, which is the more likely site.

**Two confounds that can masquerade as a cutoff leak, both unresolved:**

- **Small sample.** Post-2024 is 24 observations; LLM hit 0.38 ± 0.10 (1 SE) sits only −1.3 SE
  from a coin flip. The split is also boundary-sensitive: at a 2023 cut the drop is 0.53 →
  0.47, not 0.55 → 0.38.
- **Direction bias.** The PM runs long-yield 69% pre and 79% post. A persistent long-yield
  tilt that paid while rates rose in-sample would fail when the 2024-25 cutting cycle turned,
  with no memorization involved.

**Status: unresolved.** The only control that isolates the cutoff question must bypass every
LLM in the stack — raw-feature IC (pure pandas, cannot memorize) and a persistence benchmark,
split pre/post cutoff, exactly the `analyst-layer.md` §8 methodology applied to the five rates
drivers this pod reads (it has been run for inflation only). If the analysts' edge over those
non-LLM baselines is stable across the cutoff it is regime difficulty; if the edge is all
pre-cutoff it is the leak, and it would explain the PM drop entirely, since the PM can only be
as clean as the board beneath it.

---

## 5. Provenance and caveats

- **Board.** Every arm replays the same `reports/ab/*_on.jsonl`. Two legs of that board
  were re-run after an API billing failure and may sit on a different model snapshot behind
  the same alias; per-leg SHA256 hashes are in each run's `.meta.json` under `board_sources`.
- **Entangled arms.** The ON arm flips memory *and* `answer_space` together. The driver-IC
  recovery is attributable to `answer_space` (re-orientation is what moves it) and the
  turnover collapse to memory (it is the only thing that changes position persistence), but
  a clean four-cell factorial has not been run.
- **One pod is not a book.** The ensemble, risk, and portfolio layers of the architecture do
  not exist yet, and `FundAllocation` has no producer. Nothing here allocates across pods or
  re-clips for a vol target.
- **Not a P&L.** Yield-space `Σ w·Δy` in percentage points — no duration weighting, no carry,
  no financing, no costs. It is the quantity the model was told it would be scored on.
