# First final results — full online run, 7 analysts × 4 PMs (sonnet, 2016–2025)

Canonical, controlled run. Board rebuilt online (sonnet, memory-on, 2016-01→2025-12),
then every PM arm scored against it on the same ME clock and yield-space P&L grader.
Pre-registered decision rule: a layer is "fixable" only if the layer above it carries
signal; otherwise a shortfall is inherited or a genuine limit. See `docs/pm-layer.md`.

## Analyst layer — analyst IC vs the feature-IC floor (same 2016–2025 window)

| driver | analyst IC (t) | best-feature \|IC\| (t) | verdict |
|---|---|---|---|
| balance_sheet | **0.690** (10.3) | 0.690 (10.3) | captures the full available signal |
| financial_conditions | **0.413** (4.9) | 0.390 (4.6) | captures available signal |
| inflation | **0.415** (4.9) | 0.618 (8.5) | captures ~⅔; some left on table |
| labor_tightness | 0.155 (1.7) | 0.427 (5.1) | **FIXABLE** — signal present, under-extracted |
| curve_slope | 0.119 (1.3) | 0.306 (3.5) | **FIXABLE** — under-extracted |
| inflation_expectations | **−0.079** (−0.9) | 0.258 (2.9) | **FIXABLE** — signal exists, analyst is *anti-signal* |
| term_premium | −0.012 (−0.1) | 0.136 (1.5) | **LIMIT** — feature floor itself insignificant; ~random walk |

Three strong legs, three fixable (measurable signal the LLM fails to convert; on
inflation_expectations it reasons *backwards*), one genuine limit.

## PM layer — yield-space trade P&L per pod, per arm

`mem_on*` = memory-on restricted to the blind window, for an apples-to-apples blind read.
Mechanical abstains on the opposed (curve) pod by design → no baseline there.

| pod | arm | n | mean (pp) | t | hit | sharpe |
|---|---|---|---|---|---|---|
| duration | mem_on | 119 | +0.0055 | +0.83 | 0.487 | +0.26 |
| duration | mem_off | 109 | +0.0072 | +0.44 | 0.468 | +0.15 |
| duration | **mech** | 120 | **+0.0105** | +0.49 | 0.475 | +0.15 |
| front_end | mem_on | 120 | −0.0010 | −0.26 | 0.567 | −0.08 |
| front_end | mem_off | 114 | +0.0190 | +1.54 | 0.570 | +0.49 |
| front_end | mech | 120 | +0.0163 | +0.72 | 0.508 | +0.23 |
| front_end | **blind** | 48 | **+0.0269** | **+2.73** | 0.562 | +1.35 |
| real | mem_on | 119 | +0.0007 | +0.20 | 0.437 | +0.06 |
| real | **mech** | 120 | **+0.0129** | +1.08 | 0.467 | +0.34 |
| curve | mem_on | 116 | +0.0027 | +0.70 | 0.517 | +0.22 |
| curve | mem_off | 113 | +0.0034 | +0.52 | 0.478 | +0.17 |

### Findings
1. **The memory-on advantage did NOT replicate.** On the prior board memory-on hit
   t=+1.73 and beat mechanical; on this fresh board it beats mechanical on **0 of 3**
   pods that have a baseline (duration mem_on +0.0055 < mech +0.0105). The earlier
   "win" was board-specific noise — vindicating the one-board-fluke caveat.
2. **No full-arbitration arm clears t=2.** Every memory-on/off/mech arm sits in noise
   at n≈120.
3. **On front_end, less structure beats more.** Blind (PM sees ONE report) is the only
   t>2 in the matrix (+2.73), and memory-off (+0.019) beats memory-on (−0.001).
   Arbitrating the whole board *dilutes* here rather than helping. (Caveat: ~17 cells
   scored → expect ~1 |t|>2 by chance; weight the *direction* mem_off/blind>mem_on, not
   the lone star.)
4. **Faithfulness flag.** Scrambling which report belongs to which driver leaves the
   duration trade 74.5% direction-unchanged (net corr +0.46) — the arbitration is only
   moderately dependent on correct report attribution.

### Limit vs fixable (PM)
- **real, curve** — underperformance is largely *inherited*: real feeds on
  inflation_expectations (anti-signal) and curve on curve_slope (under-extracted). Don't
  fix the PM here; fix the analyst.
- **duration** — feeds on three *strong* analysts yet fails to beat mechanical → a
  genuine PM/trade-seam shortfall, not inherited. Fixable: arbitration/sizing or the
  yield-space trade construction is lossy.
- **front_end** — arbitration is actively diluting; blind/mem_off > mem_on. Fixable:
  weight primary drivers, shorten the brief.

## Bottom line
On a properly-controlled fresh board, **the layered PM shows no edge over the mechanical
baseline**, and the prior memory-on win was non-replicable. The layer's current
justification is not P&L. The fixable analyst legs (labor, curve, inflation_expectations)
are the highest-value next target, since the PM cannot arbitrate signal that isn't there.
