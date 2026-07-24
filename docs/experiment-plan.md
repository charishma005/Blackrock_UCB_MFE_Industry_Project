# Experiment plan — the frozen-pipe validation run

The repo is the deliverable; this file is the pre-registered list of what we run on it and
why. It is written *before* the validation run so the decision rules cannot be fitted to
the outcome — the same discipline `decisions.md` holds changes to. Companion design records
are `analyst-layer.md` and `pm-layer.md` (the built state); `first-final-results.md` holds
the last full run's numbers.

**Everything below is provisional until re-run on the completed, frozen pipeline.** The
last full run (`first-final-results.md`) was on a sonnet board over 2016–2025 with a
partial toolchain. The validation run rebuilds the board on the frozen team repo with more
faithful (Alfred) data; every number here is superseded by that run. Results carried over
from earlier work are marked *provisional*; arms that need code not yet on `main` are
marked with their dependency.

## The two claims, kept separate

- **Claim A — design.** The layered contract (one analyst per driver → `DriverView` seam →
  pod PM → `StrategyTrade`) is structurally sound: no look-ahead, measurement-not-signal,
  input isolation. Graded by whether the invariants hold under adversarial perturbation and
  whether the layer above carries signal the layer below can use. *Provisionally ~80% there.*
- **Claim B — performance.** The layered stack turns analyst IC into trade P&L that beats a
  mechanical control. Graded end-to-end in yield-space P&L. *Provisionally weak* — the LLM
  PM does not clear the mechanical baseline on the duration pod (`decisions.md`, 2026-07-22).

A layer is called **fixable** only if the layer above it carries signal it fails to convert;
otherwise a shortfall is *inherited* from below or is a *genuine limit*. This rule is fixed
in advance and applied uniformly.

---

## Part 1 — Analyst layer

One LLM per driver writes a `DriverView` (direction, conviction, report). Graded against the
**feature-IC floor**: the best `|IC|` any measurement-only feature achieves on the same
driver, same window. An analyst that beats its floor is extracting; one below it is leaving
signal on the table (or, at negative IC, reasoning backwards).

### A1 — Text channel: none / cue / whole
- **Hypothesis.** FOMC text adds forward sign-correction at policy pivots that the numeric
  channel misses, even when analyst-layer IC looks flat.
- **Method.** Same driver, same window, vary only the text the analyst sees:
  `--text-mode none` (numbers only), `cue` (the driver-specific extract), `whole` (the full
  statement, `--text-doc`). Hold model, memory, seed fixed.
- **Command.** `python -m src.run_analyst_ic --driver <d> --text-mode {none,cue,whole} [--text-doc <path>]`
- **Metric.** Own-driver IC and its t; plus the end-to-end read (does text change the PM's
  sign at pivots?) via the PM run downstream.
- **Status.** *Provisional:* whole ≈ cue at the analyst layer, but text is essential
  end-to-end (sparse pivot sign-correction). Re-confirm on the frozen board.

### A2 — Input source (add news)
- **Hypothesis.** A driver whose signal is under-extracted (labor, curve_slope,
  inflation_expectations — see `first-final-results.md`) gains from a second text source
  beyond the FOMC statement.
- **Method.** Extend `--text-doc` to a news/transcript corpus for the fixable drivers only;
  compare against the FOMC-only arm. Keep the numeric channel identical.
- **Metric.** ΔIC vs the FOMC-only baseline, on the fixable drivers.
- **Status.** *Pending* — depends on a vendored news corpus on the frozen repo.

### A3 — Isolation × anonymization
- **Hypothesis.** The structural invariants (each analyst sees only its own driver's inputs;
  the PM cannot lean on analyst *identity*) cost nothing in signal — they are honesty, not a
  handicap.
- **Method.** Two knobs. *Isolation:* confirm each analyst's inputs are its driver's only
  (audited by the no-leak test, not a run knob). *Anonymization:* strip analyst identity at
  the PM via `--scramble-reports` / `--blind` and check the driver block is unchanged in
  distribution.
- **Command.** `python -m src.run_pm_ic --pod <p> --scramble-reports` and `--blind <driver>`
- **Metric.** Driver-block IC with vs without identity; a null result *confirms* the invariant.
- **Status.** *Provisional:* on front_end, blind ≥ full (identity was a crutch) — the fixes
  branch already narrows `front_end.yaml`'s `reads` on this finding.

### A4 — Model × memory
- **Hypothesis.** A larger model and/or a memory of past meetings improves extraction on the
  fixable drivers; on the random-walk driver (term_premium) neither helps — it is a limit.
- **Method.** Cross `--model {fable-5, sonnet, opus}` with `--memory {on, off}`, one cell per
  driver, same window.
- **Command.** `python -m src.run_analyst_ic --driver <d> --model <m> [--memory]`
- **Metric.** IC(t) per cell; a memory "win" must replicate on a fresh board to count
  (the earlier t=1.73 did not — it was board-specific; `decisions.md`).
- **Status.** *Provisional / mixed.* Re-run cleanly; report the fresh-board number only.

### A5 — Perturbation robustness (Tier-1 A/B/C)
- **Hypothesis.** The analyst's view moves for the right reasons: it *should* move when the
  measurement it cites changes (A), *should not* when an irrelevant input is jittered (B), and
  *should* degrade gracefully when an input is dropped (C).
- **Method.** `--perturb` on the analyst run, scored by `perturbation_bench` against the clean
  baseline.
- **Command.** `python -m src.run_analyst --driver <d> --perturb <A|B|C>` →
  `python -m src.layered.evaluation.perturbation_bench`
- **Metric.** Sensitivity (A), invariance (B), graceful degradation (C) — pass/fail per driver.
- **Status.** *Pending* on the frozen board.

---

## Part 2 — PM layer

The pod PM consumes the `DriverView` panel and emits an arbitrated driver block **and** a
`StrategyTrade`. Two graders, kept apart: `pm_bench` on the driver block (IC vs the analyst
it must improve on), `trade_pnl` on the trade (yield-space P&L, the PM→fund seam).

### B1 — Aggregation method: prompt vs mechanical vs relevance-weighted
- **Hypothesis.** *Relevance*-weighting (weight each driver by its trailing IC to the traded
  instrument, walk-forward) beats both conviction-weighting (what the LLM does freehand) and
  equal-weighting, because "predictable = priced = untradable" makes conviction and relevance
  roughly inverted.
- **Arms.**
  - *Prompt:* `python -m src.run_pm_ic --pod <p>` (LLM arbitrates and sizes freehand).
  - *Mechanical:* `python -m src.run_pm_mechanical --pod <p>` (fixed 0.5·own + 0.5·panel
    blend; the declared control the LLM must beat).
  - *Relevance/IC-weighted:* `run_pm_mechanical --relevance --weighting ic` — **requires
    PR #13** (`RelevancePM`); not on `main` yet.
- **Metric.** Driver-block IC (`pm_bench`, the `ic_mech` column) and trade P&L (`trade_pnl`),
  head-to-head on one clock.
- **Status.** *Provisional:* LLM does not beat mechanical on duration; relevance-weighting is
  the candidate alpha (PR #13). Re-run all three on the frozen board.

### B2 — LLM overlay (anchor-and-adjust)
- **Hypothesis.** The generative layer adds value only as a *bounded* adjustment to a
  mechanical anchor, not as a freehand sizer: `w_v1 = w_v0 · m_d`, `m_d ∈ [0.5, 2.0]`.
- **Arm.** `HybridPM` over the relevance baseline — **requires PR #13**.
- **Metric.** Trade P&L of v1 vs the v0 anchor; does the multiplier earn its variance?
- **Status.** *Provisional:* hybrid did not beat the mechanical anchor on the earlier board
  (the reports get diluted). Re-test whether the improved analyst reports (Part 1) change this.

### B3 — Structural trade layer (the curve seat)
- **Hypothesis.** Holding the LLM's *driver views* fixed and building the trade by *declared
  structure* recovers a slope trade the mechanical baseline must abstain on — the one place
  the structural layer does strictly more than the arithmetic control.
- **Method.** `structural_bench` re-scores an LLM PM run's trades through `structural_trade`,
  which reads `leg_roles: {front, long}` from the pod to split a scalar rate-axis projection
  into an equal-and-opposite 2s10s steepener/flattener.
- **Enabling change (this PR).** `curve.yaml` now declares `leg_roles: {front: DGS2, long:
  DGS10}`, so the curve pod — which `first-final-results.md` records as *"mechanical abstains
  → no baseline there"* — gets a structural trade instead of abstaining. `mechanical_pm` is
  deliberately left abstaining; the structural layer is the arm that fills the gap.
- **Metric.** `trade_pnl` on the restructured curve run vs the LLM's freehand legs.
- **Status.** Mechanism verified (`test_structural.py`); P&L *pending* on the frozen board.

### B4 — Disagreement as signal (Mod D)
- **Hypothesis.** Cross-analyst disagreement on a driver is itself informative (the
  Machine-Forecast-Disagreement read) — high panel dispersion flags where the consensus is
  fragile.
- **Method.** `disagreement_signal` on the board.
- **Metric.** IC of panel disagreement to forward outcomes; sign and t.
- **Status.** *Pending* on the frozen board.

### B5 — Individual analyst contribution: permutation / leave-one-out / Shapley
- **Hypothesis.** A small number of drivers carry the pod's edge; the rest are inert or
  anti-signal (`inflation_expectations` reasons backwards). We can attribute the pod's trade
  P&L to each analyst *exactly*, because the mechanical/relevance combiner is deterministic.
- **Method (three, agreeing where the combiner is linear).**
  - *Leave-one-out.* Drop analyst d, re-combine, re-score; ΔP&L is d's LOO contribution.
  - *Permutation.* Shuffle analyst d's views across dates (destroying its timing, keeping its
    marginal distribution); the drop in P&L is d's timing contribution. Repeat for a null band.
  - *Shapley.* Average d's marginal contribution over all analyst orderings — the fair
    attribution when contributions interact. Free here: the combiner is a deterministic
    function of the panel, so every coalition is one re-scoring, no re-running the LLM.
- **Metric.** Per-analyst P&L attribution with a permutation null band; rank the pod's drivers.
- **Status.** *Pending* — deterministic post-processing over a frozen board; cheap to run.

---

## Order of operations (validation run)

1. Rebuild the board on the frozen repo + Alfred data (`run_analyst_ic` per driver, chosen
   model/memory cell from A4).
2. Part 1 arms A1–A5 per driver → analyst-layer table vs the feature-IC floor.
3. PM arms B1–B3 per pod on that board (prompt, mechanical, relevance, hybrid, structural).
4. Post-hoc B4–B5 (disagreement, contribution) over the frozen PM runs — no new spend.
5. Fill the results tables in `first-final-results.md`; log any decision-rule triggers in
   `decisions.md`. Numbers here are superseded by that run.

## Dependency ledger

| arm | module / flag | on `main`? |
|---|---|---|
| A1–A2 text | `run_analyst_ic --text-mode/--text-doc` | yes |
| A3 anonymization | `run_pm_ic --scramble-reports/--blind` | yes (fixes) |
| A4 model×memory | `run_analyst_ic --model/--memory` | yes |
| A5 perturbation | `run_analyst --perturb` + `perturbation_bench` | yes (fixes) |
| B1 prompt/mechanical | `run_pm_ic`, `run_pm_mechanical` | yes |
| B1 relevance / B2 hybrid | `RelevancePM`/`HybridPM` (`--relevance --weighting`) | **PR #13** |
| B3 structural (curve) | `structural_bench` + `curve.yaml leg_roles` | yes (fixes + this PR) |
| B4 disagreement | `disagreement_signal` | yes (fixes) |
| B5 contribution | post-hoc over frozen runs | no code needed |
