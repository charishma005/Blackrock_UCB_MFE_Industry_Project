# A layered LLM macro fund: signal generation vs. signal combination
### Scientific write-up and readiness assessment
*Consolidated record, phase 2. Companion references: `docs/metrics.md` (every formula),
`docs/experiments.md` (every experiment), `docs/decisions.md` (the ADR log), and
`notebooks/pm_trade_evaluation.ipynb` (the evidence, §7).*

---

## Abstract

We study a layered architecture in which specialist LLM "analysts" each read one macro
driver's measurements and central-bank language and emit a scoreable directional view, and a
portfolio-manager (PM) layer combines those views into a rates trade. Across a large,
controlled experiment matrix on US-rate pods (2016–2025, month-end clock), we find: (i) the
LLM PM's elaborate reasoning is **not load-bearing** — a numbers-only PM matches it and the
trade is ~75% invariant to scrambling which analyst said what; (ii) the PM (and the mechanical
consensus) weight analysts by **conviction**, which is approximately **inverted** from an
analyst's information coefficient (IC) to the *traded* instrument — "predictable ⇒ priced ⇒
untradable"; (iii) replacing conviction-weighting with a deterministic, walk-forward
**relevance weighting** (weight each analyst by its trailing IC to the traded move) roughly
doubles trade P&L over a mechanical baseline and is the first configuration to reach a
$t$-statistic near 2; (iv) an LLM adjustment layer on top of this mechanical combiner adds
nothing (v1 ≈ v0); and (v) the analysts' central-bank text is nonetheless **essential** — its
value is realised not in marginal own-driver accuracy but as a *sparse, forward-looking sign
correction* at policy pivots that is decisive after combination. The design thesis: **use the
LLM for signal generation, arithmetic for signal combination.** We present the full spec,
metrics, results, threats to validity, and an honest assessment of the distance to an
investment-grade claim.

---

## 1. The question

An LLM fund's only justification over a quantitative model is judgment — reading unstructured
policy language into a view. The question is *where* that judgment earns its keep: in
generating each analyst's directional call, in combining calls into a trade, or nowhere. We
test each seam with controlled ablations rather than asserting the architecture.

## 2. System specification (the shippable spec)

The pipeline is `data → analyst → seam → PM → trade`, governed by three structural invariants:
**no look-ahead** (every series is release-dated and read through an `AsOf` gate),
**measurement-not-signal** (features come from a closed 12-op vocabulary that cannot express a
forecast; `docs/metrics.md §1`), and **input isolation** (each analyst sees only its driver's
evidence). The frozen, evidence-backed configuration is:

| layer | spec | rejected alternative (evidence) |
|---|---|---|
| analyst text | driver-partitioned `cue` | `none` (loses signal), `whole` (dilutes it) — §F |
| seam | `DriverView → StrategyTrade` | — |
| PM combination | **relevance weight** (walk-forward trailing trade-IC) | conviction weight (§C3), LLM PM (§B4) |
| weighting scheme | `ic` (signed, shrunk) | ridge (overfits), equal (= mechanical) — §D1 |
| PM reasoning | mechanical (deterministic) | LLM prose (not load-bearing, §B4), hybrid multiplier (v1 ≈ v0, §E1) |
| PM memory | memoryless | memory-on (inflates the $t$-stat via overlapping positions) |

**Design thesis.** The LLM's contribution is *signal generation* at the analyst layer (where
partitioned text produces a better directional call); *combination* is precise arithmetic
(weight by demonstrated trade relevance) at which the LLM is worse than a formula. Every
rejected component violates this split.

## 3. Methods

All grading is on a non-overlapping monthly clock. Core metrics (full formulas in
`docs/metrics.md`): **IC** = Spearman rank correlation of a signal with the realised move;
$t=\text{IC}\sqrt{(n-2)/(1-\text{IC}^2)}$; **relevance** $\operatorname{rel}_d$ = an analyst's
walk-forward trailing IC to the *traded instrument* (vs. its own-driver IC); **trade P&L** =
$\sum_i w_i \Delta y_i$ in yield space (positive weight bets the yield rises); **Sharpe** =
$\sqrt{ppy}\cdot\text{mean}/\text{std}$, $ppy\approx12$. The v0 relevance weight is
$w_d=\operatorname{rel}_d\cdot n/(n+k)$ (shrunk), computed at each meeting only from outcomes
realised strictly earlier.

## 4. Principal results

- **Reasoning is not load-bearing (§B3–B4).** Numbers-only PM ≈ full-reports PM; scrambling
  report↔driver leaves the duration trade 74.5% unchanged; the prose the PM emphasises matches
  the driver it actually weights only ~30% of the time.
- **Confidence ≠ relevance (§C3, §C4).** own-driver IC and trade-IC are ~inverted
  (balance_sheet 0.69 vs. ≈0; labor 0.155 vs. +0.19/+0.23). The PM's weights track the own-IC
  diagonal ($r\approx+0.8$) and point away from the trade-IC one ($r\approx0$).
- **Mechanical relevance weighting works (§D1).** `ic` roughly doubles mean trade P&L over the
  mechanical baseline on all three trading pods and reaches $t\approx2$ (real $t=2.02$) — the
  first arm in the study to do so; `equal` reproduces mechanical exactly; `ridge` overfits.
- **The LLM adds nothing to combination (§E1).** v1 (mechanical baseline + bounded LLM
  multiplier) ≈ v0; slightly worse on one pod.
- **Text is essential but sparse (§F).** Analyst-layer own-IC is flat across none/cue/whole,
  yet end-to-end `cue > none > whole`: stripping text turns the duration trade negative.
  Mechanism: cue and none trades agree on 99/120 meetings; at the 21 where they disagree, cue
  is right 62% vs. 33%, and those meetings carry the entire P&L gap. The text is a *sparse
  forward sign-correction* at Fed pivots (most often the inflation call), invisible to marginal
  IC, decisive after combination. `whole` fails by *dilution*, not contamination (cross-driver
  correlation is unchanged).

## 5. Threats to validity (rigor audit)

Ranked by falsification risk. The deterministic comparisons (v0 vs. equal-weight) are airtight;
every LLM-bearing comparison carries at least one caveat.

1. **Single-draw LLM arms.** Each board (cue/none/whole), the LLM PM arms, and v1 are one
   sample; no replicate. The text finding and v1≈v0 most need replication.
2. **No out-of-sample holdout.** The study is in-sample 2016–2025; v0 is walk-forward (no
   look-ahead) but the design (pods, polarities, scheme) was chosen knowing the period.
3. **Multiple comparisons.** Dozens of arms; the $t\approx2$ winner is selected. `ic` was
   pre-registered as primary to mitigate.
4. **$t$-stat comparability.** Memory-on (overlapping positions) vs. memoryless $t$-stats are
   not directly comparable.
5. **Training-cutoff leak.** The analyst board is LLM output with a training cutoff; the
   walk-forward weighting is clean but the *views* may be memorised pre-2024 (§A3 shows no
   post-2024 collapse — mild evidence against, not closure).
6. **Sequential generation.** Boards/arms were produced at different times; one was regenerated
   mid-session. A final claim requires one pinned batch.

## 6. Readiness — distance to a credible presentation to experienced BlackRock PMs

The honest answer separates **two claims** at very different readiness.

**Claim A — the diagnosis and design ("weight by relevance, not confidence; LLM generates,
arithmetic combines").** *Near-ready.* It is economically grounded (predictable ⇒ priced ⇒
untradable is a proposition a rates PM already believes), mechanistically traced (the
confidence↔relevance inversion; the text sign-correction at pivots), supported by clean
deterministic controls, and — crucially — **bounded by negative results** (LLM PM, prose,
hybrid, whole-text all rejected with evidence). Experienced PMs reward exactly this: a falsified
hypothesis and a mechanism, not a black-box Sharpe. This is presentable now as a **research /
design contribution**. What a PM would still press: it resembles IC-momentum weighting (a known
technique) and lives on a tiny rate universe — so novelty and scale need framing.

**Claim B — the performance ("beats the mechanical rule at $t\approx2$").** *Not ready.* An
experienced desk will discount it in one glance, and correctly:
- $t\approx2$ on ~120 monthly observations, single draw, is inside the noise a careful PM
  expects from a research backtest;
- there is **no out-of-sample** period — design and evaluation share the window;
- the "P&L" is **not a return** — yield-space $\sum w\Delta y$ with no transaction costs, no
  carry, no duration weighting, no volatility targeting, no capacity/liquidity;
- **researcher degrees of freedom** (five schemes, dozens of arms) are uncorrected;
- the pre-2024 board is not leak-proof at the analyst layer.
None of these are fatal, but together they place the alpha claim below the bar at which a
BlackRock PM would allocate risk.

**Verdict.** We are ~80% of the way to a credible *scientific/design* presentation and ~30% of
the way to a credible *performance* claim. The correct move with experienced PMs is to present
Claim A as the deliverable — a rigorous, honestly-bounded design finding — with Claim B framed
explicitly as *"here is the promising signal and exactly what would make it investable,"* not
as an alpha assertion. Overclaiming at $t=2$ would forfeit credibility instantly; candidly
owning the gap is what earns it.

### The gap-closing roadmap (what turns Claim B credible)
1. **Replicates** — 3–5 LLM draws per board/arm (seed/temperature); report mean ± spread; the
   text edge and v0 must survive.
2. **Holdout** — design on 2016–2022, validate untouched on 2023–2025, then roll forward on
   live post-cutoff data (which also addresses the leak).
3. **Realistic P&L** — duration-weight the legs, add a cost/carry model, volatility-target, and
   report a return series with Sharpe / information ratio / max drawdown.
4. **Multiple-comparison discipline** — disclose the full arm count; bootstrap the $t$-stat.
5. **Breadth / scale** — extend the (pod-agnostic) v0 combiner to the team's international and
   equity analysts by declaring their pods; higher breadth is both what a multi-asset PM thinks
   in and what mechanically lifts the $t$-statistic (IR $\approx$ IC$\sqrt{\text{breadth}}$).
6. **One pinned, reproducible batch** on the merged codebase.

## 7. Convergence toward the team pipeline

The team has advanced **breadth** (the analyst layer grew from 7 US-rate personas to 18,
adding EA/UK/JP rates+equity and more macro drivers, plus international data and ECB/BoJ/BoE
corpora); this work advances **combination** (the deterministic relevance combiner). They are
complementary: v0 is pod-agnostic and is the natural combiner for the growing, unevenly-relevant
panel — but the PM currently declares only the four US-rate pods, so the new analysts feed no
trade. Convergence = (a) rebase on the team's current main and confirm the new modules compose;
(b) split the monolithic PR into a lean code PR and a separate evidence artifact; (c) land v0 as
the combiner and declare pods for the new analyst families; (d) regenerate everything once,
rigorously, on the merged tree (which is also roadmap item 6).

## 8. Conclusion

The layered LLM fund's alpha, to the extent it exists here, is **arithmetic**: weighting a panel
of analysts by demonstrated trade relevance. The LLM earns its cost upstream, in reading policy
language into a directional call — a sparse but decisive contribution — and not in the
combination, where a formula wins. This is a clean, transferable design principle and a set of
honestly-bounded findings. It is a credible *scientific* story today; it is not yet an
*investable* one, and the distance between the two is measured by replicates, a holdout, a real
return series, and scale — not by a larger claim.
