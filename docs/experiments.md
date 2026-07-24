# Experiment registry

Every experiment run, with a two-line description, the result, and (where relevant) the
apples-to-apples caveat. Evidence lives in `notebooks/pm_trade_evaluation.ipynb` (§ refs);
metric definitions in `docs/metrics.md`; decisions in `docs/decisions.md`.

**Read the caveats.** Clean = deterministic on a fixed board, only one thing varies. Most
LLM arms are **single-draw** (no replicate) and the whole study is **in-sample 2016–2025
with no holdout** — see the rigor notes at the bottom.

---

## A · Analyst layer

### A1 — Feature-IC floor (free, offline)
*Is each driver predictable at all from measurements available at time t? Grades the closed
feature vocabulary against the next-release move — the ceiling any analyst could reach.*
- **Result:** strong — balance_sheet |IC| 0.69, inflation 0.62, labor 0.43, curve 0.31, financial_conditions 0.39, inflation_expectations 0.26; **term_premium 0.14 (t≈1.5, insignificant → a genuine limit).**
- Clean (deterministic).

### A2 — LLM analyst IC (7 drivers, sonnet, cue board)
*Does each LLM analyst's signed conviction predict its own driver's next move?*
- **Result:** 3 strong — balance_sheet 0.690 (t 10.3), inflation 0.415 (4.9), financial_conditions 0.413 (4.9); 2 marginal — labor 0.155 (1.7), curve 0.119 (1.3); **2 dead/anti — inflation_expectations −0.079, term_premium −0.012.**
- Caveat: single board draw; LLM training-cutoff leak uncontrolled (affects absolute IC).

### A3 — Rolling 24-month IC (§7.9)
*Is analyst skill stable, or does it decay post-2024 (a training-cutoff tell)?*
- **Result:** **no uniform post-2024 collapse** — strong legs flat-to-improving; inflation_expectations degrades into negative (+0.06→−0.20); labor/term soften. Mild evidence *against* a severe leak.

---

## B · PM control matrix — fresh board (§7.2)

### B1 — LLM PM: memory-on vs memory-off vs mechanical (4 pods)
*Does an LLM PM reading the whole board beat a memoryless arithmetic rule?*
- **Result (mean pp / t):** duration mem_on +0.0055/0.83, mem_off +0.0072/0.44, **mech +0.0105/0.49**; front_end mem_on −0.0010, mem_off +0.0190/1.54, mech +0.0163/0.72; real mem_on +0.0007, mech +0.0129/1.08. **The prior memory-on "win" (t=1.73) did NOT replicate; no full-arbitration arm beats mechanical on any pod.**
- Caveat: memory-on holds persistent positions → overlapping obs → its t is **not** comparable to the memoryless arms.

### B2 — Blind arm (PM shown one report), short window
*Does seeing the whole panel add anything over a single analyst? (removes arbitration)*
- **Result:** front_end blind **+0.0269, t=2.73** — the only t>2 in the matrix, and it **beats** the full arm. Elsewhere blind is negative (real −0.0104, curve −0.0042). → on front_end the panel *dilutes*.
- Caveat: n=48 (2022–25) vs full arms n=120; compare only to the window-matched `mem_on@short` row.

### B3 — Scramble arm (report↔driver labels shuffled)
*Does the PM's trade actually depend on reading which analyst said what? (faithfulness)*
- **Result:** duration trade **74.5% direction-unchanged**, net-exposure corr +0.46 → the arbitration barely depends on correct attribution.

### B4 — Numbers-only PM (report prose stripped) (§7.11)
*Is the report prose load-bearing, or does the PM decide from the (direction, conviction) numbers alone?*
- **Result:** **numbers-only ≈ full** on P&L (duration −0.0086 vs −0.0050; front_end +0.0122 vs +0.0011; real −0.0034 vs −0.0068) and identical own-IC weighting (corr 0.8–1.0). → **the prose is not load-bearing.**

### B5 — Relevance-prior PM (mandate: conviction ≠ market impact) (§7.11)
*Can telling the PM about relevance fix the confidence-weighting mis-weight by prompt alone?*
- **Result:** helps the two directional pods — duration +0.019 (t 0.91), **front_end +0.030 (t 1.68, best of any arm)** — and rotated duration's corr(weight, trade-IC) from −0.10 to +0.10. But pooled it only moves the fit from anti (−0.22) to flat (−0.02); own-IC tracking unchanged (r≈+0.8). → partial, weak; the fix must be mechanical.
- Caveat: single LLM draw; ~48 obs per pod.

---

## C · Interpretability

### C1 — What did they lean on? (§7.4)
*Which analyst does the PM actually weight most?*
- **Result:** weights by conviction → **balance_sheet is loudest in 52–62%** of meetings (duration 59, front_end 52, curve 62); real leans on inflation (59%). Stated-vs-revealed: the prose emphasizes the driver it actually weights most only **~30%** of the time.

### C2 — Input sensitivity, leave-one-out (§7.8)
*Which input does the trade actually depend on? (drop each driver, measure the flip)*
- **Result:** dropping **balance_sheet flips the trade 27% (duration) / 36% (front_end) / 37% (curve)** — the most of any driver; inflation for real. Confirms C1 causally.

### C3 — Confidence vs relevance (§7.10)
*Own-driver IC (drives the PM's weighting) vs IC to the traded yields (what should).*
- **Result:** **~inverted.** balance_sheet own-IC 0.69 but trade-IC ≈ 0/negative (−0.08/−0.10); labor own-IC 0.155 but trade-IC **+0.19/+0.23** (most trade-relevant). "Predictable == priced == untradable."

### C4 — Weight-vs-IC diagonal (§7.11b)
*Does the PM's weight track own-driver IC or trade-IC?*
- **Result:** welded to the **own-IC** diagonal (pooled r +0.80, identical across mem_on/numbers-only/relevance) and off the **trade-IC** one (r ≈ −0.02 to −0.22). The mis-weighting, drawn.

---

## D · v0 — mechanical relevance combiner (§7.12)

### D1 — Weighting sweep (equal / ic / ir / rank_topk / ridge), 3 trading pods
*Weight each analyst by its walk-forward trailing IC to the traded instrument instead of by conviction — which scheme?*
- **Result (mean pp / t):** duration mech +0.0105/0.49, **ic +0.0215/1.01, rank_topk +0.0412/1.98**, ridge −0.0057; front_end **ic +0.0343/1.52**, rank_topk +0.0218, ridge +0.0071; real **ic +0.0242/2.02, rank_topk +0.0227/2.09**, ridge +0.0219. **ic roughly doubles mechanical and clears t≈2 for the first time in the study.**
- Sanity: `equal` reproduces mechanical byte-for-byte; `ridge` overfits at N=5/T≈120 (worst).
- Clean (deterministic on the cue board). Caveat: 5-scheme search → multiple comparisons; **ic pre-registered** as primary to avoid cherry-picking rank_topk.

---

## E · v1 — mechanical + LLM hybrid (§7.13)

### E1 — Hybrid vs v0 (LLM bounded multiplier on the v0 baseline weights)
*Once the weighting is fixed mechanically, does a report-justified LLM multiplier in [0.5, 2] add anything?*
- **Result:** **v1 ≈ v0.** duration v0 +0.0215 → v1 +0.0243; front_end v0 +0.0343 → v1 **+0.0285 (worse)**; real v0 +0.0242/2.02 → v1 +0.0246/2.04. On average slightly worse. → the LLM adjustment is inert; ship v0.
- Caveat: single LLM draw per pod.

---

## F · Text channel

### F1 — Text A/B: none / cue / whole, analyst layer (§7.14)
*Does the FOMC text change the analyst's view? none = numbers only, cue = driver-partitioned text (shipped), whole = entire statement.*
- **Result (mean):** own-IC none 0.266 / cue 0.264 / whole 0.266 (flat); |trade-IC| vs one axis none 0.097 / cue 0.102 / whole 0.087 (flat). **Looks inert — but this metric is a misleading single-axis magnitude proxy (see F2).**

### F2 — Text end-to-end: each board through v0 (§7.14)
*Does a text-richer board make a better TRADE after v0 combination?*
- **Result:** **cue > none > whole.** duration cue +0.0215 / none −0.0035 / whole −0.0068; front_end +0.0343 / +0.0004 / −0.0072; real +0.0242 / +0.0229 / +0.0162. **Text is essential** — stripping it makes duration negative, front_end zero — and un-partitioning (whole) is worse on every pod.
- Caveat: **one LLM run per board** (cue is the pre-existing draw) — the single biggest rigor gap; the finding needs replicates.

### F3 — Mechanism trace (duration, cue vs none)
*How does the text give downstream insight?*
- **Result:** cue and none trades agree on 99/120 meetings; they **disagree at 21**, and there cue is right **62% vs none 33%** — those 21 carry the **entire** P&L gap (axis P&L cue +1.50 vs none −1.50). The text most changes the **inflation** call, at Fed pivots (e.g. 2021 "look-through-inflation" → flat, correctly). → text = a **sparse forward sign-correction**, invisible to marginal IC, decisive for the trade.

### F4 — Why whole fails (convergence vs dilution)
*Does whole hurt by making analysts converge/contaminate?*
- **Result:** **No convergence** — cross-driver |corr| none 0.159 / cue 0.173 / **whole 0.145** (lowest), panel diversity unchanged. Whole vs cue disagree at 26/120; there cue 62% vs whole **31%** (same failure profile as none). → whole fails by **dilution** (driver signal drowned in the full doc), not contamination. Corrects the earlier "contamination" framing.

---

## Prior phase (pre-redesign, `reports/phase1_*`, `slides/…phase1`)

### P1 — Input-modality A/B: vector / text / textvec (4 analysts, 63d, 2019–24)
*Numbers only vs text only vs both — the "give it both" experiment.*
- **Result:** text-only **breaks faithfulness** (inflation own_corr 0.965→0.469, faithfulness −0.171); text helps policy-driven drivers on info_score (balance_sheet 0.603→0.698, term_premium −0.004→0.062), hurts data-driven (inflation 0.127→0.035); **textvec (both) is the sweet spot** — faithfulness restored, signal kept. Deck slide: *"the number anchors, the words inform."*
- Caveat: old 4-analyst design, different metric/horizon than the current study.

---

## Cross-cutting rigor caveats (see `docs/decisions.md` for the full audit)

1. **Single-draw LLM arms** — every board (cue/none/whole), the LLM PM arms, and v1 are one draw; no replicate. The text finding (F2) and v1≈v0 (E1) most need replication.
2. **No out-of-sample holdout** — the whole study is 2016–2025; v0 is walk-forward (no look-ahead) but the *design* (pods, polarities, ic scheme) was chosen knowing the period.
3. **Multiple comparisons** — dozens of arms; the t≈2 winner is selected from many. `ic` pre-registered as the primary.
4. **t-stat comparability** — memory-on (overlapping positions) vs memoryless (v0/mechanical) t-stats are not directly comparable.
5. **Window/pod matching** — blind/scramble are n=48; never compare to full-window n=120 except via the `@short` rows. Cross-pod means mix different instruments (curve abstains).
6. **Sequential generation** — boards/arms were generated at different times (one board was regenerated mid-session); a final deliverable should regenerate everything in one pinned batch.
