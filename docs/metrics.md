# Metrics reference — every computation, surfaced

Exact formula → plain meaning → where computed, for every metric the pipeline consumes or
produces. Four buckets: **(1) input** (features the analyst reads), **(2) within-model**
(view encoding + PM combination), **(3) evaluation** (how results are scored), **(4) analyst
diagnostics** (the phase-1 faithfulness/correctness suite). Read the **hazards** at the end —
several metrics share a name but differ in computation.

Notation: $c_d\in[0,1]$ conviction, $\delta_d\in\{$up,down,flat$\}$ direction,
$\pi_d\in\{+1,-1\}$ polarity, $s_d=c_d\operatorname{sign}(\delta_d)$ signed conviction,
$o_d=\pi_d s_d$ oriented view, $y$ traded/level series, $\Delta y$ its forward change.

---

## 1 · Input metrics — the closed feature vocabulary
`src/layered/features/ops.py`. **Exactly 12 ops**; `apply()` rejects anything else. The
invariant: *no op fits a parameter, standardizes over the full sample, looks ahead, or scores
a direction* — so a feature spec **cannot express a forecast** (measurement, not signal).

| op | formula | category |
|---|---|---|
| `level` | $x_t$ (identity) | level |
| `diff(window)` | $x_t - x_{t-\text{window}}$ | change |
| `pct_change(window)` | $(x_t/x_{t-\text{window}}-1)\cdot100$ | change % |
| `yoy(periods=12)` | $(x_t/x_{t-\text{periods}}-1)\cdot100$ | change %, YoY |
| `pct_change_annualized(window,ppy=12)` | $\big((x_t/x_{t-w})^{ppy/w}-1\big)\cdot100$ | annualized change |
| `moving_average(window)` | $\frac1{w}\sum_{i=0}^{w-1}x_{t-i}$ | smoothed level |
| `spread` (arity 2) | $a_t - b_t$ (ffill-aligned) | two-series spread |
| `ratio` (arity 2) | $a_t / b_t$ (ffill; 0→NaN) | two-series ratio |
| `lag(periods≥1)` | $x_{t-\text{periods}}$ (negative forbidden) | base-effect |
| `distance_from_reference(reference)` | $x_t - \text{reference}$ (reference is a *stated constant*, never fitted) | distance-from-anchor |
| `rolling_min(window)` | $\min(x_{t-w+1..t})$ | range floor |
| `rolling_max(window)` | $\max(x_{t-w+1..t})$ | range ceiling |

Range-position (e.g. $(x-\min_W)/(\max_W-\min_W)$) is **composed** from `rolling_min/max`,
`spread`, `ratio` via `@name` refs — never a dedicated fitted op. No-lookahead enforced three
ways: `lag` forbids negative periods; `spread`/`ratio` only forward-fill; every source is read
through the `AsOf` gate that slices to $\le$ asof.

---

## 2 · Within-model metrics

### 2.1 Analyst view encoding — `src/layered/contracts.py`
- **signed conviction** `DriverView.signed_conviction` (L133): $s_d=\{$up$:+1,$down$:-1,$flat$:0\}\cdot c_d$.
- **polarity** $\pi_d$: not a view field — declared per driver in the pod YAML (default $+1$), surfaced via `MechanicalPM.polarity`.
- **oriented view** `disagreement.oriented` (L24): $o_d=\pi_d\,s_d$ for present drivers with declared polarity (a driver with no polarity is **skipped**, not defaulted).

### 2.2 Panel structure — `src/layered/pm/disagreement.py`
- **panel_disagreement** (L41): $1-\dfrac{|\sum_d o_d|}{\sum_d|o_d|}$ (0 = one-way, 1 = maximally split; all-flat → 0).
- **override** (L60): $\dfrac{1}{2n}\sum_d|\,\text{pm}_d-s_d\,|$ over jointly-present drivers (0 = PM restated the panel, large = overrode).

### 2.3 Mechanical / consensus combination — `mechanical_pm.py`, `pm_bench.py`
- **consensus_blend** (`pm_bench.py:85`, mirrored `mechanical_pm._driver_block:164`): per driver
  $\tfrac12 s_d + \tfrac12\,\pi_d\,\overline{o}$, where $\overline{o}=\frac1N\sum_j o_j$. Half own analyst, half the oriented panel mean. Constant 0.5, declared, unfitted.
- **rate projection** (`mechanical_pm._rate_projection:186`): $P_{\text{mech}}=\frac1N\sum_d o_d$ — **equal-weight** mean of oriented views ($>0$ = net upward yield pressure).
- **trade** (`mechanical_pm._trade:196`): sign $s=\operatorname{sign}(P)$; `same` pod → all legs $=s$, single/level → representative leg $=s$; **unit-gross normalized** ($w_i/\sum|w_i|$); conviction $=\min(1,|P|)$. `opposed` pods return no trade (abstain); $P=0$ → deliberate flat.

### 2.4 v0 relevance weighting — `src/layered/pm/relevance_pm.py`
Traded axis (`_axis_series:96`): duration $\tfrac12(y_{2y}+y_{10y})$, front_end $y_{2y}$,
real $y_{\text{10y BE}}$, opposed $y_{10y}-y_{2y}$. Outcome $\Delta y(s)=y(s{+}1)-y(s)$.
Relevance $\operatorname{rel}_d(t)=\operatorname{IC}(\{o_d(s)\},\{\Delta y(s)\})$ over $s<t$ (**walk-forward**; `fit:148` uses `past = order[:i]`, only outcomes realized before $t$). Weight vector by scheme (`_weights_for:105`):

| scheme | $w_d$ |
|---|---|
| `equal` | $1$ |
| **`ic`** (primary) | $\operatorname{rel}_d\cdot\frac{n}{n+k}$ (signed, shrunk; $k=8$; $0$ if $n<\text{min\_obs}=12$) |
| `ir` | $\operatorname{rel}_d\sqrt{n}\cdot\frac{n}{n+k}$ (IC t-stat, shrunk) |
| `rank_topk` | $\operatorname{sign}(\operatorname{rel}_d)$ for top-$k$ by $|\operatorname{rel}_d|$, else $0$ |
| `ridge` | $\beta=(X_s^\top X_s+\alpha I)^{-1}X_s^\top y$ on standardized views ($\alpha=5$); warm-up if $n<\max(12,2p)$ |

Projection (`_rate_projection:168`): $P=\dfrac{\sum_d w_d o_d}{\sum_d|w_d|}$; **warm-up fallback** to the equal-weight mean when all $w=0$ (so early meetings ≡ mechanical). `equal` reproduces mechanical byte-for-byte.

### 2.5 v1 hybrid multiplier — `src/layered/pm/hybrid_pm.py`
Baseline $=$ v0 weights (or equal in warm-up). LLM returns a per-driver multiplier clamped to
$[0.5,2.0]$. Adjusted weight (`_rate_projection:132`): $w_d^{\text{v1}}=w_d^{\text{v0}}\cdot m_d$;
same normalized projection. A failed LLM call falls back to pure v0 ($m_d=1$).

---

## 3 · Evaluation metrics — `src/layered/evaluation/`

### 3.1 IC family — `ic.py`
- **outcome** (L87): $\text{level}_{t+\text{steps}}-\text{level}_t$ (non-overlapping by construction).
- **IC** (L120): $\operatorname{corr}(\operatorname{rank}s,\operatorname{rank}y)$ — Spearman via rank-then-Pearson. NaN if $n<3$ or <2 unique values.
- **t_stat** (L121): $t=\operatorname{IC}\sqrt{\dfrac{n-2}{1-\operatorname{IC}^2}}$ (correlation t; honest because obs are non-overlapping).
- **p_approx** (L38): $\operatorname{erfc}(|t|/\sqrt2)$ — normal approx, two-sided.
- **releases_per_year** (L96): $365.25/\text{median gap-in-days}$. **breadth** (L104): $=\text{releases/yr}/\text{steps}$ (bets per year).
- **required_ic** (L49): $\text{IR}/\sqrt{\text{breadth}}$ — the bar (invert $\text{IR}\approx\text{IC}\sqrt{\text{breadth}}$).
- **hit_rate** (L129): fraction where $\operatorname{sign}(s)=\operatorname{sign}(y)$, zeros excluded. *(sign-agreement — see hazard.)*
- **ic_from_conviction** (`calibration_split:141`): $\operatorname{IC}(s)-\operatorname{IC}(\operatorname{sign}s)$ — extra ordering info the conviction magnitude adds over the sign alone.
- **signal_sharpe** (L161): $\sqrt{ppy}\cdot\text{mean}/\text{std}$ of the pseudo-P&L $s\cdot\text{outcome}$ (std $ddof{=}1$; **not** a tradable Sharpe).

### 3.2 Trade P&L — `trade_pnl.py`
- **yield_pnl** (L215): $\text{P\&L}_t=\sum_i w_i\,\Delta y_i$ in **pp of yield**; positive weight bets the yield **rises** (not a bond return; no duration weighting). Only meetings with a trade scored; missing forward obs → NaN, dropped (never a zero).
- **score_trades** (L239, guard $n<3$): mean $=\overline r$; std $ddof{=}1$; **t_stat** $=\overline r/(\text{std}/\sqrt n)$ (one-sample mean t); p_approx $=\operatorname{erfc}(|t|/\sqrt2)$; **hit_rate** $=\Pr(\text{P\&L}>0)$ *(fraction profitable — different from ic.py)*; **sharpe_ann** $=\text{mean}/\text{std}\cdot\sqrt{ppy}$, $ppy=365.25/\text{median gap}$; **ic_conviction** $=\operatorname{corr}(\operatorname{rank}c,\operatorname{rank}\text{P\&L})$ (did it size well).
- **trade_validity** (L287): sign_violation_rate, legs_dropped (universe/zero counts), flat_rate, emitted/grounded/rejected rates, mean gross/legs/conviction/rationale-words.
- **load_trades** (L109): per meeting — `net` $=\sum w_i$, `gross` $=\sum|w_i|$, `conviction`, `flat` (trade present, no legs), `emitted`/`has_trade`, per-instrument $w_i$.

### 3.3 PM-vs-analyst benchmark — `pm_bench.py:102`
Per driver, three ICs on the PM's month-end clock against the same driver-level outcome:
**ic_analyst** (the analyst view the PM saw), **ic_pm** (the PM's conviction), **ic_mech**
(the consensus_blend control); **d_ic** $=\text{ic\_pm}-\text{ic\_analyst}$; **t_pm**/**t_analyst**
the correlation t; plus hit rates, breadth, `ic_for_ir_1` $=\text{required\_ic}(1,\text{breadth})$.

---

## 4 · Analyst diagnostics (phase-1) — **in git history, not the current tree**
Computed by `src/layered/diagnostics.py`, **deleted** in `322e8b2`; last present at commit
`09adb04` (line refs below from `git show 09adb04:…`). The `reports/phase1_*.md` files are its
frozen output. The current tree's `evaluation/ic.py` is a newer rewrite and does **not** emit
these columns.

- **own_corr** (`faithfulness:160`): $\operatorname{corr}$(agent view$_d$, honest measurement$_d$) — does the view track its own driver (deterministic = 1 by construction).
- **cross_corr_max / _mean**: $\max$/mean of $|\operatorname{corr}(\text{view}_d,\text{measurement}_j)|$, $j\ne d$ — contamination; **most_contaminating** = the arg-max driver.
- **faithfulness** $=\text{own\_corr}-\text{cross\_corr\_max}$ (range $[-2,1]$, ~1 ideal).
- **correctness** (`correctness:227`, 63d horizon): realized $=$ level$_{t+h}-$level$_t$; **hit_rate** $=$ hits/scored; **info_score** $=$ mean of $(\text{call}\cdot c_t\cdot\operatorname{sign}(\text{realized}))$ (conviction-weighted signed accuracy); **persistence_hit** (last-move-continues baseline); **edge_vs_persistence** $=(\text{hits}-\text{base\_hits})/\text{scored}$; **edge_vs_random** $=\text{hit\_rate}-0.5$.
- **prescience** (`prescience:270`, leak test): **information_gain** $=\text{llm\_hit}-\text{det\_hit}$ (LLM edge over the provably-no-future deterministic run); **override_hit** = LLM accuracy where it disagreed with the mechanical call. Large positive on `--source fred` ⇒ possible training-cutoff leak.
- **agent_correlation**: mean $|$off-diagonal$|$ of the signed-conviction correlation matrix (ensemble redundancy).

---

## 5 · Hazards — same name, different computation (read before any A/B)
1. **`hit_rate` has two definitions.** `ic.py`/`pm_bench` = *sign-agreement* (signal vs outcome, zeros excluded). `trade_pnl.py` = *fraction of P&L > 0*. Not interchangeable.
2. **Two t-stats.** IC t $=\operatorname{IC}\sqrt{(n-2)/(1-\operatorname{IC}^2)}$ (ic.py); trade-P&L t $=\overline r/(\text{std}/\sqrt n)$ (trade_pnl.py). Both need **non-overlapping** obs — so memory-on (persistent positions) t is not comparable to memoryless (v0/mechanical) t.
3. **Sharpe is not a tradable-return Sharpe.** Yield-space $\sum w\Delta y$, no duration weighting / carry / costs; annualization $\sqrt{ppy}$ with $ppy\approx12$ inferred from the date gap.
4. **IC is Spearman (rank).** Everywhere — via `.rank().corr()`, not Pearson on levels.
5. **The phase-1 faithfulness/correctness suite is not in the current tree** (deleted `322e8b2`); to reproduce those columns on the current analysts, restore `diagnostics.py` from `09adb04` or port it onto `evaluation/`.
6. **`ic_conviction` appears in two places** with the same idea but different inputs: `ic.py` (ordering info of magnitude over sign, on the release clock) vs `trade_pnl.score_trades` (conviction vs realized P&L). Don't conflate.
