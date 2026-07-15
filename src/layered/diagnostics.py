"""Phase-1 diagnostics for the analyst layer — deterministic vs LLM, side by side.

Four questions, each answered for BOTH the deterministic Phase-1 agents and the
LLM-refined agents, so you can see what the LLM adds (skill? correlation? a
future leak?) rather than just whether the plumbing runs.

  1. FAITHFULNESS   is the inflation agent really focused on inflation?
     - input isolation (structural): does it read only its own series, nothing
       past `asof`?  → ``input_isolation``
     - responsiveness vs contamination: does its view track its OWN driver's
       honest measurement and NOT the other drivers'?  → ``faithfulness``
     - on-topic reasoning (LLM): does the text stay about the driver?  (lexicon proxy)

  2. CORRECTNESS    is the view any good, and good relative to what?
     - horizon-aware hit rate + information score vs a persistence baseline
       and a random (0.5) baseline.  → ``correctness``

  3. LOOKAHEAD      could an agent have known the future?
     - data-slice audit via an access-logging probe (complements the unit test).
     - LLM training-cutoff *prescience*: information gain of the LLM over the
       deterministic baseline, which provably has no future info. Only
       meaningful on REAL data — on synthetic series the LLM cannot leak a
       future that was fabricated.  → ``prescience``

  4. CORRELATION    are the agents actually independent?
     - pairwise correlation of the analysts' signed-conviction streams; low
       off-diagonal = the isolation the thesis is buying.  → ``agent_correlation``

Everything is a pure function of collected view series, so the same code grades
deterministic and LLM runs identically. Runs offline; the LLM columns are simply
absent (n/a) when no client is supplied.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.layered.analysts.base import SingleDriverAnalyst
from src.layered.contracts import DriverView
from src.layered.timeline import AsOf


# ── data collection ─────────────────────────────────────────────────────────
@dataclass
class AnalystRun:
    """One pass of the analyst layer across a schedule of meetings."""

    label: str                                    # "deterministic" | "llm"
    signed: pd.DataFrame                           # (date x driver) signed conviction
    direction: pd.DataFrame                        # (date x driver) direction strings
    conviction: pd.DataFrame                       # (date x driver) conviction 0..1
    level: pd.DataFrame                            # (date x driver) measured level
    reasoning: dict[str, dict[pd.Timestamp, str]]  # driver -> date -> text
    views: dict[str, list[DriverView]]             # driver -> chronological views


def collect(
    analysts: list[SingleDriverAnalyst],
    macro: dict[str, pd.Series],
    prices: pd.DataFrame,
    dates: pd.DatetimeIndex,
    label: str,
) -> AnalystRun:
    """Run every analyst at every date (via the AsOf gate) and tabulate views."""
    signed, direction, conviction, level = {}, {}, {}, {}
    reasoning: dict[str, dict] = {}
    views: dict[str, list[DriverView]] = {}
    for asof in dates:
        world = AsOf(asof=asof, macro=macro, prices=prices)   # slices to <= asof
        for a in analysts:
            v = a.form_view(world)
            signed.setdefault(a.driver, {})[asof] = v.signed_conviction
            direction.setdefault(a.driver, {})[asof] = v.direction
            conviction.setdefault(a.driver, {})[asof] = v.conviction
            level.setdefault(a.driver, {})[asof] = v.level if v.level is not None else np.nan
            reasoning.setdefault(a.driver, {})[asof] = v.reasoning
            views.setdefault(a.driver, []).append(v)
    return AnalystRun(
        label=label,
        signed=pd.DataFrame(signed).sort_index(),
        direction=pd.DataFrame(direction).sort_index(),
        conviction=pd.DataFrame(conviction).sort_index(),
        level=pd.DataFrame(level).sort_index(),
        reasoning=reasoning,
        views=views,
    )


# ── 1a. faithfulness: input isolation (structural / runtime) ────────────────
class _AsOfProbe:
    """AsOf look-alike that records which series an analyst touches and the
    latest index it returns, so we can verify (a) the analyst reads only its
    declared inputs and (b) never a value dated after ``asof``."""

    def __init__(self, world: AsOf):
        self._w = world
        self.asof = world.asof
        self.accessed: set[str] = set()
        self.max_index: pd.Timestamp | None = None

    def _note(self, key: str, idx) -> None:
        self.accessed.add(key)
        if len(idx):
            m = pd.Timestamp(idx.max())
            self.max_index = m if self.max_index is None else max(self.max_index, m)

    def series(self, sid: str) -> pd.Series:
        s = self._w.series(sid)
        self._note(sid, s.index)
        return s

    def price(self, sym: str) -> pd.Series:
        s = self._w.price(sym)
        self._note(sym, s.index)
        return s

    def frame(self, symbols=None) -> pd.DataFrame:
        f = self._w.frame(symbols)
        self._note("|".join(f.columns), f.index)
        return f


def input_isolation(
    analysts: list[SingleDriverAnalyst],
    macro: dict[str, pd.Series],
    prices: pd.DataFrame,
    asof: pd.Timestamp,
) -> pd.DataFrame:
    """Per analyst: did it read only its declared inputs, and nothing past asof?"""
    world = AsOf(asof=asof, macro=macro, prices=prices)
    rows = []
    for a in analysts:
        probe = _AsOfProbe(world)
        a.read(probe)                                   # type: ignore[arg-type]  (duck-typed)
        declared = set(a.inputs)
        undeclared = probe.accessed - declared
        no_future = probe.max_index is None or probe.max_index <= asof
        rows.append({
            "analyst": a.driver,
            "declared_inputs": ",".join(sorted(declared)),
            "accessed": ",".join(sorted(probe.accessed)) or "(none)",
            "isolation_ok": len(undeclared) == 0,
            "undeclared_reads": ",".join(sorted(undeclared)) or "-",
            "no_lookahead": bool(no_future),
            "latest_index": None if probe.max_index is None else probe.max_index.date(),
        })
    return pd.DataFrame(rows).set_index("analyst")


# ── 1b. faithfulness: responsiveness vs contamination ───────────────────────
def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    aligned = pd.concat([a, b], axis=1).dropna()
    if len(aligned) < 5 or aligned.iloc[:, 0].nunique() <= 1 or aligned.iloc[:, 1].nunique() <= 1:
        return float("nan")
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))


def faithfulness(agent: AnalystRun, truth: AnalystRun) -> pd.DataFrame:
    """How much each agent view tracks its OWN driver vs the other drivers.

    ``truth`` is the deterministic run — the honest per-driver measurement.
    own_corr should be high (the view responds to its driver); cross_corr should
    be low (the view is not contaminated by other drivers). ``faithfulness`` =
    own_corr - max|cross_corr|, in [-2, 1]; ~1 is ideal. For a deterministic
    agent graded against itself own_corr = 1 by construction.
    """
    drivers = list(agent.signed.columns)
    rows = []
    for d in drivers:
        own = _safe_corr(agent.signed[d], truth.signed.get(d, pd.Series(dtype=float)))
        cross = {j: abs(_safe_corr(agent.signed[d], truth.signed[j])) for j in drivers if j != d}
        cross_vals = [c for c in cross.values() if c == c]           # drop NaN
        cross_max = max(cross_vals) if cross_vals else float("nan")
        cross_mean = float(np.mean(cross_vals)) if cross_vals else float("nan")
        worst = max(cross, key=lambda k: (cross[k] if cross[k] == cross[k] else -1), default="-")
        rows.append({
            "driver": d,
            "own_corr": round(own, 3) if own == own else float("nan"),
            "cross_corr_max": round(cross_max, 3) if cross_max == cross_max else float("nan"),
            "cross_corr_mean": round(cross_mean, 3) if cross_mean == cross_mean else float("nan"),
            "most_contaminating": worst,
            "faithfulness": round(own - cross_max, 3) if (own == own and cross_max == cross_max) else float("nan"),
        })
    return pd.DataFrame(rows).set_index("driver")


# lexicon proxy for on-topic reasoning (a cheap stand-in for an LLM judge)
_DRIVER_LEXICON: dict[str, list[str]] = {
    "inflation": ["inflation", "cpi", "price", "yoy", "disinflat", "deflation"],
    "labor_tightness": ["labor", "unemploy", "employ", "payroll", "jobs", "wage", "sahm"],
    "balance_sheet": ["balance sheet", "qt", "runoff", "reserve", "liquidity", "quantitative", "fed asset", "walcl"],
    "term_premium": ["term premium", "long end", "long-end", "10-year", "10y", "duration", "supply", "issuance"],
}
_TRADE_TERMS = ["flattener", "steepener", "2s10s", "curve", "overweight", "underweight",
                "long the", "short the", "position", "spread trade"]


def reasoning_on_topic(run: AnalystRun) -> pd.DataFrame:
    """Per driver: share of views whose reasoning stays on its own driver vs
    mentions ANOTHER driver's vocabulary or a trade (contamination). Lexicon
    proxy — coarse, but it flags an analyst drifting off its mandate."""
    rows = []
    for d, by_date in run.reasoning.items():
        texts = [t.lower() for t in by_date.values() if t]
        if not texts:
            rows.append({"driver": d, "on_topic_rate": float("nan"), "contamination_rate": float("nan"), "n": 0})
            continue
        own_terms = _DRIVER_LEXICON.get(d, [d])
        other_terms = [w for j, ws in _DRIVER_LEXICON.items() if j != d for w in ws] + _TRADE_TERMS
        on = sum(any(w in t for w in own_terms) for t in texts) / len(texts)
        contam = sum(any(w in t for w in other_terms) for t in texts) / len(texts)
        rows.append({"driver": d, "on_topic_rate": round(on, 3), "contamination_rate": round(contam, 3), "n": len(texts)})
    return pd.DataFrame(rows).set_index("driver")


# ── 2. correctness: horizon-aware, vs baselines ─────────────────────────────
def _realized_change(level: pd.Series, t: pd.Timestamp, horizon_days: int) -> float:
    """Change in the driver's measured level from t to the first meeting >= t+horizon."""
    future = level.loc[t + pd.Timedelta(days=horizon_days):].dropna()
    if future.empty or pd.isna(level.get(t, np.nan)):
        return float("nan")
    return float(future.iloc[0] - level.loc[t])


def correctness(agent: AnalystRun, truth_level: pd.DataFrame, horizon_days: int = 63) -> pd.DataFrame:
    """Grade each agent's direction calls against the driver's realized move over
    its horizon, and compare to a persistence baseline (predict the last move
    continues) and the 0.5 random line."""
    rows = []
    eps = 1e-9
    for d in agent.direction.columns:
        level = truth_level[d] if d in truth_level else pd.Series(dtype=float)
        dir_series = agent.direction[d].dropna()
        conv_series = agent.conviction[d]
        hits = base_hits = scored = 0
        info = []
        prev_level = None
        prev_change = 0.0
        for t in dir_series.index:
            realized = _realized_change(level, t, horizon_days)
            cur_level = level.get(t, np.nan)
            if not pd.isna(cur_level) and prev_level is not None:
                prev_change = cur_level - prev_level
            if not pd.isna(cur_level):
                prev_level = cur_level
            if pd.isna(realized) or abs(realized) <= eps:
                continue                                       # driver didn't move → ungradeable
            realized_dir = np.sign(realized)
            call = {"up": 1.0, "down": -1.0, "flat": 0.0}[dir_series[t]]
            base_call = np.sign(prev_change)                   # persistence baseline
            scored += 1
            hits += int(call == realized_dir and call != 0.0)
            base_hits += int(base_call == realized_dir and base_call != 0.0)
            info.append(call * conv_series.get(t, 0.0) * realized_dir)
        rows.append({
            "driver": d,
            "n": scored,
            "hit_rate": round(hits / scored, 3) if scored else float("nan"),
            "info_score": round(float(np.mean(info)), 3) if info else float("nan"),
            "persistence_hit": round(base_hits / scored, 3) if scored else float("nan"),
            "edge_vs_persistence": round((hits - base_hits) / scored, 3) if scored else float("nan"),
            "edge_vs_random": round(hits / scored - 0.5, 3) if scored else float("nan"),
        })
    return pd.DataFrame(rows).set_index("driver")


# ── 3. lookahead: LLM training-cutoff prescience ────────────────────────────
_LEAK_MIN_OVERRIDES = 10   # below this, override_hit is noise, not evidence
_GAIN_NOISE_BAND = 0.02    # |information_gain| within this ≈ no real difference


def _leak_verdict(gain: float, ov_n: int, ov_hit: float, source: str) -> str:
    """A computed verdict per driver — replaces the old fixed template that
    printed 'possible leak' on every row regardless of the numbers.

    A training-cutoff leak reveals itself as the LLM being *more accurate* than
    the no-future-info baseline. No gain ⇒ nothing to attribute to leakage.
    Even a real gain is only suggestive without a post-cutoff control slice."""
    if source != "fred":
        return "synthetic — leak not testable (no real future to memorize)"
    if gain != gain:                                    # NaN
        return "n/a"
    if gain <= _GAIN_NOISE_BAND:
        return "no gain → no leak signal"
    if ov_n >= _LEAK_MIN_OVERRIDES and ov_hit == ov_hit and ov_hit >= 0.6:
        return "gain + accurate overrides → possible leak; verify on a post-cutoff window"
    return "small gain, unconfirmed → needs a post-cutoff control window"


def prescience(
    det: pd.DataFrame,
    llm: pd.DataFrame | None,
    det_run: AnalystRun,
    llm_run: AnalystRun | None,
    truth_level: pd.DataFrame,
    horizon_days: int,
    source: str,
) -> pd.DataFrame:
    """Information gain of the LLM over the deterministic baseline, per driver.

    The deterministic agent provably has NO future information — it is a
    mechanical read of point-in-time data. So a large, systematic accuracy gain
    from the LLM is suspicious: on REAL history it can mean the model is drawing
    on training knowledge of what came after ``asof``. ``override_hit`` is the
    LLM's accuracy specifically on dates where it *disagreed* with the
    deterministic call — implausibly high override accuracy is the sharpest tell.

    On synthetic data the series are fabricated, so the LLM cannot have memorized
    their future; the ``verdict`` column says so. This is only a real leak test on
    ``--source fred`` — and only bites when the window spans the model's cutoff.
    """
    if llm is None or llm_run is None:
        return pd.DataFrame(columns=["driver", "det_hit", "llm_hit", "information_gain",
                                     "override_n", "override_hit", "verdict"]).set_index("driver")
    eps = 1e-9
    rows = []
    for d in det.index:
        det_hit = det.loc[d, "hit_rate"]
        llm_hit = llm.loc[d, "hit_rate"] if d in llm.index else float("nan")
        # override accuracy: dates where LLM direction != deterministic direction
        dd = det_run.direction[d] if d in det_run.direction else pd.Series(dtype=object)
        ld = llm_run.direction[d] if d in llm_run.direction else pd.Series(dtype=object)
        level = truth_level[d] if d in truth_level else pd.Series(dtype=float)
        ov_n = ov_hit = 0
        for t in ld.index.intersection(dd.index):
            if ld[t] == dd[t]:
                continue
            realized = _realized_change(level, t, horizon_days)
            if pd.isna(realized) or abs(realized) <= eps:
                continue
            call = {"up": 1.0, "down": -1.0, "flat": 0.0}[ld[t]]
            ov_n += 1
            ov_hit += int(call == np.sign(realized) and call != 0.0)
        gain = round(llm_hit - det_hit, 3) if (llm_hit == llm_hit and det_hit == det_hit) else float("nan")
        ov_hit_frac = round(ov_hit / ov_n, 3) if ov_n else float("nan")
        rows.append({
            "driver": d,
            "det_hit": det_hit,
            "llm_hit": llm_hit,
            "information_gain": gain,
            "override_n": ov_n,
            "override_hit": ov_hit_frac,
            "verdict": _leak_verdict(gain, ov_n, ov_hit_frac, source),
        })
    return pd.DataFrame(rows).set_index("driver")


# ── 4. correlation between agents ───────────────────────────────────────────
def agent_correlation(run: AnalystRun) -> tuple[pd.DataFrame, float]:
    """Pairwise correlation of the analysts' signed-conviction streams, plus the
    average absolute off-diagonal (one number: how redundant the ensemble is)."""
    corr = run.signed.corr()
    n = corr.shape[0]
    if n < 2:
        return corr, float("nan")
    mask = ~np.eye(n, dtype=bool)
    off = np.abs(corr.values[mask])
    off = off[~np.isnan(off)]
    return corr.round(3), (round(float(off.mean()), 3) if len(off) else float("nan"))


# ── orchestration ───────────────────────────────────────────────────────────
@dataclass
class DiagnosticsReport:
    source: str
    regime: str | None
    horizon_days: int
    dates: pd.DatetimeIndex
    has_llm: bool
    input_isolation: pd.DataFrame
    faithfulness_det: pd.DataFrame
    faithfulness_llm: pd.DataFrame | None
    reasoning_det: pd.DataFrame
    reasoning_llm: pd.DataFrame | None
    correctness_det: pd.DataFrame
    correctness_llm: pd.DataFrame | None
    prescience: pd.DataFrame
    corr_det: pd.DataFrame
    corr_llm: pd.DataFrame | None
    avg_offdiag_det: float
    avg_offdiag_llm: float
    meta: dict = field(default_factory=dict)


def run_diagnostics(
    make_analysts,
    macro: dict[str, pd.Series],
    prices: pd.DataFrame,
    start: str,
    end: str,
    freq: str = "W-FRI",
    horizon_days: int = 63,
    llm_client=None,
    source: str = "synthetic",
    regime: str | None = None,
) -> DiagnosticsReport:
    """Run all four diagnostics for the deterministic agents and (if a client is
    given) the LLM agents, on the same schedule and data.

    ``make_analysts(llm_client)`` builds a fresh analyst list — called once with
    None (deterministic) and once with the client (LLM) so the two runs don't
    share state.
    """
    targets = pd.date_range(start, end, freq=freq)
    dates = pd.DatetimeIndex(sorted({prices.index[prices.index <= d][-1]
                                     for d in targets if len(prices.index[prices.index <= d])}))

    det_run = collect(make_analysts(None), macro, prices, dates, "deterministic")
    llm_run = collect(make_analysts(llm_client), macro, prices, dates, "llm") if llm_client else None
    truth_level = det_run.level                       # objective per-driver measurements

    iso = input_isolation(make_analysts(None), macro, prices, dates[-1])
    faith_det = faithfulness(det_run, det_run)
    faith_llm = faithfulness(llm_run, det_run) if llm_run else None
    reason_det = reasoning_on_topic(det_run)
    reason_llm = reasoning_on_topic(llm_run) if llm_run else None
    corr_det = correctness(det_run, truth_level, horizon_days)
    corr_llm = correctness(llm_run, truth_level, horizon_days) if llm_run else None
    presc = prescience(corr_det, corr_llm, det_run, llm_run, truth_level, horizon_days, source)
    cmat_det, off_det = agent_correlation(det_run)
    cmat_llm, off_llm = agent_correlation(llm_run) if llm_run else (None, float("nan"))

    return DiagnosticsReport(
        source=source, regime=regime, horizon_days=horizon_days, dates=dates, has_llm=llm_run is not None,
        input_isolation=iso,
        faithfulness_det=faith_det, faithfulness_llm=faith_llm,
        reasoning_det=reason_det, reasoning_llm=reason_llm,
        correctness_det=corr_det, correctness_llm=corr_llm,
        prescience=presc,
        corr_det=cmat_det, corr_llm=cmat_llm,
        avg_offdiag_det=off_det, avg_offdiag_llm=off_llm,
        meta={"n_dates": len(dates)},
    )
