"""Is panel disagreement a signal? — the Machine-Forecast-Disagreement read (Mod D).

The PM already computes ``panel_disagreement`` per meeting (``pm/disagreement.py``) and
uses it only as a size-down flag. Bali-Kelly-Mörke-Rahman argue the *dispersion* across
heterogeneous forecasters is itself informative. This module tests that offline, over a
saved PM run — no model calls — with four questions:

1. **Uncertainty read.** Does disagreement at meeting *t* predict the realised magnitude
   of the panel's move to *t+1* (mean ``|Δ level|`` across drivers)? A yes makes it a
   volatility/uncertainty signal even if it says nothing about direction.
2. **PM accuracy conditioning.** Is the PM's own directional hit-rate and IC lower on
   high-disagreement meetings than low? If so, disagreement flags where to trust the PM
   less — the calibrated use of the number the pod mandate already gestures at.
3. **Trade the dispersion.** The signed IC of disagreement against the panel move,
   reported so its sign answers whether a split panel precedes reversion or continuation.
4. **Graph → disagreement.** Does the analyst ``missing_inputs`` dependency-graph density
   (how much evidence the panel says it lacks) predict a more split panel? MFD's
   structural predictor of *where* disagreement appears. Needs the analyst board dir.

It reuses ``pm_bench.driver_levels`` to rebuild outcomes on the PM's month-end clock, so
it imports ``layered.pm`` and — like ``pm_bench`` — is NOT re-exported from the package
``__init__``; import it by module path.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from src.layered.evaluation.ic import ICEvaluator, _two_sided_p
from src.layered.evaluation.pm_bench import driver_levels
from src.layered.evaluation.pm_runs import PMRun, load_pm_run
from src.layered.evaluation.runs import load_run
from src.layered.pm.board import ViewBoard


# ── a rank IC against an arbitrary target (not the shift-diff outcome) ───────────
def _rank_ic(signal: pd.Series, target: pd.Series, name: str = "") -> dict:
    """Spearman rank IC of ``signal`` against ``target``, aligned by index label.

    The same Pearson-on-ranks the ``ICEvaluator`` uses (scipy-free), but against a
    target that is not necessarily a next-release *change* — here a move magnitude or a
    graph density — so it lives here rather than being forced through ``evaluate``.
    """
    a = pd.concat([signal.rename("s"), target.rename("y")], axis=1).dropna()
    n = len(a)
    if n < 3 or a["s"].nunique() < 2 or a["y"].nunique() < 2:
        return {"name": name, "n": n, "ic": float("nan"), "t_stat": float("nan"),
                "p_approx": float("nan")}
    ic = float(a["s"].rank().corr(a["y"].rank()))
    t = ic * math.sqrt((n - 2) / (1.0 - ic * ic)) if abs(ic) < 1.0 else float("nan")
    return {"name": name, "n": n, "ic": ic, "t_stat": t, "p_approx": _two_sided_p(t)}


# ── outcomes on the PM clock ────────────────────────────────────────────────────
def _driver_moves(run: PMRun, steps: int, macro: Optional[dict]) -> pd.DataFrame:
    """Per-driver signed move to the next meeting, on the PM's own clock.

    One column per driver, indexed by meeting date, aligned to the run's (degraded
    dropped) frame index. Computed through the same ``AsOf``-gated ``FeaturePanel`` the
    analyst used, so the outcome is the quantity the driver was graded on.
    """
    dates = pd.DatetimeIndex(run.frame.index)
    drivers = run.drivers
    levels = driver_levels(drivers, dates, macro=macro)
    moves = {}
    for d in drivers:
        ev = ICEvaluator(levels[d].dropna(), steps=steps)
        moves[d] = ev.outcome
    return pd.DataFrame(moves).reindex(dates)


def panel_magnitude(run: PMRun, steps: int = 1, macro: Optional[dict] = None) -> pd.Series:
    """Mean absolute driver move to the next meeting — the panel's realised volatility."""
    return _driver_moves(run, steps, macro).abs().mean(axis=1, skipna=True)


# ── the four questions ──────────────────────────────────────────────────────────
def disagreement_vs_magnitude(run: PMRun, steps: int = 1,
                              macro: Optional[dict] = None) -> dict:
    """Q1/Q3: does disagreement predict the size of the next move? Signed IC, so the
    sign is informative on its own (a split panel preceding a big move is positive)."""
    disagree = run.disagreement.reindex(run.frame.index)
    return _rank_ic(disagree, panel_magnitude(run, steps, macro), "disagreement→|move|")


def _pm_pairs(run: PMRun, steps: int, macro: Optional[dict]) -> pd.DataFrame:
    """Long (meeting, driver) table of the PM's signed call, the realised move, and the
    meeting's disagreement — the substrate for the conditioning split."""
    moves = _driver_moves(run, steps, macro)
    disagree = run.disagreement.reindex(run.frame.index)
    rows = []
    for d in run.drivers:
        pair = pd.concat([run.frame[d].rename("pm"), moves[d].rename("move")],
                         axis=1).dropna()
        for ts, r in pair.iterrows():
            rows.append({"asof": ts, "driver": d, "pm": r["pm"], "move": r["move"],
                         "disagreement": float(disagree.get(ts, np.nan))})
    return pd.DataFrame(rows)


def pm_accuracy_by_disagreement(run: PMRun, steps: int = 1,
                                macro: Optional[dict] = None) -> pd.DataFrame:
    """Q2: pooled PM hit-rate and IC on low- vs high-disagreement meetings.

    A median split on the meeting's disagreement, pooling (driver, meeting) calls. If the
    PM is materially worse on the high-disagreement half, disagreement is a usable
    trust-discount signal.
    """
    pairs = _pm_pairs(run, steps, macro).dropna(subset=["disagreement"])
    nz = pairs[(pairs["pm"] != 0) & (pairs["move"] != 0)]
    out = []
    if len(nz) >= 6:
        med = nz["disagreement"].median()
        for label, sub in (("low_disagreement", nz[nz["disagreement"] <= med]),
                           ("high_disagreement", nz[nz["disagreement"] > med])):
            r = _rank_ic(sub["pm"], sub["move"], label)
            hit = float((np.sign(sub["pm"]) == np.sign(sub["move"])).mean()) if len(sub) else float("nan")
            out.append({**r, "hit_rate": hit})
    return pd.DataFrame(out).set_index("name") if out else pd.DataFrame()


def graph_density(board_dir: str, dates: pd.DatetimeIndex, suffix: str = "_on",
                  drivers: Optional[list[str]] = None) -> pd.Series:
    """Total ``missing_inputs`` edges the panel declares, as of each meeting date.

    Built from the analyst run files the board was assembled from, through the same
    as-of gate the PM reads them with, so density and disagreement land on one clock.
    """
    board = ViewBoard.from_dir(board_dir, suffix, drivers=drivers, check_identity=False)
    out = {}
    for ts in dates:
        m = board.at(ts)
        out[ts] = sum(len(e.view.missing_inputs) for e in m.entries.values() if e.present)
    return pd.Series(out).sort_index()


def disagreement_vs_graph(run: PMRun, board_dir: str, suffix: str = "_on") -> dict:
    """Q4: does a denser dependency graph predict a more split panel?"""
    dates = pd.DatetimeIndex(run.disagreement.index)
    density = graph_density(board_dir, dates, suffix, drivers=run.drivers)
    return _rank_ic(density, run.disagreement.reindex(density.index),
                    "graph_density→disagreement")


# ── aggregator + narration ──────────────────────────────────────────────────────
def evaluate_run(pm_path: str, board_dir: Optional[str] = None, board_suffix: str = "_on",
                 steps: int = 1, macro: Optional[dict] = None) -> dict:
    """Every question that the inputs allow, over one PM run. ``board_dir`` unlocks Q4."""
    run = load_pm_run(pm_path)
    result: dict = {
        "run": run.name,
        "pod": run.pod,
        "n_meetings": int(len(run.frame)),
        "disagreement_mean": float(run.disagreement.mean()),
        "magnitude": disagreement_vs_magnitude(run, steps, macro),
        "accuracy_split": pm_accuracy_by_disagreement(run, steps, macro),
    }
    if board_dir is not None:
        try:
            result["graph"] = disagreement_vs_graph(run, board_dir, board_suffix)
        except (FileNotFoundError, ValueError) as e:
            result["graph"] = {"error": f"{type(e).__name__}: {e}"}
    return result


def summarize(result: dict) -> str:
    """The honest reading. Breadth is ~12 meetings a year, so read the t-stat, not the
    third decimal, and treat any single IC below |t|≈2 as consistent with noise."""
    mag = result.get("magnitude") or {}
    lines = [
        f"{result.get('run', '?')}: {result.get('n_meetings', 0)} meetings, "
        f"mean disagreement {result.get('disagreement_mean', float('nan')):.3f}.",
        f"disagreement → next |move|: IC {mag.get('ic', float('nan')):+.3f} "
        f"(t {mag.get('t_stat', float('nan')):+.2f}, n {mag.get('n', 0)}).",
    ]
    split = result.get("accuracy_split")
    if isinstance(split, pd.DataFrame) and not split.empty:
        lo = split.loc["low_disagreement"] if "low_disagreement" in split.index else None
        hi = split.loc["high_disagreement"] if "high_disagreement" in split.index else None
        if lo is not None and hi is not None:
            lines.append(
                f"PM hit-rate low vs high disagreement: {lo['hit_rate']:.2f} vs "
                f"{hi['hit_rate']:.2f}; PM IC {lo['ic']:+.3f} vs {hi['ic']:+.3f}.")
    graph = result.get("graph")
    if isinstance(graph, dict) and "ic" in graph:
        lines.append(f"graph density → disagreement: IC {graph['ic']:+.3f} "
                     f"(t {graph.get('t_stat', float('nan')):+.2f}, n {graph.get('n', 0)}).")
    lines.append("At ~12 meetings/yr, read the t-statistic; a lone |t| < 2 is noise.")
    return "\n".join(lines)
