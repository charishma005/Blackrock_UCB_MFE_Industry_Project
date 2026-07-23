"""Load a saved PM run (``run_pm_ic`` JSONL + its ``.meta.json``).

A separate loader from ``runs.load_run`` rather than a generalization of it, because
the two contracts genuinely differ: ``ArbitratedView`` carries ``drivers`` (a dict of
signed convictions, N per meeting) and has no ``direction``, ``conviction``, ``level``,
``degraded`` or ``carried``. Forcing one loader to serve both would mean either
widening the frozen contract or filling those fields with lies.

Everything the PM layer needs that the contract does not carry — whether the call
degraded, how stale the board was, how far the PM moved the panel — lives in the JSONL
*envelope*, which is why this reads the whole record rather than just ``["arbitrated"]``.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class PMRun:
    """One PM run, as scoreable frames."""

    path: str
    pod: str
    model: str
    meta: dict
    frame: pd.DataFrame        # asof × driver → PM signed conviction (NaN where absent)
    disagreement: pd.Series
    coverage: pd.Series
    degraded: pd.Series
    age: pd.DataFrame          # asof × driver → board age in days
    notes: pd.Series
    # asof → the pod's ``StrategyTrade`` as a dict, or None where it took no trade.
    # Kept raw rather than parsed into the contract because a run file is a record of
    # what happened, and a meeting whose trade grounding failed still needs a row here.
    # ``evaluation.trade_pnl.load_trades`` is what turns these into scoreable frames.
    trades: pd.Series

    @property
    def name(self) -> str:
        return f"{self.pod}:{self.model}" if self.model else self.pod

    @property
    def drivers(self) -> list[str]:
        return list(self.frame.columns)


def load_pm_run(path: str) -> PMRun:
    """Parse a PM run JSONL (+ sibling ``.meta.json``) into a :class:`PMRun`.

    Degraded meetings are dropped from ``frame`` on the same principle
    ``runs.load_run`` drops them for analysts: an abstention after a failed call is
    not a flat view, and scoring it as one would credit the model with a neutral call
    it never made.
    """
    with open(path) as fh:
        recs = [json.loads(line) for line in fh if line.strip()]
    if not recs:
        raise ValueError(f"{path}: no records")

    meta_path = os.path.splitext(path)[0] + ".meta.json"
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

    idx = pd.DatetimeIndex([pd.Timestamp(r["asof"]) for r in recs])
    drivers = list(meta.get("listens_to") or
                   sorted({d for r in recs for d in (r.get("arbitrated") or {}).get("drivers", {})}))

    rows, ages = [], []
    for r in recs:
        av = r.get("arbitrated") or {}
        conv = av.get("drivers") or {}
        rows.append({d: float(conv[d]) if d in conv else np.nan for d in drivers})
        board = r.get("board") or {}
        ages.append({d: (board.get(d) or {}).get("age_days", np.nan) for d in drivers})

    frame = pd.DataFrame(rows, index=idx).sort_index()
    age = pd.DataFrame(ages, index=idx).sort_index()
    degraded = pd.Series([bool(r.get("degraded", False)) for r in recs], index=idx).sort_index()
    disagreement = pd.Series(
        [float((r.get("arbitrated") or {}).get("disagreement", np.nan)) for r in recs],
        index=idx).sort_index()
    coverage = pd.Series([float(r.get("coverage", np.nan)) for r in recs], index=idx).sort_index()
    notes = pd.Series([str((r.get("arbitrated") or {}).get("notes", "")) for r in recs],
                      index=idx).sort_index()
    # ``dtype=object`` explicitly: a run in which no meeting produced a trade would
    # otherwise become an all-NaN float Series, and ``trades.notna()`` would then be the
    # only surviving evidence that the column ever held dicts.
    trades = pd.Series([(r.get("arbitrated") or {}).get("trade") for r in recs],
                       index=idx, dtype=object).sort_index()

    frame = frame.loc[~degraded]

    return PMRun(
        path=path,
        pod=meta.get("pod", ""),
        model=(meta.get("config") or {}).get("model", ""),
        meta=meta,
        frame=frame,
        disagreement=disagreement,
        coverage=coverage,
        degraded=degraded,
        age=age,
        notes=notes,
        trades=trades,
    )


def discover_pm_runs(reports_dir: str = "reports/pm", pattern: str = "*.jsonl") -> list[str]:
    """PM run JSONLs. Kept out of ``reports/`` root deliberately: ``discover_runs`` is
    a flat glob there and would hand a PM file to the analyst loader, which cannot
    validate it."""
    paths = sorted(glob.glob(os.path.join(reports_dir, pattern)))
    return [p for p in paths
            if not os.path.basename(p).startswith("_") and not p.endswith(".meta.json")]
