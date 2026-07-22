"""Load a saved analyst run (``run_analyst_ic`` JSONL + its ``.meta.json``).

One place that turns a run file into scoreable objects — the signed-conviction and
level series the ``ICEvaluator`` grades, plus a per-meeting frame carrying the prose
report for the quality checks. Rebuilding the ``DriverView`` gives the signed signal
from the same ``signed_conviction`` property the live runner uses, so nothing here
re-hardcodes a direction→sign map. Reused by the evaluation notebook and by
``compare_sweep``.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.layered.contracts import DriverView


def _view_from(vd: dict) -> DriverView:
    """Rebuild a DriverView from its JSON dump. ``asof`` is coerced back to a
    Timestamp because the contract types it as one (``arbitrary_types_allowed`` means
    pydantic checks the instance rather than parsing the ISO string)."""
    vd = dict(vd)
    vd["asof"] = pd.Timestamp(vd["asof"])
    return DriverView.model_validate(vd)


@dataclass
class Run:
    """One evaluated run, keyed by driver + model."""

    path: str
    driver: str
    model: str
    meta: dict
    views: pd.DataFrame        # per-meeting rows, indexed by asof (all meetings)
    signed: pd.Series          # signed conviction, degraded dropped
    level: pd.Series           # driver level, degraded dropped

    @property
    def name(self) -> str:
        return f"{self.driver}:{self.model}" if self.model else self.driver


def load_run(path: str) -> Run:
    """Parse a run JSONL (+ sibling ``.meta.json``) into a :class:`Run`."""
    with open(path) as fh:
        recs = [json.loads(line) for line in fh if line.strip()]
    if not recs:
        raise ValueError(f"{path}: no records")

    meta_path = os.path.splitext(path)[0] + ".meta.json"
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

    views = [_view_from(r["view"]) for r in recs]
    idx = pd.DatetimeIndex([v.asof for v in views])
    df = pd.DataFrame({
        "direction": [v.direction for v in views],
        "conviction": [v.conviction for v in views],
        "signed": [v.signed_conviction for v in views],
        "level": [v.level if v.level is not None else np.nan for v in views],
        "degraded": [v.degraded for v in views],
        "carried": [v.carried for v in views],
        "report": [v.report or v.reasoning for v in views],
        "key_evidence": [list(v.key_evidence) for v in views],
        "falsifier": [v.falsifier for v in views],
    }, index=idx).sort_index()

    keep = ~df["degraded"]
    driver = meta.get("driver") or (views[0].driver if views else "")
    model = (meta.get("config") or {}).get("model", "")
    return Run(
        path=path, driver=driver, model=model, meta=meta, views=df,
        signed=df.loc[keep, "signed"], level=df.loc[keep, "level"],
    )


def discover_runs(reports_dir: str = "reports", pattern: str = "*.jsonl") -> list[str]:
    """Run JSONLs in ``reports_dir`` (skips ``_scratch`` files and ``.meta.json``)."""
    paths = sorted(glob.glob(os.path.join(reports_dir, pattern)))
    return [p for p in paths
            if not os.path.basename(p).startswith("_")
            and not p.endswith(".meta.json")]
