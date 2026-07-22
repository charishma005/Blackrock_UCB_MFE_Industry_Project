"""Vendored equity-driver feature series — no API key, no network.

The four equity analysts (`sector_breadth`, `vol_regime`, `positioning`,
`risk_appetite`) read features that the layered op vocabulary cannot express
(cross-sectional breadth, rolling-std vol, 52-week z-scores). Those are computed
upstream in the sibling **macro-llm** repo and vendored here as weekly series at
``data/equity/EQ_*.csv`` (see that directory's README for provenance).

One contract difference from ``fred_local``, and it matters: these series are
already indexed by their **decision Friday** — the date the value was knowable —
with the COT publication lag baked in upstream. So there is **no** release-date
shift on load here. ``AsOf.series`` then slices ``<= asof`` exactly as for FRED,
and the two together are the end-to-end no-lookahead guarantee.

    from src.data.equity_local import load_series, load_any_bundle
    vix = load_series("EQ_VIX")                 # already point-in-time

``load_any_bundle`` dispatches ``EQ_*`` ids here and everything else to
``fred_local``, so a mixed persona (equity + FRED inputs) still loads in one call.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

# .../src/data/equity_local.py -> parents[2] == repo root
_REPO_DIR = Path(__file__).resolve().parents[2] / "data" / "equity"

PREFIX = "EQ_"


def csv_dir() -> Path:
    env = os.environ.get("EQUITY_CSV_DIR")
    return Path(env) if env else _REPO_DIR


def available() -> list[str]:
    d = csv_dir()
    return sorted(p.stem for p in d.glob("*.csv")) if d.exists() else []


def load_series(series_id: str, start: str | None = None,
                end: str | None = None) -> pd.Series:
    """One vendored equity series, indexed by decision Friday. No lag shift —
    the point-in-time correction is already baked into the vendored data."""
    path = csv_dir() / f"{series_id}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{series_id}.csv not found in {csv_dir()}. Available: "
            f"{', '.join(available()) or '(none)'}. Point EQUITY_CSV_DIR at the "
            f"vendored equity directory, or regenerate it with macro-llm's "
            f"export_equity_series."
        )
    df = pd.read_csv(path)
    date_col, value_col = df.columns[0], df.columns[1]
    s = pd.Series(
        pd.to_numeric(df[value_col], errors="coerce").values,
        index=pd.to_datetime(df[date_col]),
        name=series_id,
    ).dropna().sort_index()
    if start is not None:
        s = s.loc[pd.Timestamp(start):]
    if end is not None:
        s = s.loc[:pd.Timestamp(end)]
    return s


def load_bundle(series_ids: list[str], start: str | None = None,
                end: str | None = None) -> dict[str, pd.Series]:
    return {sid: load_series(sid, start, end) for sid in series_ids}


def load_any_bundle(series_ids: list[str], start: str | None = None,
                    end: str | None = None) -> dict[str, pd.Series]:
    """Load a mixed bundle: ``EQ_*`` ids from here, all others from ``fred_local``.

    Lets a runner call ``load_any_bundle(list(analyst.inputs))`` without caring
    whether the analyst is an equity persona or a FRED-macro one.
    """
    from src.data import fred_local

    out: dict[str, pd.Series] = {}
    for sid in series_ids:
        if sid.startswith(PREFIX):
            out[sid] = load_series(sid, start, end)
        else:
            out[sid] = fred_local.load_series(sid, start, end)
    return out
