"""Real FRED series from local CSVs — no API key, no network.

The raw FRED CSVs are **vendored into this repo** at ``data/fred/*.csv`` (public-
domain data), so the analyst layer runs on real history the moment the repo is
cloned — no `FRED_API_KEY`, no sibling checkout. That matters here: the pipeline is
developed against true data rather than a fabricated regime, so what the analyst
reads is what a person would have read. ``scripts/fetch_fred.py`` refreshes the set
or adds a new series (that step, and only that step, needs a key).

One correctness note carries over from ``markets.fetch_fred``. These CSVs are
indexed by *observation* date — the month the data describes — not by the date it
was published. May CPI is stamped ``2026-05-01`` but was not released until mid
June, so slicing ``.loc[:asof]`` on the raw index would hand an analyst a print
weeks before the world had it. Every series is therefore shifted to its
approximate release date on load, reusing the same lag table as the API path so
the two sources agree.

    from src.data.fred_local import load_series, load_bundle
    cpi = load_series("CPIAUCSL")                  # already release-dated

Resolution order for the CSV directory: ``FRED_CSV_DIR`` if set, else the vendored
``data/fred/``, else the legacy sibling ``watching-crowding-build`` layout.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.data.markets import PUBLICATION_LAG_DAYS

# .../Blackrock_UCB_MFE_Industry_Project/src/data/fred_local.py -> parents[2] == repo root
_REPO_DIR = Path(__file__).resolve().parents[2] / "data" / "fred"
# Legacy fallback: the sibling FOMC repo the CSVs were first sourced from.
_SIBLING_DIR = (Path(__file__).resolve().parents[3]
                / "watching-crowding-build" / "FOMC" / "data" / "raw" / "fred")


def csv_dir() -> Path:
    env = os.environ.get("FRED_CSV_DIR")
    if env:
        return Path(env)
    if _REPO_DIR.exists():
        return _REPO_DIR
    return _SIBLING_DIR


def available() -> list[str]:
    """Series ids present on disk."""
    d = csv_dir()
    return sorted(p.stem for p in d.glob("*.csv")) if d.exists() else []


def load_series(series_id: str, start: str | None = None, end: str | None = None) -> pd.Series:
    """One FRED series, indexed by approximate *release* date (not observation date)."""
    path = csv_dir() / f"{series_id}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{series_id}.csv not found in {csv_dir()}. Available: {', '.join(available()) or '(none)'}. "
            f"Point FRED_CSV_DIR at the raw FRED directory, or use src.data.markets.fetch_fred with a key."
        )
    df = pd.read_csv(path)
    date_col, value_col = df.columns[0], df.columns[1]
    s = pd.Series(
        pd.to_numeric(df[value_col], errors="coerce").values,
        index=pd.to_datetime(df[date_col]),
        name=series_id,
    ).dropna().sort_index()

    # Shift observation date → release date so downstream `.loc[:asof]` slicing
    # cannot see a print before it existed.
    lag = PUBLICATION_LAG_DAYS.get(series_id, 0)
    if lag:
        s.index = s.index + pd.Timedelta(days=lag)

    if start is not None:
        s = s.loc[pd.Timestamp(start):]
    if end is not None:
        s = s.loc[:pd.Timestamp(end)]
    return s


def load_bundle(series_ids: list[str], start: str | None = None,
                end: str | None = None) -> dict[str, pd.Series]:
    return {sid: load_series(sid, start, end) for sid in series_ids}
