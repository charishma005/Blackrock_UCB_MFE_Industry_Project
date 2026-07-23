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
weeks before the world had it. Every series is therefore shifted to its release
date on load: to its TRUE first-publication date when a vendored ALFRED vintage
file exists for it (``data/fred_vintage/<id>.csv`` — see ``fred_vintage.py`` and
``scripts/fetch_fred_vintage.py``), else to the same fixed-lag approximation the
API path (``markets.PUBLICATION_LAG_DAYS``) uses, so the two sources still agree
wherever neither has vintage data.

    from src.data.fred_local import load_series, load_bundle
    cpi = load_series("CPIAUCSL")                  # already release-dated

Resolution order for the CSV directory: ``FRED_CSV_DIR`` if set, else the vendored
``data/fred/``, else the legacy sibling ``watching-crowding-build`` layout.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.data import fred_vintage
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
    """One FRED series, indexed by release date (not observation date).

    Prefers a vendored ALFRED vintage file for this series when one exists
    (``fred_vintage.available()``) — each observation gets its TRUE first-publication
    date rather than the fixed lag. Falls back to that fixed lag, unchanged, for any
    series not yet vendored there.
    """
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

    if series_id in fred_vintage.available():
        s = _release_date_from_vintage(s, series_id)
    else:
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


def _release_date_from_vintage(s: pd.Series, series_id: str) -> pd.Series:
    """Reindex an observation-dated series onto TRUE ALFRED release dates.

    Falls back to the fixed per-series lag only for an observation the vintage file
    does not cover (e.g. history predating the vintage file's own start) — partial
    coverage must never silently drop those rows or leave them observation-dated,
    either of which would be a bigger leak than the fixed-lag approximation this
    function exists to replace.

    ``s`` is sorted by observation date on entry. Real statistical agencies always
    publish in reference-period order, so the resulting release-date index should
    already be increasing; every rolling/diff/pct_change feature op in ``ops.py``
    reads this index positionally as "oldest -> newest reference period", so if the
    vendored vintage file ever produced an out-of-order result this refuses to
    silently re-sort the series — that would compute every feature over the wrong
    sequence — and raises instead.
    """
    release = fred_vintage.load_release_dates(series_id)
    aligned = release.reindex(s.index)
    if aligned.isna().any():
        lag_days = PUBLICATION_LAG_DAYS.get(series_id, 0)
        fallback = pd.Series(s.index + pd.Timedelta(days=lag_days), index=s.index)
        aligned = aligned.fillna(fallback)
    new_index = pd.DatetimeIndex(aligned)
    if not new_index.is_monotonic_increasing:
        raise ValueError(
            f"{series_id}: ALFRED release dates in data/fred_vintage/{series_id}.csv "
            f"are not increasing with observation date — refusing to reorder the "
            f"series, since every rolling/diff feature op depends on this index "
            f"staying in reference-period order. Inspect the vendored file."
        )
    out = s.copy()
    out.index = new_index
    return out


def load_bundle(series_ids: list[str], start: str | None = None,
                end: str | None = None) -> dict[str, pd.Series]:
    return {sid: load_series(sid, start, end) for sid in series_ids}
