"""Vendored ALFRED vintage release dates — each observation's TRUE first-publication
date, where available, as the rigorous alternative to ``fred_local``'s fixed
per-series lag shift.

``markets.PUBLICATION_LAG_DAYS`` is a declared, constant approximation ("CPI
publishes ~14 days after its reference month"), stated honestly as such rather than
hidden. It is wrong whenever a real release lands later than assumed — a government
shutdown, a schedule change — and because it is a silent assumption rather than a
measured fact, a wrong lag would show up as an IC that is a little better than
reality, never as a crash or a failing test. That made it the one soft spot in an
otherwise mechanically enforced no-look-ahead guarantee (``timeline.AsOf``,
``pm.board.ViewBoard``).

This module is the fix. ``scripts/fetch_fred_vintage.py`` (needs a ``FRED_API_KEY``,
run separately — not part of the default offline path) queries ALFRED's full revision
history via ``markets.fetch_fred_vintage`` and vendors each series' TRUE first-release
date per observation into ``data/fred_vintage/<series>.csv``. ``fred_local.load_series``
prefers this file when it exists for a series and falls back to the fixed lag,
unchanged, for any series not yet vendored here — so covering a series is additive and
never a behavior change for the rest.

    from src.data.fred_vintage import load_release_dates
    dates = load_release_dates("CPIAUCSL")   # observation date -> true release date
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

# .../Blackrock_UCB_MFE_Industry_Project/src/data/fred_vintage.py -> parents[2] == repo root
_REPO_DIR = Path(__file__).resolve().parents[2] / "data" / "fred_vintage"


def csv_dir() -> Path:
    env = os.environ.get("FRED_VINTAGE_CSV_DIR")
    return Path(env) if env else _REPO_DIR


def available() -> set[str]:
    """Series ids with a vendored true-release-date file on disk.

    A set, not a sorted list like ``fred_local.available()`` — this is consulted as a
    membership test on every ``load_series`` call, not browsed.
    """
    d = csv_dir()
    return {p.stem for p in d.glob("*.csv")} if d.exists() else set()


def load_release_dates(series_id: str) -> pd.Series:
    """One series' TRUE first-release date per observation date, sorted.

    Raises if the file is missing — callers are expected to check ``available()``
    first (``fred_local.load_series`` does), so reaching this without the file on
    disk is a caller bug, not a normal fallback path.
    """
    path = csv_dir() / f"{series_id}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{series_id}.csv not found in {csv_dir()}. Available: "
            f"{', '.join(sorted(available())) or '(none)'}. Run "
            f"scripts/fetch_fred_vintage.py {series_id} with a FRED_API_KEY, or check "
            f"fred_local.load_series's fixed-lag fallback is what you meant to use."
        )
    df = pd.read_csv(path)
    s = pd.Series(
        pd.to_datetime(df["first_release_date"]).values,
        index=pd.to_datetime(df["observation_date"]),
        name=series_id,
    ).sort_index()
    return s
