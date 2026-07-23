"""Market + macro data layer (modification #1 infrastructure).

Three sources, one interface:
  * yfinance  — futures, ETFs, FX (free, no key)
  * FRED      — macro series (free key from fred.stlouisfed.org)
  * financialdatasets.ai — equity fundamentals (free tier: AAPL/MSFT/NVDA/...)
    -> for equities, port the fetchers from virattt/ai-hedge-fund src/tools/api.py
       (MIT license — keep the attribution notice).
"""
from __future__ import annotations

import os

import pandas as pd
import requests

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# FRED indexes each observation by its REFERENCE date, not its publication date.
# March CPI is stamped 2024-03-01 but isn't released until ~mid-April, so slicing
# `.loc[:asof]` in the backtest would let an agent see data weeks before the
# real world had it (look-ahead leak #2). We shift each series forward by its
# typical release lag so the value only becomes visible on/after its true
# publication date. (ALFRED vintage archives are the rigorous fix; this fixed
# per-series shift is the standard practical correction.) Daily market series
# (Treasury yields) publish same day -> lag 0 (default).
PUBLICATION_LAG_DAYS: dict[str, int] = {
    "CPIAUCSL": 14,   # monthly CPI, released ~2 weeks after the reference month
    "CPILFESL": 14,   # core CPI, same release as headline
    "PCEPILFE": 30,   # core PCE — the Fed's actual 2% target measure, released ~1 month later
    "UNRATE": 7,      # monthly jobs report, released ~1st Friday of following month
    "PAYEMS": 7,      # nonfarm payrolls, same release as UNRATE
    "NFCI": 7,        # weekly (Wed-dated) financial conditions, released the following week
    "WALCL": 2,       # weekly Fed balance sheet (H.4.1), Wed-dated, released next day
    # T10YIE / DFII10 are daily market series → publish same-day → default lag 0.
}

# The rigorous fix promised above: ALFRED's full revision history, queried with these
# two sentinel values (FRED's own documented convention for "every realtime period
# there has ever been"), returns one row per (reference date, realtime_start) — a new
# row each time a value was revised. `fetch_fred_vintage` reduces that to each
# observation's TRUE first-publication date, which `fred_local.load_series` prefers
# over the fixed lag table above wherever a series has been vendored into
# ``data/fred_vintage/`` (`scripts/fetch_fred_vintage.py`). A series not yet vendored
# there keeps using the fixed lag, unchanged — covering a series with real vintage
# data is additive, never a behavior change for the rest.
_ALFRED_REALTIME_START = "1776-07-04"
_ALFRED_REALTIME_END = "9999-12-31"


def _first_release_dates_from_observations(observations: list[dict]) -> "pd.Series":
    """Reduce a full ALFRED vintage history to each observation's first release.

    Every time a value is revised, ALFRED emits a new row for the same reference
    ``date`` with a later ``realtime_start``. The earliest ``realtime_start`` for a
    given ``date`` is the day that print actually became public — the fact the fixed
    per-series lag only approximates. Later revisions are deliberately discarded: this
    answers "when could an analyst have known this existed", not "what did it turn
    out to be" — pulled out as a pure function so it is testable without a network call.
    """
    first: dict[pd.Timestamp, pd.Timestamp] = {}
    for o in observations:
        if o.get("value") == ".":     # FRED's missing-observation sentinel
            continue
        d = pd.Timestamp(o["date"])
        rt = pd.Timestamp(o["realtime_start"])
        if d not in first or rt < first[d]:
            first[d] = rt
    return pd.Series(first, name="first_release_date").sort_index()


def fetch_prices(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    """Daily close prices via yfinance. Returns (date x symbol)."""
    import yfinance as yf  # lazy import
    data = yf.download(symbols, start=start, end=end, progress=False, auto_adjust=True)
    closes = data["Close"]
    if isinstance(closes, pd.Series):  # single symbol
        closes = closes.to_frame(symbols[0])
    return closes.dropna(how="all")


def fetch_fred(series_id: str, start: str, end: str, api_key: str | None = None) -> pd.Series:
    """One FRED series as a pd.Series indexed by date."""
    key = api_key or os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError("Set FRED_API_KEY (free at fred.stlouisfed.org)")
    r = requests.get(FRED_BASE, params={
        "series_id": series_id, "api_key": key, "file_type": "json",
        "observation_start": start, "observation_end": end,
    }, timeout=30)
    r.raise_for_status()
    obs = r.json()["observations"]
    s = pd.Series(
        {pd.Timestamp(o["date"]): float(o["value"]) for o in obs if o["value"] != "."},
        name=series_id,
    ).sort_index()
    # Move each value to its (approximate) publication date so `.loc[:asof]`
    # slicing downstream can't see data before it was actually released.
    lag = PUBLICATION_LAG_DAYS.get(series_id, 0)
    if lag:
        s.index = s.index + pd.Timedelta(days=lag)
    return s


def fetch_fred_vintage(series_id: str, start: str, end: str,
                       api_key: str | None = None) -> pd.Series:
    """Each observation's TRUE first-publication date, from ALFRED's full vintage
    history — the rigorous alternative to the fixed ``PUBLICATION_LAG_DAYS`` shift.

    Returns a ``pd.Series`` indexed by observation (reference) date, valued by the
    ``pd.Timestamp`` on which that observation first became public. This is what
    ``scripts/fetch_fred_vintage.py`` vendors into ``data/fred_vintage/``, which
    ``fred_local.load_series`` prefers over the fixed lag table whenever a series has
    been fetched here.

    Needs a ``FRED_API_KEY``; queried, not vendored automatically, exactly like
    ``fetch_fred``. The full-revision-history query is a larger payload than a plain
    observations call, so expect it to be slower per series.
    """
    key = api_key or os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError("Set FRED_API_KEY (free at fred.stlouisfed.org)")
    r = requests.get(FRED_BASE, params={
        "series_id": series_id, "api_key": key, "file_type": "json",
        "observation_start": start, "observation_end": end,
        "realtime_start": _ALFRED_REALTIME_START, "realtime_end": _ALFRED_REALTIME_END,
        "output_type": 2,
    }, timeout=60)
    r.raise_for_status()
    return _first_release_dates_from_observations(r.json()["observations"])


def fetch_macro_bundle(series_ids: list[str], start: str, end: str) -> dict[str, pd.Series]:
    return {sid: fetch_fred(sid, start, end) for sid in series_ids}


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all")
