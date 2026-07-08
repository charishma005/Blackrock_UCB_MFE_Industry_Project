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
    "UNRATE": 7,      # monthly jobs report, released ~1st Friday of following month
    "PAYEMS": 7,      # nonfarm payrolls, same release as UNRATE
}


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


def fetch_macro_bundle(series_ids: list[str], start: str, end: str) -> dict[str, pd.Series]:
    return {sid: fetch_fred(sid, start, end) for sid in series_ids}


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all")
