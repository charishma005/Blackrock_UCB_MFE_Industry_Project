"""Synthetic macro + price data so the layered fund runs with zero setup.

Real inputs come from FRED (CPIAUCSL, UNRATE, WALCL, DGS10, DGS2) and yfinance
(SHY, IEF); see ``src/data/markets.py``. But the whole architecture must be
demonstrable offline, with no keys and no network — for the smoke test and for
anyone reading the repo. This module fabricates internally-consistent series for
two textbook regimes:

    "hawkish"  inflation accelerating, labor tightening, balance sheet in runoff,
               front-end yields rising faster than the long end  → a 2s10s FLATTENER
    "dovish"   the mirror image                                  → a 2s10s STEEPENER

Bond prices are derived from the synthetic yields via ``ΔP ≈ -duration × Δy`` so
returns are consistent with the curve the analysts read — the fund layer's vol
targeting then sees a coherent covariance, not noise. Deterministic (seeded).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Approximate effective durations for the ETF proxies (years) — matches the PM
# persona's instrument block so synthetic prices and the trade agree.
_DUR = {"SHY": 1.9, "IEF": 7.5}


def generate(
    start: str = "2022-01-01",
    end: str = "2024-12-31",
    regime: str = "hawkish",
    seed: int = 0,
) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    """Return ``(macro_bundle, prices_frame)`` for the requested regime."""
    rng = np.random.default_rng(seed)
    s = -1.0 if regime == "dovish" else 1.0   # sign of the whole regime

    bdays = pd.bdate_range(start, end)
    months = pd.date_range(start, end, freq="MS")
    weeks = pd.date_range(start, end, freq="W-WED")
    n_days, n_mo = len(bdays), len(months)
    t_mo = np.linspace(0.0, 1.0, n_mo)
    t_day = np.linspace(0.0, 1.0, n_days)

    # ── CPI: index whose YoY drifts from ~3% toward ~5% (hawkish) ───────────
    yoy = 0.03 + s * 0.02 * t_mo                          # target YoY path
    monthly = (1.0 + yoy) ** (1.0 / 12.0) - 1.0
    cpi_vals = 300.0 * np.cumprod(1.0 + monthly + rng.normal(0, 0.0004, n_mo))
    cpi = pd.Series(cpi_vals, index=months, name="CPIAUCSL")

    # ── Unemployment: falling = tightening (hawkish) ────────────────────────
    unrate = pd.Series(4.2 - s * 0.6 * t_mo + rng.normal(0, 0.03, n_mo),
                       index=months, name="UNRATE").round(2)

    # ── Fed balance sheet: shrinking = runoff/QT (hawkish) ──────────────────
    walcl = pd.Series(8.5e6 * (1.0 - s * 0.10 * np.linspace(0, 1, len(weeks)))
                      + rng.normal(0, 5e3, len(weeks)),
                      index=weeks, name="WALCL").round(1)

    # ── Yields: front end rises faster than long end (hawkish → flattening) ──
    d2 = 4.0 + s * 0.8 * t_day + rng.normal(0, 0.01, n_days)     # 2y
    d10 = 4.0 + s * 0.3 * t_day + rng.normal(0, 0.01, n_days)    # 10y (moves less)
    dgs2 = pd.Series(d2, index=bdays, name="DGS2").round(3)
    dgs10 = pd.Series(d10, index=bdays, name="DGS10").round(3)

    # ── Bond prices from yields: ΔP ≈ -duration × Δy (per 1.0 = 100bp) ──────
    def price_from_yield(yld: pd.Series, dur: float) -> pd.Series:
        dprice = -dur * yld.diff().fillna(0.0) / 100.0 * 100.0  # yields already in %
        return (100.0 + dprice.cumsum()).rename(None)

    shy = price_from_yield(dgs2, _DUR["SHY"])
    ief = price_from_yield(dgs10, _DUR["IEF"])
    prices = pd.DataFrame({"SHY": shy.values, "IEF": ief.values}, index=bdays)

    macro = {"CPIAUCSL": cpi, "UNRATE": unrate, "WALCL": walcl, "DGS10": dgs10, "DGS2": dgs2}
    return macro, prices
