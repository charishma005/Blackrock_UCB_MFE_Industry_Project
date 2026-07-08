"""Instrument abstraction — replaces the original repo's bare `tickers: list[str]`.

Every tradeable (or observable) thing in the system is an Instrument with an
asset_class. Agents declare which asset classes they cover; the orchestrator
routes only relevant instruments to each agent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class AssetClass(str, Enum):
    EQUITY = "equity"
    FIXED_INCOME = "fixed_income"
    COMMODITY = "commodity"
    FX = "fx"
    MACRO = "macro"  # non-tradeable observables (CPI, curve) used for regime signals


@dataclass(frozen=True)
class Instrument:
    symbol: str            # e.g. "AAPL", "TLT", "USO", "GLD", "DGS10" (FRED)
    asset_class: AssetClass
    name: str = ""
    data_source: str = "yfinance"  # "financialdatasets" | "yfinance" | "fred"
    tradeable: bool = True
    meta: dict = field(default_factory=dict)


# ── Default universe ────────────────────────────────────────────────────────
# Equities keep the financialdatasets free tier; everything else is yfinance/FRED.
DEFAULT_UNIVERSE: list[Instrument] = [
    # Equities (financialdatasets free tickers)
    Instrument("AAPL", AssetClass.EQUITY, "Apple", "financialdatasets"),
    Instrument("MSFT", AssetClass.EQUITY, "Microsoft", "financialdatasets"),
    Instrument("NVDA", AssetClass.EQUITY, "NVIDIA", "financialdatasets"),
    # Fixed income (ETF proxies — clean daily prices, no futures roll headaches)
    Instrument("TLT", AssetClass.FIXED_INCOME, "20+yr Treasuries"),
    Instrument("IEF", AssetClass.FIXED_INCOME, "7-10yr Treasuries"),
    Instrument("LQD", AssetClass.FIXED_INCOME, "IG Corporate Credit"),
    Instrument("HYG", AssetClass.FIXED_INCOME, "High Yield Credit"),
    # Commodities (ETF proxies, NOT front-month futures: GC=F/CL=F/HG=F are
    # continuous front-month series whose returns are dominated by roll yield —
    # not actually investable as modeled. GLD/USO/CPER are tradeable funds whose
    # daily prices already embed roll cost, making the sleeve honest.)
    Instrument("GLD", AssetClass.COMMODITY, "Gold (SPDR Gold Shares)"),
    Instrument("USO", AssetClass.COMMODITY, "WTI Crude (US Oil Fund)"),
    Instrument("CPER", AssetClass.COMMODITY, "Copper (US Copper Index Fund)"),
    # Macro observables (FRED series — not traded; consumed as regime inputs)
    Instrument("DGS10", AssetClass.MACRO, "10yr Treasury Yield", "fred", tradeable=False),
    Instrument("DGS2", AssetClass.MACRO, "2yr Treasury Yield", "fred", tradeable=False),
    Instrument("CPIAUCSL", AssetClass.MACRO, "CPI", "fred", tradeable=False),
    Instrument("UNRATE", AssetClass.MACRO, "Unemployment", "fred", tradeable=False),
]


def filter_universe(universe: list[Instrument], classes: set[AssetClass]) -> list[Instrument]:
    return [i for i in universe if i.asset_class in classes]
