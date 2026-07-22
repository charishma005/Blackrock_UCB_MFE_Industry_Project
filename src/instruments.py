"""The tradeable universe + macro drivers — the instrument-space definition.

This module is the missing dependency the backtest engine imports
(``from src.instruments import DEFAULT_UNIVERSE, AssetClass, Instrument``). It is
deliberately tiny and declarative: it says *what exists*, never *what to do with
it*. Judgment (analysts), arbitration (PM pods), and sizing (ensemble/risk) all
live elsewhere and read this only to know the symbol set and its asset classes.

Two kinds of entries:
  * tradeable market instruments (equities, rates, commodity, FX) — priced via
    ``src.data.markets.fetch_prices`` (yfinance-style tickers / ETF proxies).
  * macro drivers (``tradeable=False``) — FRED series ids the analysts read.
    They are NOT positions; they are the raw feeds behind the driver views.

Keep this list the single source of truth for the symbol set so every layer
agrees on the universe by construction rather than by convention.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AssetClass(str, Enum):
    EQUITY = "equity"
    RATES = "rates"
    COMMODITY = "commodity"
    FX = "fx"
    MACRO = "macro"   # a FRED driver, not a position


@dataclass(frozen=True)
class Instrument:
    """One symbol in the universe.

    ``symbol``      yfinance ticker (tradeable) or FRED series id (macro).
    ``asset_class`` which bucket it belongs to.
    ``tradeable``   True for market instruments the book can hold; False for
                    macro drivers, which are read-only inputs to the analysts.
    ``name``        human label for reports (optional).
    """

    symbol: str
    asset_class: AssetClass
    tradeable: bool = True
    name: str = ""

    def __post_init__(self):
        # Macro drivers are inputs, never positions — enforce the invariant so a
        # FRED series can never accidentally be sized into the book.
        if self.asset_class == AssetClass.MACRO and self.tradeable:
            object.__setattr__(self, "tradeable", False)


# ── the default universe ─────────────────────────────────────────────────────
# Tradeable legs span the three asset classes the PM pods trade per the design
# note (bonds, equities, commodity), expressed as liquid ETF proxies so the
# backtest prices cleanly off free data. Extend this list to widen the universe;
# nothing downstream is hard-coded to these specific symbols.
DEFAULT_UNIVERSE: list[Instrument] = [
    # equities — broad + a couple of GICS sector proxies
    Instrument("SPY", AssetClass.EQUITY, name="S&P 500"),
    Instrument("QQQ", AssetClass.EQUITY, name="Nasdaq 100"),
    Instrument("XLF", AssetClass.EQUITY, name="Financials sector"),
    Instrument("XLE", AssetClass.EQUITY, name="Energy sector"),
    # rates / bonds
    Instrument("IEF", AssetClass.RATES, name="7-10y Treasuries"),
    Instrument("TLT", AssetClass.RATES, name="20y+ Treasuries"),
    # commodity
    Instrument("GLD", AssetClass.COMMODITY, name="Gold"),
    Instrument("DBC", AssetClass.COMMODITY, name="Broad commodities"),
    # fx
    Instrument("UUP", AssetClass.FX, name="US Dollar index"),

    # ── macro drivers (read-only FRED feeds behind the analyst views) ─────────
    Instrument("CPIAUCSL", AssetClass.MACRO, name="CPI"),
    Instrument("UNRATE", AssetClass.MACRO, name="Unemployment rate"),
    Instrument("WALCL", AssetClass.MACRO, name="Fed balance sheet"),
    Instrument("DGS2", AssetClass.MACRO, name="2y Treasury yield"),
    Instrument("DGS10", AssetClass.MACRO, name="10y Treasury yield"),
    Instrument("T10YIE", AssetClass.MACRO, name="10y breakeven inflation"),
    Instrument("NFCI", AssetClass.MACRO, name="Financial conditions index"),
]
