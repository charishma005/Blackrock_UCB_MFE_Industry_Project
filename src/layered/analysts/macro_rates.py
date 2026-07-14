"""Single-driver analysts feeding the macro rates strategy.

Four isolated experts, each watching exactly one force the thesis names as an
example driver — inflation, the labor market, the Fed's balance sheet, and a
single long-end point on the curve. None of them knows what a "flattener" is;
each only reports on its own driver. Turning these four views into a trade is
the PM's job (``src/layered/pm/macro_rates.py``).

Direction convention (a statement about the DRIVER, never an instrument):

    inflation        up = inflation accelerating
    labor_tightness  up = labor market tightening (unemployment falling)
    balance_sheet    up = balance sheet expanding (QE); down = runoff (QT)
    term_premium     up = long-end premium / 10y yield rising

The Phase-1 readings below are deliberately simple, transparent measurements —
the thesis commits to "no forecasting technique," so these are honest
placeholders that make the pipeline run end to end and can be swapped for
anything richer without touching the PM, exactly because the ``DriverView``
contract stays fixed. An optional LLM Phase-2 (see base class) can refine them.
"""
from __future__ import annotations

import numpy as np

from src.layered.analysts.base import SingleDriverAnalyst
from src.layered.contracts import DriverDirection, DriverView
from src.layered.timeline import AsOf

# FRED series the macro-rates analysts read. WALCL (Fed total assets) is not in
# the flat ensemble's DEFAULT_UNIVERSE, so the layered orchestrator fetches this
# bundle itself.
MACRO_RATES_SERIES: tuple[str, ...] = ("CPIAUCSL", "UNRATE", "WALCL", "DGS10", "DGS2")


def _direction(momentum: float, eps: float) -> DriverDirection:
    if momentum > eps:
        return "up"
    if momentum < -eps:
        return "down"
    return "flat"


def _conviction(magnitude: float, scale: float) -> float:
    """Squash a raw magnitude into [0, 1]; ``scale`` is the 'strong move' size."""
    if scale <= 0:
        return 0.0
    return float(np.clip(magnitude / scale, 0.0, 1.0))


class InflationAnalyst(SingleDriverAnalyst):
    driver = "inflation"
    inputs = ("CPIAUCSL",)

    def read(self, world: AsOf) -> DriverView:
        cpi = world.series("CPIAUCSL")
        if len(cpi) < 16:
            return DriverView(driver=self.driver, asof=world.asof, direction="flat",
                              conviction=0.0, horizon_days=self.horizon_days,
                              reasoning="insufficient CPI history")
        yoy = cpi.pct_change(12).dropna()
        current = float(yoy.iloc[-1])
        momentum = current - float(yoy.iloc[-4])         # 3-month change in YoY
        direction = _direction(momentum, eps=0.0005)
        # conviction blends how fast inflation is moving with how far it sits from
        # a ~2% target (a driver both accelerating and already high is a strong view).
        conv = 0.6 * _conviction(abs(momentum), 0.010) + 0.4 * _conviction(abs(current - 0.02), 0.02)
        return DriverView(
            driver=self.driver, asof=world.asof, direction=direction,
            conviction=round(conv, 3), horizon_days=self.horizon_days, level=round(current, 4),
            reasoning=f"CPI YoY {current:.1%}, 3m change {momentum:+.2%} → inflation {direction}",
        )


class LaborMarketAnalyst(SingleDriverAnalyst):
    driver = "labor_tightness"
    inputs = ("UNRATE",)

    def read(self, world: AsOf) -> DriverView:
        unrate = world.series("UNRATE")
        if len(unrate) < 13:
            return DriverView(driver=self.driver, asof=world.asof, direction="flat",
                              conviction=0.0, horizon_days=self.horizon_days,
                              reasoning="insufficient unemployment history")
        current = float(unrate.iloc[-1])
        change_3m = current - float(unrate.iloc[-4])      # falling unemployment = tightening
        low_12m = float(unrate.tail(12).min())
        # tightening when unemployment is falling; loosening when it is rising
        direction = _direction(-change_3m, eps=0.03)
        conv = _conviction(abs(change_3m), 0.4)
        # Sahm-rule spirit: unemployment well off its 12m low is a strong loosening view
        if current > low_12m + 0.5:
            direction, conv = "down", max(conv, 0.7)
        return DriverView(
            driver=self.driver, asof=world.asof, direction=direction,
            conviction=round(conv, 3), horizon_days=self.horizon_days, level=round(current, 2),
            reasoning=f"unemployment {current:.1f}% (3m {change_3m:+.1f}, 12m low {low_12m:.1f}) "
                      f"→ labor {'tightening' if direction=='up' else 'loosening' if direction=='down' else 'stable'}",
        )


class BalanceSheetAnalyst(SingleDriverAnalyst):
    driver = "balance_sheet"
    inputs = ("WALCL",)

    def read(self, world: AsOf) -> DriverView:
        walcl = world.series("WALCL")
        if len(walcl) < 14:
            return DriverView(driver=self.driver, asof=world.asof, direction="flat",
                              conviction=0.0, horizon_days=self.horizon_days,
                              reasoning="insufficient balance-sheet history")
        current = float(walcl.iloc[-1])
        # WALCL is weekly; ~13 obs ≈ one quarter. pct change over the quarter.
        prior = float(walcl.iloc[-13])
        change = (current / prior - 1.0) if prior else 0.0    # + = expanding, - = runoff
        direction = _direction(change, eps=0.002)
        conv = _conviction(abs(change), 0.03)                 # ~3%/qtr move = strong
        return DriverView(
            driver=self.driver, asof=world.asof, direction=direction,
            conviction=round(conv, 3), horizon_days=self.horizon_days, level=round(current, 1),
            reasoning=f"Fed assets {change:+.1%}/qtr → balance sheet "
                      f"{'expanding' if direction=='up' else 'in runoff (QT)' if direction=='down' else 'flat'}",
        )


class TermPremiumAnalyst(SingleDriverAnalyst):
    driver = "term_premium"
    inputs = ("DGS10",)

    def read(self, world: AsOf) -> DriverView:
        d10 = world.series("DGS10")
        if len(d10) < 30:
            return DriverView(driver=self.driver, asof=world.asof, direction="flat",
                              conviction=0.0, horizon_days=self.horizon_days,
                              reasoning="insufficient 10y history")
        current = float(d10.iloc[-1])
        # ~63 trading days ≈ a quarter of daily yields; change in the long-end level
        # is a transparent (crude) stand-in for a clean ACM term-premium series.
        window = d10.dropna()
        prior = float(window.iloc[-63]) if len(window) >= 63 else float(window.iloc[0])
        change = current - prior                              # in yield points
        direction = _direction(change, eps=0.05)
        conv = _conviction(abs(change), 0.75)                 # ~75bp/qtr move = strong
        return DriverView(
            driver=self.driver, asof=world.asof, direction=direction,
            conviction=round(conv, 3), horizon_days=self.horizon_days, level=round(current, 3),
            reasoning=f"10y {current:.2f}% ({change:+.2f} over ~1q) → long-end premium {direction}",
        )


def macro_rates_analysts(llm_client=None) -> list[SingleDriverAnalyst]:
    """The research pool the macro rates PM listens to."""
    return [
        InflationAnalyst(llm_client),
        LaborMarketAnalyst(llm_client),
        BalanceSheetAnalyst(llm_client),
        TermPremiumAnalyst(llm_client),
    ]
