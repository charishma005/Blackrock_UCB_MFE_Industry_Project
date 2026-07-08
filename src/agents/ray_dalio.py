"""Ray Dalio macro regime agent (modification #1 — first non-equity agent).

Phase 1 computes a growth x inflation regime from FRED series plus market
confirmations (curve slope, credit spreads via HYG/LQD ratio, gold trend).
Phase 2 lets the persona-conditioned LLM translate the regime into per-
instrument signals for fixed income and commodities.

This agent also exposes `regime()` so the Portfolio Manager can consume the
macro view as context even for instruments Dalio doesn't directly signal.
"""
from __future__ import annotations

import pandas as pd

from src.agents.base import BaseInvestorAgent
from src.instruments import AssetClass, Instrument


class RayDalioAgent(BaseInvestorAgent):
    name = "ray_dalio"
    covers = {AssetClass.FIXED_INCOME, AssetClass.COMMODITY}

    def __init__(self, llm_client=None, macro_data: dict[str, pd.Series] | None = None,
                 prices: pd.DataFrame | None = None):
        super().__init__(llm_client)
        self.macro = macro_data or {}   # {"DGS10": series, "DGS2": ..., "CPIAUCSL": ..., "UNRATE": ...}
        self.prices = prices            # (date x symbol) for market confirmations

    # ── Regime detection (pure Python) ──────────────────────────────────
    def regime(self) -> dict:
        out: dict = {}

        cpi = self.macro.get("CPIAUCSL")
        if cpi is not None and len(cpi) > 13:
            yoy = cpi.pct_change(12).dropna()
            out["inflation_yoy"] = round(float(yoy.iloc[-1]), 4)
            out["inflation_direction"] = "rising" if yoy.iloc[-1] > yoy.iloc[-4] else "falling"

        unrate = self.macro.get("UNRATE")
        if unrate is not None and len(unrate) > 7:
            out["unemployment"] = float(unrate.iloc[-1])
            # rising unemployment ≈ slowing growth (Sahm-rule spirit)
            out["growth_direction"] = "slowing" if unrate.iloc[-1] > unrate.tail(12).min() + 0.3 else "expanding"

        d10, d2 = self.macro.get("DGS10"), self.macro.get("DGS2")
        if d10 is not None and d2 is not None:
            slope = float(d10.iloc[-1] - d2.iloc[-1])
            out["curve_slope_10s2s"] = round(slope, 3)
            out["curve_inverted"] = slope < 0

        if self.prices is not None:
            if {"HYG", "LQD"} <= set(self.prices.columns):
                ratio = (self.prices["HYG"] / self.prices["LQD"]).dropna()
                if len(ratio) > 63:
                    out["credit_stress"] = bool(ratio.iloc[-1] < ratio.tail(63).mean() * 0.98)
            if "GLD" in self.prices.columns:
                gold = self.prices["GLD"].dropna()
                if len(gold) > 63:
                    out["gold_3m_trend"] = round(float(gold.iloc[-1] / gold.iloc[-63] - 1), 4)

        # Four-quadrant label
        g = out.get("growth_direction"), out.get("inflation_direction")
        quadrant = {
            ("expanding", "rising"): "growth+ inflation+ (overheating)",
            ("expanding", "falling"): "growth+ inflation- (goldilocks)",
            ("slowing", "rising"): "growth- inflation+ (stagflation)",
            ("slowing", "falling"): "growth- inflation- (deflationary slowdown)",
        }.get(g, "indeterminate")
        out["quadrant"] = quadrant
        return out

    # ── BaseInvestorAgent interface ──────────────────────────────────────
    def compute_facts(self, instrument: Instrument, data: dict) -> dict:
        facts = {"regime": self.regime(), "instrument_class": instrument.asset_class.value}
        if self.prices is not None and instrument.symbol in self.prices.columns:
            px = self.prices[instrument.symbol].dropna()
            if len(px) > 63:
                facts["instrument_3m_return"] = round(float(px.iloc[-1] / px.iloc[-63] - 1), 4)
                facts["instrument_vol_annualized"] = round(float(px.pct_change().tail(63).std() * (252 ** 0.5)), 4)
        return facts
