"""The macro rates PM — the worked example from the thesis, in code.

    "Suppose the inflation analyst sees upside surprises building, the
     unemployment analyst sees the labor market still tight, and the balance-
     sheet analyst sees liquidity draining under continued runoff. No single
     view names a trade. The PM reads them together: persistent inflation plus
     a tight labor market implies a more hawkish path for policy, which bears
     hardest on the front end of the curve, while runoff pressures the term
     premium at the long end. The clean expression is not an outright short in
     rates ... but a relative-value position in the curve, a 2s10s flattener,
     that isolates precisely the joint view the analysts collectively hold and
     hedges away what they do not."

The transmission map (the PM's edge) lives in ``personas/macro_rates_pm.yaml``.
``express`` projects the four driver views onto two curve points, decides
flattener vs steepener from which point faces more yield pressure, and builds
DV01-neutral legs so only the 2s10s slope is expressed and the level is hedged.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from src.instruments import AssetClass, Instrument
from src.layered.contracts import ArbitratedView, DriverView, StrategyTrade
from src.layered.pm.base import PortfolioManagerBase
from src.layered.timeline import AsOf

_PERSONA_PATH = Path(__file__).parent / "personas" / "macro_rates_pm.yaml"

# Tradeable universe for the rates strategy. SHY is not in the flat ensemble's
# DEFAULT_UNIVERSE, so the layered orchestrator fetches these itself.
RATES_UNIVERSE: list[Instrument] = [
    Instrument("SHY", AssetClass.FIXED_INCOME, "1-3yr Treasuries (2y proxy)"),
    Instrument("IEF", AssetClass.FIXED_INCOME, "7-10yr Treasuries (10y proxy)"),
]

_DEFAULT_MAP = {
    "front_end_weights": {"inflation": 0.50, "labor_tightness": 0.35, "balance_sheet": -0.15},
    "long_end_weights": {"term_premium": 0.60, "balance_sheet": -0.40},
    "instruments": {"front": {"symbol": "SHY", "duration": 1.9},
                    "long": {"symbol": "IEF", "duration": 7.5}},
    "disagreement_haircut": 0.5,
}


class MacroRatesPM(PortfolioManagerBase):
    strategy = "macro_rates"
    listens_to = ("inflation", "labor_tightness", "balance_sheet", "term_premium")

    def __init__(self, llm_client=None):
        super().__init__(llm_client)
        self.map = self._load_map()

    def _load_map(self) -> dict:
        if _PERSONA_PATH.exists():
            loaded = yaml.safe_load(_PERSONA_PATH.read_text()) or {}
            return {**_DEFAULT_MAP, **loaded}
        return dict(_DEFAULT_MAP)

    # ── arbitrate: add a cross-driver disagreement on the hawkishness axis ──
    def arbitrate(self, views: list[DriverView]) -> ArbitratedView:
        av = super().arbitrate(views)
        # Project each driver's contribution onto the front-end (policy) axis and
        # measure how much the specialists pull against each other there. Unlike
        # the base same-driver measure, THIS captures the meaningful conflict:
        # e.g. inflation pushing hawkish while labor pushes dovish.
        contribs = [w * av.drivers.get(d, 0.0) for d, w in self.map["front_end_weights"].items()]
        denom = sum(abs(c) for c in contribs)
        cross = (1.0 - abs(sum(contribs)) / denom) if denom > 1e-9 else 0.0
        av.disagreement = round(max(av.disagreement, cross), 3)
        return av

    # ── express: the driver → instrument transmission map ───────────────────
    def express(self, view: ArbitratedView, world: AsOf) -> StrategyTrade:
        d = view.drivers

        # 1. Project the joint view onto two points on the curve. Positive =
        #    upward yield pressure at that maturity.
        front = float(np.clip(sum(w * d.get(k, 0.0) for k, w in self.map["front_end_weights"].items()), -1, 1))
        long_ = float(np.clip(sum(w * d.get(k, 0.0) for k, w in self.map["long_end_weights"].items()), -1, 1))

        # 2. The curve view is the DIFFERENCE, not the level — the level (a
        #    parallel shift) is exactly what an RV curve trade hedges away.
        #    front > long  ⇒ front-end rises more ⇒ 2s10s spread (10y−2y) falls
        #    ⇒ curve flattens ⇒ FLATTENER. Otherwise a STEEPENER.
        slope_pressure = front - long_
        flattener = slope_pressure > 0
        structure = "2s10s flattener" if flattener else "2s10s steepener"

        conviction = float(np.clip(abs(slope_pressure), 0.0, 1.0))
        conviction *= (1.0 - self.map["disagreement_haircut"] * view.disagreement)  # split committee → smaller bet
        conviction = round(float(np.clip(conviction, 0.0, 1.0)), 3)

        # 3. Build DV01-neutral legs. Long duration at the 10y (long IEF) and
        #    short duration at the 2y (short SHY) for a flattener; flip for a
        #    steepener. Sizing the legs' durations equal-and-opposite means a
        #    parallel move nets ~0 and only the slope change shows up as P&L.
        f_sym, f_dur = self.map["instruments"]["front"]["symbol"], self.map["instruments"]["front"]["duration"]
        l_sym, l_dur = self.map["instruments"]["long"]["symbol"], self.map["instruments"]["long"]["duration"]
        sign = 1.0 if flattener else -1.0
        long_leg = sign * 1.0                       # +IEF for flattener
        front_leg = -sign * (l_dur / f_dur)         # DV01-neutralizing SHY leg
        legs = {l_sym: long_leg, f_sym: front_leg}

        # Normalize to unit gross; conviction rides on the contract's own field so
        # the fund layer sizes against it (a weak view is a small allocation, not
        # a distorted trade shape).
        gross = sum(abs(v) for v in legs.values()) or 1.0
        legs = {s: round(w / gross, 4) for s, w in legs.items()}
        net_duration = round(legs[l_sym] * l_dur + legs[f_sym] * f_dur, 3)  # ≈ 0 by construction

        dominant = max(self.map["front_end_weights"], key=lambda k: abs(self.map["front_end_weights"][k] * d.get(k, 0.0)), default="")
        rationale = (
            f"front-end pressure {front:+.2f} vs long-end {long_:+.2f} → {structure}; "
            f"isolates the 2s10s slope, hedges the level. Dominant driver: {dominant}."
        )
        return StrategyTrade(
            strategy=self.strategy, asof=view.asof, legs=legs, conviction=conviction,
            rationale=rationale,
            risk={
                "structure": structure,
                "front_pressure": round(front, 3),
                "long_pressure": round(long_, 3),
                "slope_pressure": round(slope_pressure, 3),
                "net_duration": net_duration,     # residual level exposure, ≈ 0
                "isolates": "2s10s curve slope",
                "hedges": "parallel level of rates",
                "disagreement": view.disagreement,
                "driver_view": {k: round(v, 3) for k, v in d.items()},
            },
        )
