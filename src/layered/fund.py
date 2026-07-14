"""The unifying layer — running the fund as a portfolio of strategies.

"Beneath the several portfolio managers sits a layer whose responsibility is the
fund as a whole ... it runs the fund as a portfolio of strategies. It nets
exposures across strategies ... and it allocates capital among the PMs, sizing
each strategy by conviction, by risk, and by how much it diversifies the others."

``FundAllocator`` does exactly those three things and nothing more — it is "a
control layer, not a forecasting one." It emits a ``FundAllocation`` (capital
multipliers + constraints) fed back DOWN to the PMs, closing the loop.

Deliberate difference from ``src/risk/manager.py``: that risk manager reshapes a
single blended book and clips individual names — appropriate for the flat
ensemble. This layer must NOT clip a PM's individual legs, because doing so would
un-hedge a carefully DV01-neutral curve trade. So it sizes the *netted* book to a
vol target and caps gross, but leaves each PM's trade shape intact. Netting +
vol-targeting math is shared with the risk manager in spirit; the control
surface is different.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.layered.contracts import FundAllocation, StrategyTrade
from src.layered.timeline import AsOf


@dataclass
class FundConfig:
    target_vol: float = 0.06        # annualized vol target for the whole fund
                                    # (rates RV books run lower vol than an equity book)
    max_gross_leverage: float = 4.0  # curve trades are naturally high-gross (large,
                                    # offsetting DV01 legs) — a 1.5x cap would be binding
                                    # on a hedged trade and distort it
    lookback_days: int = 63
    min_obs: int = 20
    div_floor: float = 0.25         # a fully-redundant strategy still keeps this
                                    # fraction of its conviction-based size


class FundAllocator:
    def __init__(self, config: FundConfig | None = None):
        self.config = config or FundConfig()

    # ── diversification: reward strategies that decorrelate the others ──────
    def _diversification(
        self, trades: list[StrategyTrade], strat_returns: dict[str, pd.Series] | None
    ) -> dict[str, float]:
        """Multiplier per strategy from how much it diversifies the rest.

        With realized per-strategy paper returns, a strategy is down-weighted in
        proportion to its average absolute correlation with the others (redundant
        risk earns less capital). Without histories (e.g. early in a run, or a
        single strategy), every strategy gets 1.0 — diversification can only be a
        first-class edge once there are several return streams to compare.
        """
        strategies = [t.strategy for t in trades]
        if not strat_returns or len(strategies) < 2:
            return {s: 1.0 for s in strategies}
        frame = pd.DataFrame({s: strat_returns[s] for s in strategies if s in strat_returns}).dropna()
        if len(frame) < self.config.min_obs or frame.shape[1] < 2:
            return {s: 1.0 for s in strategies}
        corr = frame.corr().abs()
        out: dict[str, float] = {}
        for s in strategies:
            if s in corr.columns:
                others = corr.loc[s].drop(labels=[s]).mean()
                out[s] = float(max(self.config.div_floor, 1.0 - others))
            else:
                out[s] = 1.0
        return out

    # ── vol targeting on the NETTED book ────────────────────────────────────
    def _vol_scale(self, netted_unit: pd.Series, rets: pd.DataFrame) -> tuple[float, dict]:
        cfg = self.config
        if netted_unit.abs().sum() < 1e-9 or len(rets) < cfg.min_obs:
            return 1.0, {"reason": "flat or insufficient history", "scale": 1.0}
        cols = netted_unit.index.intersection(rets.columns)
        w = netted_unit[cols].values
        cov = (rets[cols].tail(cfg.lookback_days).cov() * 252).values
        vol = float(np.sqrt(max(w @ cov @ w, 0.0)))
        scale = (cfg.target_vol / vol) if vol > 1e-9 else 1.0
        gross = float(np.abs(w * scale).sum())
        if gross > cfg.max_gross_leverage:
            scale *= cfg.max_gross_leverage / gross
        post_w = w * scale
        return scale, {
            "pre_scale_vol": round(vol, 4),
            "target_vol": cfg.target_vol,
            "vol_scale": round(scale, 3),
            "netted_gross": round(float(np.abs(post_w).sum()), 3),
            "post_scale_vol": round(float(np.sqrt(max(post_w @ cov @ post_w, 0.0))), 4),
        }

    # ── entry point ─────────────────────────────────────────────────────────
    def allocate(
        self,
        trades: list[StrategyTrade],
        world: AsOf,
        strat_returns: dict[str, pd.Series] | None = None,
    ) -> FundAllocation:
        """Size the strategies and return the capital + constraints, fed back down."""
        cfg = self.config
        div = self._diversification(trades, strat_returns)
        # base sizing = conviction × diversification (the "by conviction, by how
        # much it diversifies the others" part of the mandate).
        base = {t.strategy: max(t.conviction, 0.0) * div.get(t.strategy, 1.0) for t in trades}

        instruments = sorted({s for t in trades for s in t.legs})
        netted_unit = pd.Series(0.0, index=instruments, dtype=float)
        for t in trades:
            for sym, w in t.legs.items():
                netted_unit[sym] += base[t.strategy] * w

        rets = world.frame(instruments).pct_change().dropna(how="all")
        scale, vol_diag = self._vol_scale(netted_unit, rets)  # the "by risk" part

        capital = {s: round(base[s] * scale, 4) for s in base}
        constraints = {s: {"max_gross_leverage": cfg.max_gross_leverage} for s in base}

        netted_final = (netted_unit * scale).round(4)
        diagnostics = {
            "diversification": {s: round(v, 3) for s, v in div.items()},
            "base_size": {s: round(v, 3) for s, v in base.items()},
            "netted_book": {k: float(v) for k, v in netted_final.items() if abs(v) > 1e-9},
            **vol_diag,
        }
        return FundAllocation(asof=world.asof, capital=capital,
                              constraints=constraints, diagnostics=diagnostics)
