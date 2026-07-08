"""Multi-asset risk manager (modification #2).

Turns raw target weights (from agent blending) into risk-constrained weights.
Three constraints, applied in order:

  1. Per-instrument cap: no single instrument exceeds `max_weight_per_name`.
  2. Volatility targeting: scale gross exposure so the portfolio's ex-ante
     annualized volatility (from the trailing covariance matrix) hits
     `target_vol`. This is the main lever — it de-levers in turbulent regimes
     and levers up (to a cap) in calm ones.
  3. Correlation-aware haircut: if the book is concentrated in highly
     correlated names, shrink gross exposure further (a crude effective-breadth
     adjustment — fewer independent bets => less risk budget).

Everything here is deterministic Python. No LLM. This is the auditable
risk layer that sits between agent opinions and the (optionally LLM-driven)
portfolio manager.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RiskConfig:
    target_vol: float = 0.10          # annualized portfolio vol target
    max_weight_per_name: float = 0.25  # cap on any single instrument (abs)
    max_gross_leverage: float = 1.5    # ceiling on sum(|w|) after vol scaling
    lookback_days: int = 63            # trailing window for covariance
    min_obs: int = 20                  # need at least this many returns to size


class RiskManager:
    def __init__(self, config: RiskConfig | None = None):
        self.config = config or RiskConfig()

    def _annualized_cov(self, returns: pd.DataFrame) -> pd.DataFrame:
        window = returns.tail(self.config.lookback_days)
        return window.cov() * 252

    def portfolio_vol(self, weights: pd.Series, cov: pd.DataFrame) -> float:
        cols = weights.index.intersection(cov.columns)
        w = weights[cols].values
        c = cov.loc[cols, cols].values
        var = float(w @ c @ w)
        return float(np.sqrt(max(var, 0.0)))

    def effective_breadth(self, weights: pd.Series, corr: pd.DataFrame) -> float:
        """Rough count of independent bets = (sum|w|)^2 / (w' |C| w).
        High pairwise correlation shrinks this toward 1."""
        cols = weights.index.intersection(corr.columns)
        w = np.abs(weights[cols].values)
        if w.sum() == 0:
            return 0.0
        c = np.abs(corr.loc[cols, cols].values)
        denom = float(w @ c @ w)
        return float((w.sum() ** 2) / denom) if denom > 0 else 0.0

    def apply(self, target_weights: pd.Series, returns: pd.DataFrame) -> tuple[pd.Series, dict]:
        """Return (risk_adjusted_weights, diagnostics)."""
        cfg = self.config
        w = target_weights.copy().astype(float)

        # 1. per-name cap
        w = w.clip(-cfg.max_weight_per_name, cfg.max_weight_per_name)

        if w.abs().sum() == 0 or len(returns) < cfg.min_obs:
            return w, {"reason": "flat or insufficient history", "scale": 1.0}

        cov = self._annualized_cov(returns)
        corr = returns.tail(cfg.lookback_days).corr()

        # 2. vol targeting
        current_vol = self.portfolio_vol(w, cov)
        scale = (cfg.target_vol / current_vol) if current_vol > 1e-9 else 1.0

        # 3. correlation haircut: fewer independent bets => less risk budget.
        #    breadth of N names ranges 1..N; scale the risk budget by
        #    sqrt(breadth / N) so a fully-correlated book gets ~1/sqrt(N).
        n = int((w != 0).sum())
        breadth = self.effective_breadth(w, corr)
        breadth_haircut = float(np.sqrt(breadth / n)) if n > 0 and breadth > 0 else 1.0
        scale *= breadth_haircut

        # cap gross leverage
        w_scaled = w * scale
        gross = w_scaled.abs().sum()
        if gross > cfg.max_gross_leverage:
            w_scaled *= cfg.max_gross_leverage / gross
            scale *= cfg.max_gross_leverage / gross

        diagnostics = {
            "pre_scale_vol": round(current_vol, 4),
            "target_vol": cfg.target_vol,
            "vol_scale": round(cfg.target_vol / current_vol, 3) if current_vol > 1e-9 else 1.0,
            "effective_breadth": round(breadth, 2),
            "n_positions": n,
            "breadth_haircut": round(breadth_haircut, 3),
            "final_scale": round(scale, 3),
            "final_gross": round(w_scaled.abs().sum(), 3),
            "post_scale_vol": round(self.portfolio_vol(w_scaled, cov), 4),
        }
        return w_scaled, diagnostics
