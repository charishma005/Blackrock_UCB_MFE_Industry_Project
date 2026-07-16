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
    shrinkage: float = 0.20            # Ledoit-Wolf-style shrink of the sample
                                       # covariance toward a diagonal target. With
                                       # a 63-day window and a growing universe the
                                       # sample cov is ill-conditioned; shrinkage
                                       # keeps portfolio_vol / vol-scaling stable so
                                       # the book isn't over- or under-levered on
                                       # noise. 0 = raw sample cov, 1 = pure diagonal.
    ewma_halflife: float | None = None # if set, weight the covariance window by an
                                       # exponential decay with this half-life (days)
                                       # instead of equal-weighting all lookback_days.
                                       # More responsive to regime shifts, so vol
                                       # targeting de-levers faster into turbulence.
                                       # None = equal-weighted sample covariance.


class RiskManager:
    def __init__(self, config: RiskConfig | None = None):
        self.config = config or RiskConfig()

    def _annualized_cov(self, returns: pd.DataFrame) -> pd.DataFrame:
        window = returns.tail(self.config.lookback_days)
        # Drop instruments that don't have enough overlapping history in the
        # window to estimate a variance — otherwise their column is all/partly
        # NaN, which propagates into w @ C @ w and silently poisons the vol
        # estimate (NaN is not caught by the aggregate min_obs row check).
        window = window.dropna(axis=1, how="any")
        if self.config.ewma_halflife and len(window) > 1:
            cov = self._ewma_cov(window) * 252
        else:
            cov = window.cov() * 252
        return self._shrink(cov)

    def _ewma_cov(self, window: pd.DataFrame) -> pd.DataFrame:
        """Exponentially-weighted covariance over the window.

        Recent days get more weight (half-life = ``ewma_halflife``), so the
        estimate reacts faster to a volatility regime change than an equal-
        weighted sample covariance that treats a 3-month-old calm day the same
        as yesterday's shock.
        """
        hl = float(self.config.ewma_halflife)
        n = len(window)
        # newest row has the largest weight
        age = np.arange(n - 1, -1, -1, dtype=float)
        wts = 0.5 ** (age / hl)
        wts /= wts.sum()
        X = window.values
        mean = np.average(X, axis=0, weights=wts)
        Xc = X - mean
        cov = (Xc * wts[:, None]).T @ Xc
        # de-bias for the weighting (1 - sum w^2) so variances aren't understated
        cov /= max(1.0 - np.sum(wts ** 2), 1e-9)
        return pd.DataFrame(cov, index=window.columns, columns=window.columns)

    def _shrink(self, cov: pd.DataFrame) -> pd.DataFrame:
        """Convex shrinkage toward a diagonal (variance-only) target.

        cov_hat = (1-d)*sample + d*diag(sample). Off-diagonal (correlation)
        terms are the noisiest part of a short-window sample covariance, so
        shrinking them toward zero conditions the matrix without distorting the
        marginal variances that vol targeting depends on most.
        """
        d = float(min(max(self.config.shrinkage, 0.0), 1.0))
        if d == 0.0 or cov.empty:
            return cov
        target = pd.DataFrame(
            np.diag(np.diag(cov.values)), index=cov.index, columns=cov.columns
        )
        return (1.0 - d) * cov + d * target

    def portfolio_vol(self, weights: pd.Series, cov: pd.DataFrame) -> float:
        cols = weights.index.intersection(cov.columns)
        w = weights[cols].values
        c = cov.loc[cols, cols].values
        var = float(w @ c @ w)
        return float(np.sqrt(max(var, 0.0)))

    def effective_breadth(self, weights: pd.Series, corr: pd.DataFrame) -> float:
        """Count of independent bets = (sum|w|)^2 / (w' C w), using SIGNED
        weights and SIGNED correlation.

        Using |corr| (the previous version) treated a genuine hedge — long A,
        short B, positively correlated — as concentrated risk and cut exposure
        to it, which is backwards: that pair is *diversifying*. With signed
        weights, an offsetting position lowers the quadratic form w'Cw, raising
        breadth and easing the haircut, exactly as it should for a hedged book.
        Breadth is clamped to [0, n] so this stays a haircut (never levers up)."""
        cols = weights.index.intersection(corr.columns)
        w = weights[cols].values  # signed
        if np.abs(w).sum() == 0:
            return 0.0
        c = corr.loc[cols, cols].values  # signed
        denom = float(w @ c @ w)
        if denom <= 0:
            return 0.0
        breadth = float((np.abs(w).sum() ** 2) / denom)
        return float(min(breadth, len(cols)))

    def apply(self, target_weights: pd.Series, returns: pd.DataFrame) -> tuple[pd.Series, dict]:
        """Return (risk_adjusted_weights, diagnostics)."""
        cfg = self.config
        w = target_weights.copy().astype(float)

        # 1. per-name cap
        w = w.clip(-cfg.max_weight_per_name, cfg.max_weight_per_name)

        if w.abs().sum() == 0 or len(returns) < cfg.min_obs:
            return w, {"reason": "flat or insufficient history", "scale": 1.0}

        cov = self._annualized_cov(returns)
        # correlation on the same NaN-free window used for covariance, so the
        # breadth haircut isn't computed from partly-undefined pairwise corrs.
        corr = returns.tail(cfg.lookback_days).dropna(axis=1, how="any").corr()

        # 2. vol targeting
        current_vol = self.portfolio_vol(w, cov)
        # A NaN/degenerate vol estimate (e.g. brand-new instruments with no
        # usable covariance) must not scale the whole book by NaN — leave gross
        # unchanged and flag it rather than silently corrupting every weight.
        if not np.isfinite(current_vol) or current_vol <= 1e-9:
            return w, {"reason": "no usable covariance estimate", "scale": 1.0}
        scale = cfg.target_vol / current_vol

        # 3. correlation haircut: fewer independent bets => less risk budget.
        #    breadth of N names ranges 1..N; scale the risk budget by
        #    sqrt(breadth / N) so a fully-correlated book gets ~1/sqrt(N).
        n = int((w != 0).sum())
        breadth = self.effective_breadth(w, corr)
        breadth_haircut = float(np.sqrt(breadth / n)) if n > 0 and breadth > 0 else 1.0
        scale *= breadth_haircut

        # scale, then RE-CLIP per name: vol targeting can lever a name back
        # above max_weight_per_name, so the pre-scale clip alone doesn't hold.
        # Re-clipping after scaling is what actually enforces the per-name cap.
        w_scaled = (w * scale).clip(-cfg.max_weight_per_name, cfg.max_weight_per_name)

        # cap gross leverage
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
            # surface the hard limits so a downstream LLM PM knows the actual
            # numeric bounds it must respect, not just the post-scale weights.
            "max_weight_per_name": cfg.max_weight_per_name,
            "max_gross_leverage": cfg.max_gross_leverage,
        }
        return w_scaled, diagnostics
