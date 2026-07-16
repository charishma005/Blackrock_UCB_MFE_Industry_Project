"""Portfolio performance metrics (modification #2).

Works on a daily portfolio-value series. Also used per-agent by the
attribution layer, so keep everything a pure function of return series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
RISK_FREE_ANNUAL = 0.0434  # match original repo's assumption; make configurable


def daily_returns(values: pd.Series) -> pd.Series:
    return values.pct_change().dropna()


def sharpe_ratio(returns: pd.Series, rf_annual: float = RISK_FREE_ANNUAL) -> float:
    excess = (returns - rf_annual / TRADING_DAYS).dropna()
    std = excess.std()
    if len(excess) < 2 or not np.isfinite(std) or std == 0:
        return float("nan")
    return float(np.sqrt(TRADING_DAYS) * excess.mean() / std)


def sortino_ratio(returns: pd.Series, rf_annual: float = RISK_FREE_ANNUAL) -> float:
    excess = (returns - rf_annual / TRADING_DAYS).dropna()
    downside = excess[excess < 0]
    std = downside.std()
    if len(excess) < 2 or len(downside) < 2 or not np.isfinite(std) or std == 0:
        return float("nan")
    return float(np.sqrt(TRADING_DAYS) * excess.mean() / std)


def max_drawdown(values: pd.Series) -> float:
    """Returns max drawdown as a negative fraction (e.g. -0.23)."""
    peak = values.cummax()
    dd = values / peak - 1.0
    return float(dd.min())


def calmar_ratio(values: pd.Series) -> float:
    rets = daily_returns(values)
    if len(rets) < 2:
        return float("nan")
    ann_return = (values.iloc[-1] / values.iloc[0]) ** (TRADING_DAYS / len(rets)) - 1
    mdd = abs(max_drawdown(values))
    return float(ann_return / mdd) if mdd > 0 else float("nan")


def hit_rate(signal_returns: pd.Series) -> float:
    """Fraction of periods where the signal-aligned return was positive.

    signal_returns should already be signed: position_direction * asset_return.
    """
    active = signal_returns[signal_returns != 0]
    if len(active) == 0:
        return float("nan")
    return float((active > 0).mean())


def turnover(weights: pd.DataFrame) -> float:
    """Average daily one-way turnover from a (date x instrument) weight matrix."""
    if len(weights) < 2:
        return float("nan")
    return float(weights.diff().abs().sum(axis=1).mean() / 2)


def information_coefficient(signals: pd.Series, forward_returns: pd.Series) -> float:
    """Spearman rank IC between signal strength and next-period return.

    Signals encoded as signed confidence: +conf for bullish, -conf for bearish, 0 neutral.
    """
    aligned = pd.concat([signals, forward_returns], axis=1).dropna()
    if len(aligned) < 5:
        return float("nan")
    # a constant signal (e.g. an agent that never changed its view in this
    # window) has undefined correlation — return NaN directly rather than
    # letting scipy warn on every call.
    if aligned.iloc[:, 0].nunique() <= 1 or aligned.iloc[:, 1].nunique() <= 1:
        return float("nan")
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman"))


def summary(values: pd.Series, weights: pd.DataFrame | None = None) -> dict:
    rets = daily_returns(values)
    out = {
        "total_return": float(values.iloc[-1] / values.iloc[0] - 1),
        "annualized_return": float((values.iloc[-1] / values.iloc[0]) ** (TRADING_DAYS / max(len(rets), 1)) - 1),
        "annualized_vol": float(rets.std() * np.sqrt(TRADING_DAYS)),
        "sharpe": sharpe_ratio(rets),
        "sortino": sortino_ratio(rets),
        "max_drawdown": max_drawdown(values),
        "calmar": calmar_ratio(values),
    }
    if weights is not None:
        out["avg_daily_turnover"] = turnover(weights)
    return out
