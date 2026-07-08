"""Per-agent performance attribution (modification #3, part 1).

Each agent gets a *paper portfolio*: what would P&L have been if we had traded
only that agent's signals? This gives us per-agent return streams, hit rates,
and rolling ICs — the raw material the weighting layer consumes.

Signal convention (matches original repo): each agent emits, per instrument,
  {"signal": "bullish"|"bearish"|"neutral", "confidence": 0-100}
We encode that as signed exposure in [-1, +1]: direction * confidence/100.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from src.backtest.metrics import hit_rate, information_coefficient, sharpe_ratio

_DIRECTION = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0}


def encode_signal(signal: str, confidence: float) -> float:
    return _DIRECTION.get(signal, 0.0) * (confidence or 0.0) / 100.0


class AttributionTracker:
    """Accumulates each agent's daily signals and marks them against realized returns."""

    def __init__(self):
        # {agent: {date: {symbol: signed_exposure}}}
        self._exposures: dict[str, dict[pd.Timestamp, dict[str, float]]] = defaultdict(dict)

    def record(self, agent: str, date: pd.Timestamp, signals: dict[str, dict]) -> None:
        """signals: {symbol: {"signal": ..., "confidence": ...}} for one rebalance date."""
        self._exposures[agent][date] = {
            sym: encode_signal(s.get("signal", "neutral"), s.get("confidence", 0.0))
            for sym, s in signals.items()
        }

    def exposure_frame(self, agent: str) -> pd.DataFrame:
        """(date x symbol) signed-exposure matrix, forward-filled between rebalances."""
        frame = pd.DataFrame.from_dict(self._exposures[agent], orient="index").sort_index()
        return frame.ffill().fillna(0.0)

    def paper_returns(self, agent: str, asset_returns: pd.DataFrame) -> pd.Series:
        """Daily return of the agent's paper portfolio.

        asset_returns: (date x symbol) daily simple returns.
        Exposure at date t is applied to returns at t+1 (no lookahead).
        Exposures are normalized so gross exposure sums to 1 (or 0 if flat).
        """
        expo = self.exposure_frame(agent).reindex(asset_returns.index).ffill().fillna(0.0)
        gross = expo.abs().sum(axis=1).replace(0.0, np.nan)
        norm = expo.div(gross, axis=0).fillna(0.0)
        # shift(1): today's return earned on yesterday's stance
        common = norm.columns.intersection(asset_returns.columns)
        return (norm[common].shift(1) * asset_returns[common]).sum(axis=1).fillna(0.0)

    def scorecard(self, asset_returns: pd.DataFrame, window: int = 60) -> pd.DataFrame:
        """One row per agent: rolling + full-sample quality stats."""
        rows = []
        for agent in self._exposures:
            pr = self.paper_returns(agent, asset_returns)
            recent = pr.tail(window)
            # observations that actually carry information: days the agent held a
            # non-flat position within the window. The weighting layer uses this
            # to avoid firing an agent on a Sharpe computed from a handful of days.
            n_obs = int((recent != 0.0).sum())
            # per-symbol IC averaged across symbols
            expo = self.exposure_frame(agent).reindex(asset_returns.index).ffill().fillna(0.0)
            ics = []
            for sym in expo.columns.intersection(asset_returns.columns):
                ics.append(information_coefficient(expo[sym], asset_returns[sym].shift(-1)))
            ics = [i for i in ics if i == i]  # drop NaN
            rows.append({
                "agent": agent,
                "full_sharpe": sharpe_ratio(pr),
                f"rolling_{window}d_return": float(recent.sum()),
                f"rolling_{window}d_sharpe": sharpe_ratio(recent),
                "n_obs": n_obs,
                "hit_rate": hit_rate((expo.shift(1) * asset_returns[expo.columns.intersection(asset_returns.columns)]).sum(axis=1)),
                "avg_ic": sum(ics) / len(ics) if ics else float("nan"),
            })
        return pd.DataFrame(rows).set_index("agent")
