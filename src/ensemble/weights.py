"""Dynamic agent weighting — the "firing" mechanism (modification #3, part 2).

Design choices (defend these in your writeup):
  * Continuous decay, not binary firing: weights shrink smoothly via a softmax
    over rolling performance, so one bad window doesn't whipsaw the ensemble.
  * Regime-aware floors: macro/tail-risk agents earn their keep episodically,
    so they get a minimum weight and a longer evaluation window instead of
    being fired during quiet regimes.
  * Hard fire only on persistence: an agent is excluded (weight 0) only after
    `fire_after` consecutive evaluations below the fire threshold.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class AgentPolicy:
    window: int = 60          # evaluation window (days) — use longer for macro agents
    floor: float = 0.0        # minimum weight (e.g. 0.05 for macro/tail agents)
    fire_threshold: float = -0.5   # rolling Sharpe below this counts as a strike
    fire_after: int = 3       # consecutive strikes before hard exclusion
    min_obs: int = 20         # need this many real return observations before a
                              # strike counts — a rolling Sharpe on ~10 daily points
                              # is noise; without this gate an agent could be fired
                              # on under a month of data.


@dataclass
class WeightManager:
    temperature: float = 1.0  # softmax temperature; lower = more aggressive tilting
    policies: dict[str, AgentPolicy] = field(default_factory=dict)
    _strikes: dict[str, int] = field(default_factory=dict)
    _fired: set[str] = field(default_factory=set)

    def policy(self, agent: str) -> AgentPolicy:
        return self.policies.get(agent, AgentPolicy())

    def update(self, scorecard: pd.DataFrame, score_col: str = "rolling_60d_sharpe") -> dict[str, float]:
        """Compute new weights from the attribution scorecard.

        Returns {agent: weight}, weights sum to 1 across non-fired agents.
        """
        scores: dict[str, float] = {}
        for agent, row in scorecard.iterrows():
            if agent in self._fired:
                continue
            s = row.get(score_col, float("nan"))
            s = 0.0 if s != s else float(s)  # NaN -> neutral score

            n_obs = row.get("n_obs", 0)
            n_obs = 0 if n_obs != n_obs else int(n_obs)  # NaN -> 0

            # strike accounting for hard firing — only once we have enough real
            # observations for the rolling Sharpe to mean anything (min_obs gate).
            pol = self.policy(agent)
            if s < pol.fire_threshold and n_obs >= pol.min_obs:
                self._strikes[agent] = self._strikes.get(agent, 0) + 1
                if self._strikes[agent] >= pol.fire_after:
                    self._fired.add(agent)
                    continue
            elif s >= pol.fire_threshold:
                self._strikes[agent] = 0
            scores[agent] = s

        if not scores:
            return {}

        # softmax over scores
        arr = np.array(list(scores.values()), dtype=float)
        expd = np.exp((arr - arr.max()) / max(self.temperature, 1e-6))
        soft = expd / expd.sum()
        weights = dict(zip(scores.keys(), soft))

        # apply floors, then renormalize
        for agent in weights:
            weights[agent] = max(weights[agent], self.policy(agent).floor)
        total = sum(weights.values())
        return {a: w / total for a, w in weights.items()}

    @property
    def fired(self) -> set[str]:
        return set(self._fired)

    def rehire(self, agent: str) -> None:
        """Manual override — e.g. after a regime change."""
        self._fired.discard(agent)
        self._strikes[agent] = 0
