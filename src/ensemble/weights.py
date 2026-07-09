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

    def update(self, scorecard: pd.DataFrame, score_col: str = "rolling_sharpe") -> dict[str, float]:
        """Compute new weights from the attribution scorecard.

        Returns {agent: weight}, weights sum to 1 across non-fired agents.
        """
        scores: dict[str, float] = {}
        idle: list[str] = []  # non-fired agents holding no position this window
        for agent, row in scorecard.iterrows():
            full_sharpe = row.get("full_sharpe", float("nan"))
            full_sharpe = 0.0 if full_sharpe != full_sharpe else float(full_sharpe)  # NaN -> 0

            if agent in self._fired:
                # Recovery: a hard-fired agent is reinstated once its FULL-sample
                # track record turns positive again. Firing must not be a one-way
                # ratchet — at weekly cadence a high-turnover agent racks up many
                # noisy short-window evaluations, and a permanent exile there
                # eventually strands the ensemble in whichever agent never traded
                # (an all-neutral agent can't lose, so it can't be fired).
                if full_sharpe > 0:
                    self.rehire(agent)
                else:
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
                # HARD fire only when recent weakness is corroborated by a losing
                # FULL-sample record. An agent whose full-sample Sharpe is still
                # positive (e.g. a macro agent having a noisy quarter) is benched
                # via low softmax weight, not permanently exiled on short-window
                # noise — which is what killed net-positive Ray Dalio before.
                if self._strikes[agent] >= pol.fire_after and full_sharpe < 0:
                    self._fired.add(agent)
                    continue
            elif s >= pol.fire_threshold:
                self._strikes[agent] = 0

            # Bench idle agents: one holding no position over its window (n_obs
            # == 0) contributes zero exposure to the blend, so handing it softmax
            # weight just idles risk budget and lets a chronically-neutral agent
            # (e.g. an all-"neutral" Buffett) crowd out the agents actually
            # taking views. It re-enters automatically once it trades again.
            if n_obs == 0:
                idle.append(agent)
                continue
            scores[agent] = s

        if not scores:
            # Nobody is actively positioned yet (e.g. the first evaluations,
            # before returns accrue). Fall back to equal weight among idle
            # non-fired agents rather than returning empty — the blend is flat
            # either way, but this keeps the ensemble's bookkeeping sensible.
            if idle:
                return {a: 1.0 / len(idle) for a in idle}
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
