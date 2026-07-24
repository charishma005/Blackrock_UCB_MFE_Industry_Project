"""v0 — the relevance-weighted mechanical combiner, with a pluggable weighting scheme.

The confidence-vs-relevance finding (`notebook §7.10`, `docs/decisions.md` 2026-07-23): the
PM weights each analyst by conviction, but conviction measures certainty about a driver's
*own* series, which §7.10 shows is ~inverted from that driver's power over the *traded*
instrument. So both the mechanical `consensus_blend` and the LLM PM load the least-tradable
driver (balance_sheet: own-IC 0.69, trade-IC ≈ 0).

This combiner fixes the weighting mechanically: for each meeting it weights each analyst's
oriented view by a **walk-forward** estimate of that analyst's trade relevance, then projects
the weighted panel onto the rate axis exactly as `MechanicalPM` does. "Which relevance
weighting" is itself an empirical choice, so the scheme is pluggable (`weighting=`) and the
whole set is a free, deterministic sweep on one board:

  equal      w=1 (→ the equal-weight `consensus_blend` baseline; also the warm-up fallback)
  ic         signed trailing rank-IC to the traded instrument, shrunk by n/(n+k)
  ir         IC · sqrt(n) (an IC t-stat) — rewards *stable/longer-history* relevance, shrunk
  rank_topk  keep the top-k analysts by |IC|, equal magnitude with the IC's sign (sparse)
  ridge      ridge regression of the traded move on the oriented views (optimal-linear,
             handles analyst correlation; regularized so it does not overfit at low N/T)

**No look-ahead.** The weight at meeting *t* uses only `(oriented view at s, realized move
over s→s+1)` pairs whose outcome is known strictly before *t* (`s ≤ t-1`). Below `min_obs`
usable pairs an analyst gets zero weight; while *all* analysts are cold the combiner falls
back to the equal-weight panel mean — i.e. `MechanicalPM`'s projection, so early meetings
reproduce the baseline exactly.

Only `_rate_projection` (the trade) changes; the driver block stays `consensus_blend`
(graded in driver space, where relevance-weighting has no meaning) and an `opposed` pod
still abstains on the trade — so this improves the pods that trade (duration, front_end,
real), not curve.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.layered.pm.disagreement import oriented
from src.layered.pm.mechanical_pm import POD_DIR, MechanicalPM

WEIGHTINGS = ("equal", "ic", "ir", "rank_topk", "ridge")

_DESCRIPTION = (
    "RELEVANCE PM (v0) — a deterministic combiner that weights each analyst's oriented view "
    "by a walk-forward estimate of its trade relevance (weighting scheme is configurable), "
    "instead of by conviction. Warm-up / all-cold meetings fall back to the equal-weight "
    "panel mean (= the mechanical baseline). No look-ahead: weights at t use only outcomes "
    "realized before t; nothing is fitted to the future."
)


def _rank_ic(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    pair = pd.concat([x, y], axis=1).dropna()
    n = len(pair)
    if n < 2 or pair.iloc[:, 0].std() == 0 or pair.iloc[:, 1].std() == 0:
        return float("nan"), n
    return float(pair.iloc[:, 0].rank().corr(pair.iloc[:, 1].rank())), n


class RelevancePM(MechanicalPM):
    """MechanicalPM with a relevance-weighted (not equal-weighted) rate-axis projection.

    Construct with a ``weighting`` scheme, then call :meth:`fit` with the board, meeting
    dates, and macro bundle to precompute the walk-forward weight schedule.
    """

    def __init__(self, pod: str, config: dict, weighting: str = "ic",
                 min_obs: int = 12, shrink_k: float = 8.0, topk: Optional[int] = None,
                 ridge_alpha: float = 5.0):
        super().__init__(pod, config)
        if weighting not in WEIGHTINGS:
            raise ValueError(f"weighting must be one of {WEIGHTINGS}, got {weighting!r}")
        self.weighting = weighting
        self.min_obs = int(min_obs)
        self.shrink_k = float(shrink_k)
        self.topk = topk
        self.ridge_alpha = float(ridge_alpha)
        self._weights: dict[pd.Timestamp, dict[str, float]] = {}
        self.weight_frame: Optional[pd.DataFrame] = None

    @classmethod
    def from_pod(cls, pod: str, pod_dir: Optional[Path] = None, *, weighting: str = "ic",
                 min_obs: int = 12, shrink_k: float = 8.0, topk: Optional[int] = None,
                 ridge_alpha: float = 5.0) -> "RelevancePM":
        import yaml
        path = (pod_dir or POD_DIR) / f"{pod}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"no pod spec for {pod!r} at {path}")
        return cls(pod=pod, config=yaml.safe_load(path.read_text()) or {}, weighting=weighting,
                   min_obs=min_obs, shrink_k=shrink_k, topk=topk, ridge_alpha=ridge_alpha)

    # ── the traded-outcome axis this pod's relevance is measured against ──────
    def _axis_series(self, macro) -> pd.Series:
        cfg = self.trade_config
        univ = [str(s) for s in (cfg.get("universe") or [])]
        ys = [macro[s].dropna().resample(self.clock_freq).last() for s in univ]
        if str(cfg.get("sign_convention", "") or "").strip().lower() == "opposed" and len(ys) >= 2:
            return (ys[-1] - ys[0]).rename("axis")          # slope (opposed pod abstains anyway)
        return (sum(ys) / len(ys)).rename("axis")           # level / single-instrument

    # ── weighting schemes: (trailing X: dates×drivers, y: outcome) → {driver: weight} ──
    def _weights_for(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        drv = self.listens_to
        if self.weighting == "equal":
            return {d: 1.0 for d in drv}

        if self.weighting in ("ic", "ir"):
            w = {}
            for d in drv:
                ic, n = _rank_ic(X[d], y)
                if n < self.min_obs or ic != ic:
                    w[d] = 0.0
                elif self.weighting == "ic":
                    w[d] = ic * n / (n + self.shrink_k)            # signed IC, shrunk
                else:                                             # ir: IC t-stat, shrunk
                    w[d] = ic * np.sqrt(n) * n / (n + self.shrink_k)
            return w

        if self.weighting == "rank_topk":
            ics = {}
            for d in drv:
                ic, n = _rank_ic(X[d], y)
                ics[d] = ic if (n >= self.min_obs and ic == ic) else 0.0
            k = self.topk or max(1, (len(drv) + 1) // 2)
            top = sorted(drv, key=lambda d: -abs(ics[d]))[:k]
            return {d: (np.sign(ics[d]) if (d in top and ics[d] != 0.0) else 0.0) for d in drv}

        if self.weighting == "ridge":
            rows = X.dropna(how="all")
            pair = pd.concat([rows, y.rename("__y__")], axis=1).dropna(subset=["__y__"])
            if len(pair) < max(self.min_obs, 2 * len(drv)):       # too little data → warm up
                return {d: 0.0 for d in drv}
            Xm = pair[drv].fillna(0.0).to_numpy(float)
            yv = pair["__y__"].to_numpy(float)
            mu, sd = Xm.mean(0), Xm.std(0)
            sd[sd == 0] = 1.0
            Xs = (Xm - mu) / sd                                   # standardize columns
            p = Xs.shape[1]
            beta = np.linalg.solve(Xs.T @ Xs + self.ridge_alpha * np.eye(p), Xs.T @ yv)
            return {d: float(beta[i]) for i, d in enumerate(drv)}

        raise ValueError(self.weighting)                          # unreachable

    # ── precompute the walk-forward weight schedule ──────────────────────────
    def fit(self, board, dates, macro) -> "RelevancePM":
        pol, drv = self.polarity, self.listens_to
        idx = pd.DatetimeIndex(dates)
        omat = pd.DataFrame(
            {pd.Timestamp(dt): {d: oriented(board.at(dt), pol).get(d, np.nan) for d in drv}
             for dt in dates}
        ).T.reindex(idx)
        axis = self._axis_series(macro)
        outcome = (axis.shift(-1) - axis).reindex(idx)            # realized at s+1, known before t

        order = list(idx)
        weights = {}
        for i, t in enumerate(order):
            past = order[:i]                                      # strictly before t
            weights[pd.Timestamp(t)] = self._weights_for(omat.loc[past], outcome.loc[past])
        self._weights = weights
        self.weight_frame = pd.DataFrame(weights).T
        return self

    # ── the only behavioural override: a relevance-weighted projection ───────
    def _rate_projection(self, m) -> float:
        o = oriented(m, self.polarity)
        if not o:
            return 0.0
        w = self._weights.get(pd.Timestamp(m.asof), {})
        num = sum(w.get(d, 0.0) * o[d] for d in o)
        den = sum(abs(w.get(d, 0.0)) for d in o)
        if den > 0:
            return float(num / den)
        return float(sum(o.values()) / len(o))                    # warm-up: equal-weight baseline

    def _system_prompt(self) -> str:
        return (f"{_DESCRIPTION}\n\npod: {self.pod}\nweighting: {self.weighting} "
                f"min_obs: {self.min_obs} shrink_k: {self.shrink_k} topk: {self.topk} "
                f"ridge_alpha: {self.ridge_alpha}\nrule: {self._trade_rule_label()}")
