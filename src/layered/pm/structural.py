"""The structural trade layer — a deterministic map from driver views to legs.

**Why this exists.** Measured on the fresh board (`first-final-results.md`, and the
weight-vs-IC scatter that motivated it): the PM's leg weights track each driver's own
analyst IC almost perfectly (corr +0.80) but are *negatively* related to the thing that
actually converts to a good trade (corr −0.22 with trade IC). Read plainly, the PM is a
good *aggregator* and a bad *position constructor* — it sizes by "how convinced is the
analyst," not by "how does this view transmit to the tradeable instrument." The two are
different jobs, and the LLM is only good at the first.

So this layer splits them. The LLM keeps the job it does well — arbitrating seven reports
into a per-driver conviction (``ArbitratedView.drivers``). This module takes that block
and constructs the trade *structurally*: orient each driver onto the pod's rate axis by
its **declared** polarity, project to one scalar, and map that onto the pod's canonical
legs. Nothing here is fitted; every constant is the polarity sign already declared in the
pod YAML, the same signs the mechanical control and the disagreement measure read.

**Relationship to the mechanical PM.** ``mechanical_pm._trade`` already does exactly this
projection — but from *its own* consensus_blend driver block. This is the same arithmetic
pointed at the *LLM's* arbitrated block instead, so "structural trade on LLM views" is a
clean third arm between "LLM freehand trade" and "fully mechanical," and it isolates the
one thing the scatter says is broken: the view→leg map, holding the views fixed.

**Sign convention** matches ``trade_pnl``: a leg weight is signed on the instrument's
YIELD, P&L is ``Σ w·Δy``, so a positive weight bets the yield rises and net upward rate
pressure (projection > 0) takes positive legs.
"""
from __future__ import annotations

from typing import Mapping, Optional

import pandas as pd

from src.layered.contracts import StrategyTrade


def _sign(x: float) -> float:
    return 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)


def rate_axis_projection(drivers: Mapping[str, float],
                         polarity: Mapping[str, float]) -> float:
    """The driver block projected onto the pod's rate axis — mean oriented conviction.

    Each driver's conviction is multiplied by its declared polarity (so a −1-polarity
    driver whose own measurement falls contributes *upward* rate pressure) and averaged.
    A driver with no declared polarity is skipped, never assumed +1 — the same rule
    ``disagreement.oriented`` follows, so a pod reading more analysts than it owns cannot
    have the extras leak into its projection.
    """
    oriented = [float(polarity[d]) * float(v)
                for d, v in drivers.items() if d in polarity]
    return sum(oriented) / len(oriented) if oriented else 0.0


def structural_trade(drivers: Mapping[str, float], polarity: Mapping[str, float],
                     trade_config: Optional[dict], *, pod: str,
                     asof) -> Optional[StrategyTrade]:
    """Build the pod's canonical position from an arbitrated driver block, deterministically.

    Returns ``None`` — a genuine abstention, never a degraded row — when the pod declares
    no trade, has no instrument universe, or is an ``opposed`` pod without a declared
    leg-role decomposition (see below). Returns a gross-0 flat ``StrategyTrade`` when the
    projection is exactly zero: a decided flat, scored as a real zero, distinct from an
    abstention — the same line the LLM PM's ``flat`` flag and the mechanical PM both draw.

    ``opposed`` pods (a 2s10s slope) need to know which leg is the front and which is the
    long end to turn a scalar projection into a steepener/flattener. The pod config does
    not declare that by default, so — like the mechanical control — this abstains rather
    than invent the split. A pod MAY declare ``leg_roles: {front: DGS2, long: DGS10}`` in
    its ``trade`` block to opt in; then the projection sizes an equal-and-opposite trade
    (long-end leg takes the projection's sign, front-end the opposite). This is the one
    place the structural layer can do strictly more than the mechanical baseline, and only
    when the structure is *declared*, never fitted.
    """
    cfg = trade_config or {}
    if not cfg:
        return None
    universe = [str(s) for s in (cfg.get("universe") or [])]
    if not universe:
        return None

    asof_ts = asof if isinstance(asof, pd.Timestamp) else pd.Timestamp(asof)
    proj = rate_axis_projection(drivers, polarity)
    s = _sign(proj)
    conv = str(cfg.get("sign_convention", "") or "").strip().lower()

    def _finish(legs: dict[str, float], detail: str) -> StrategyTrade:
        gross = sum(abs(w) for w in legs.values())
        if gross > 0:
            legs = {inst: w / gross for inst, w in legs.items()}   # unit gross
        return StrategyTrade(
            strategy=pod, asof=asof_ts, legs=legs,
            conviction=float(min(1.0, abs(proj))),
            rationale=f"structural: rate-axis projection {proj:+.2f} → {detail}",
            risk={"tags": []},
        )

    if s == 0.0:
        return _finish({}, "flat (no net rate-axis pressure)")

    if conv == "opposed":
        roles = cfg.get("leg_roles") or {}
        front, long = str(roles.get("front", "")), str(roles.get("long", ""))
        if front not in universe or long not in universe or front == long:
            # No declared decomposition — abstain rather than fabricate a front/long split.
            return None
        # A steepener (projection up → higher long-end yields) is long the long end,
        # short the front. Equal and opposite so a parallel shift nets out.
        return _finish({long: s, front: -s}, f"{'steepen' if s > 0 else 'flatten'} on leg_roles")

    if conv == "same":
        max_legs = cfg.get("max_legs")
        if max_legs is not None and len(universe) > int(max_legs):
            # Choosing which same-signed legs to hold is an undeclared decision; abstain,
            # matching the LLM PM (rejects an over-legged trade) and the mechanical control.
            return None
        return _finish({inst: s for inst in universe}, "level (all legs one sign)")

    # single-instrument / unconstrained: one representative leg.
    return _finish({universe[-1]: s}, "single representative leg")
