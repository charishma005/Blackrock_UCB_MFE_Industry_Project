"""The mechanical PM — a deterministic arbitrator, the control the LLM PM must beat.

Why this exists. ``pm_bench`` already grades the LLM PM's *driver block* against a
mechanical control (``consensus_blend`` → the ``ic_mech`` column): half a driver's own
analyst, half the oriented panel. But the PM's other output — the ``StrategyTrade`` that
crosses the PM→fund seam, and the one output where the first duration run found "no
detectable edge" (t=+0.08) — is graded by ``trade_pnl`` against *nothing at all*. A P&L
of t=+0.08 is only interpretable next to what a rule got on the same board.

So this is that rule. It reads the same ``Meeting`` the LLM PM reads, off the same board,
and emits a full ``ArbitratedView`` — a driver block AND a trade — by arithmetic alone,
no model and no spend. Written to the same JSONL schema (see ``run_pm_mechanical``), it
is scored by the exact same ``pm_bench.benchmark`` and ``trade_pnl`` the LLM run is, so
"does arbitration add anything?" becomes a head-to-head on one clock and one outcome.

This is the sibling project's discipline, transferred: `watching-crowding-build`'s
flat ensemble kept a mechanical PM as a *named baseline the LLM had to beat*, and its
comomentum track's whole identification rested on running the identical estimator on a
neutral control (D56) and reporting the two bare. The design note is logged in
``docs/decisions.md`` (2026-07-22, "Mechanical-PM trade baseline").

What it is NOT. It is not a rival PM and it takes no view the config does not license.
Every judgment it makes is declared in the pod YAML and never fitted to outcomes — the
same line the polarity signs and the LLM mandate are held to. Two deliberate honesties:

  * **The driver block is ``consensus_blend`` per meeting** — the batched form lives in
    ``pm_bench`` and is the same arithmetic, so grading this run's driver block should
    reproduce that run's ``ic_mech`` column. That is a consistency check, not new signal:
    the driver-space control already existed; this exists for the trade.
  * **An ``opposed`` (slope) pod gets no mechanical trade.** Turning a panel into a 2s10s
    steepener needs a front-end-vs-long-end split that the pod config does not declare,
    so fabricating one would be exactly the unaudited rule this baseline refuses to be.
    It abstains and records why, rather than inventing a decomposition. A ``same`` (level)
    pod and a single-instrument pod both have an unambiguous mapping and do get a trade.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from src.layered.contracts import ArbitratedView, StrategyTrade
from src.layered.pm.board import Meeting, ViewBoard
from src.layered.pm.brief import render_brief
from src.layered.pm.disagreement import oriented, panel_disagreement

POD_DIR = Path(__file__).parent / "pods"

# How the driver's own analyst and the oriented panel are blended into the mechanical
# driver conviction. 0.5 is ``pm_bench.consensus_blend``'s default, declared in advance
# there and matched here so the two are one number computed in two places, never two.
_OWN_WEIGHT = 0.5

_DESCRIPTION = (
    "MECHANICAL PM — a deterministic control, no model. The driver block is a fixed "
    "blend (half each driver's own analyst, half the oriented panel mean). The trade is "
    "a fixed rule: project the panel onto the pod's rate axis and take the pod's "
    "canonical position scaled by that projection. Every constant is declared, none is "
    "fitted. It exists so the LLM PM's trade P&L has a baseline to be read against."
)


def _sign(x: float) -> float:
    return 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)


class MechanicalPM:
    """One pod, one meeting, one arbitrated view — by arithmetic, not a model.

    Duck-types the slice of ``LLMPM`` that ``build_board`` and the run loop touch
    (``reads``, ``board_kwargs``, ``clock_freq``, ``polarity``, ``listens_to``,
    ``trade_config``, ``answer_space``, ``build_inputs``, ``arbitrate``, ``why``,
    ``last_raw``), so it runs through the same harness with no special-casing.
    """

    def __init__(self, pod: str, config: dict):
        self.pod = pod
        self.config = config
        # Present so the runner and meta writer can treat it exactly like an LLMPM.
        self.use_memory = False
        self.blind: Optional[str] = None
        self.last_raw: Optional[str] = None

    @classmethod
    def from_pod(cls, pod: str, pod_dir: Optional[Path] = None) -> "MechanicalPM":
        path = (pod_dir or POD_DIR) / f"{pod}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"no pod spec for {pod!r} at {path}")
        return cls(pod=pod, config=yaml.safe_load(path.read_text()) or {})

    # ── identity (mirrors LLMPM) ─────────────────────────────────────────────
    @property
    def listens_to(self) -> list[str]:
        return list((self.config.get("listens_to") or {}).keys())

    @property
    def polarity(self) -> dict[str, float]:
        block = self.config.get("listens_to") or {}
        return {d: float((cfg or {}).get("polarity", 1.0)) for d, cfg in block.items()}

    @property
    def reads(self) -> Optional[list[str]]:
        r = self.config.get("reads")
        if r is None:
            return list(self.listens_to)
        if isinstance(r, str):
            return None if r.strip().lower() == "all" else [r]
        return [str(x) for x in r]

    @property
    def trade_config(self) -> dict:
        return self.config.get("trade") or {}

    @property
    def answer_space(self) -> str:
        # The mechanical driver block is always in driver space (it grades against each
        # driver's own level). A pod that declares `rate` is describing what it asks the
        # *model* for; the arithmetic control does not need it, and forcing a rate-space
        # reading here would just re-invert what pm_bench then re-inverts back.
        return "driver"

    @property
    def clock_freq(self) -> str:
        return self.config.get("clock_freq", "ME")

    @property
    def board_kwargs(self) -> dict:
        b = self.config.get("board") or {}
        return {"stale_after_days": int(b.get("stale_after_days", 45)),
                "expire_after_days": int(b.get("expire_after_days", 95))}

    @property
    def memory(self):
        return None

    def _system_prompt(self) -> str:
        """No prompt is sent. Returned so the run meta records what the rule was."""
        return f"{_DESCRIPTION}\n\npod: {self.pod}\nrule: {self._trade_rule_label()}"

    def _trade_rule_label(self) -> str:
        conv = str(self.trade_config.get("sign_convention", "") or "").strip().lower()
        if not self.trade_config:
            return "driver-space only; no trade"
        if conv == "opposed":
            return "opposed/slope pod — no mechanical trade (decomposition not declared)"
        if conv == "same":
            return "same/level pod — both legs sign(rate-axis projection)"
        return "single/level pod — representative leg sign(rate-axis projection)"

    # ── inputs (mirrors LLMPM) ───────────────────────────────────────────────
    def build_inputs(self, board: ViewBoard, meeting) -> Meeting:
        return board.at(meeting)

    def _user_prompt(self, meeting: Meeting, memory=None) -> str:
        """The brief the LLM PM would have seen. Not consumed by the arithmetic; kept so
        ``brief_sha256`` still records the panel this decision was formed against, exactly
        as the LLM run does."""
        return render_brief(meeting, drivers=self.reads)

    # ── the arithmetic ───────────────────────────────────────────────────────
    def _driver_block(self, m: Meeting) -> dict[str, float]:
        """Per-meeting ``consensus_blend``: half own analyst, half oriented panel.

        Identical arithmetic to ``pm_bench.consensus_blend`` (batched over dates there,
        one meeting here). Only drivers the pod both listens to and has a present view
        for get an entry — an absent driver stays out, never filled with 0.0, the same
        rule the LLM PM's ``_parse_drivers`` and its whole layer hold to.
        """
        pol = self.polarity
        orient = oriented(m, pol)                 # {driver: polarity * signed_conviction}
        if not orient:
            return {}
        panel_mean = sum(orient.values()) / len(orient)
        out: dict[str, float] = {}
        for d, e in m.entries.items():
            if d not in pol or not e.present:
                continue
            own = e.view.signed_conviction
            blended = _OWN_WEIGHT * own + (1.0 - _OWN_WEIGHT) * pol[d] * panel_mean
            out[d] = float(min(1.0, max(-1.0, blended)))
        return out

    def _rate_projection(self, m: Meeting) -> float:
        """The panel projected onto the pod's axis — the mean oriented conviction.

        ``> 0`` is net upward pressure on the axis (for a rates pod: yields up). This is
        the single scalar every mechanical trade is built from, and it is the same
        ``oriented`` map the disagreement measure uses, only averaged instead of split.
        """
        orient = oriented(m, self.polarity)
        return sum(orient.values()) / len(orient) if orient else 0.0

    def _trade(self, m: Meeting) -> Optional[StrategyTrade]:
        """The pod's canonical position, scaled by the rate-axis projection.

        Sign convention matches ``trade_pnl``: a leg weight is signed on the instrument's
        YIELD and P&L is ``Σ w·Δy``, so a positive weight bets the yield rises. Net upward
        rate pressure (projection > 0) therefore takes positive legs.
        """
        cfg = self.trade_config
        if not cfg:
            return None
        universe = [str(s) for s in (cfg.get("universe") or [])]
        if not universe:
            return None
        conv = str(cfg.get("sign_convention", "") or "").strip().lower()
        if conv == "opposed":
            # A slope trade needs a front-vs-long split the config does not declare.
            # Abstaining is the honest baseline; inventing the split would make this the
            # unaudited rule it exists not to be.
            return None

        proj = self._rate_projection(m)
        s = _sign(proj)
        if s == 0.0:
            # No net pressure at all: a deliberate flat, scored as a real zero, not an
            # abstention — the same distinction the LLM PM's `flat` flag draws.
            return StrategyTrade(strategy=self.pod, asof=m.asof, legs={},
                                 conviction=0.0,
                                 rationale="mechanical: no net rate-axis pressure",
                                 risk={"tags": []})

        if conv == "same":
            max_legs = cfg.get("max_legs")
            if max_legs is not None and len(universe) > int(max_legs):
                # A same-sign position over more instruments than `max_legs` permits
                # requires choosing WHICH legs to hold — an undeclared decision. The
                # LLM PM rejects an over-legged trade outright rather than trimming it
                # (`llm_pm._parse_trade`), and the opposed branch above abstains rather
                # than fabricate a missing rule; do the same here instead of silently
                # taking all of them. (No-op for the shipped 2-instrument `duration`.)
                return None
            legs = {inst: s for inst in universe}          # level view: all legs one sign
        else:
            legs = {universe[-1]: s}                        # single/level: representative leg
        gross = sum(abs(w) for w in legs.values())
        legs = {inst: w / gross for inst, w in legs.items()}   # unit gross

        return StrategyTrade(
            strategy=self.pod, asof=m.asof, legs=legs,
            conviction=float(min(1.0, abs(proj))),
            rationale=(f"mechanical: rate-axis projection {proj:+.2f} → "
                       f"{'up' if s > 0 else 'down'} on {self._trade_rule_label()}"),
            risk={"tags": []},
        )

    def arbitrate(self, meeting: Meeting) -> ArbitratedView:
        """One meeting → one ``ArbitratedView``, deterministically.

        Sets ``last_raw`` to a synthesized tool-shaped reply so ``trade_pnl.load_trades``
        reports this run as fully emitted with nothing dropped by grounding — which is
        true, since the arithmetic only ever names in-universe instruments.
        """
        drivers = self._driver_block(meeting)
        trade = self._trade(meeting)

        if not drivers:
            self.last_raw = None
            return ArbitratedView(asof=meeting.asof, drivers={}, disagreement=0.0,
                                  notes="mechanical: no present driver to blend")

        legs = list(trade.legs.items()) if trade is not None else []
        self.last_raw = json.dumps({
            "notes": _DESCRIPTION,
            "drivers": [{"driver": d, "conviction": v, "why": "mechanical blend"}
                        for d, v in drivers.items()],
            "trade": None if trade is None else {
                "flat": trade is not None and not legs,
                "legs": [{"instrument": k, "weight": w} for k, w in legs],
                "conviction": trade.conviction,
                "rationale": trade.rationale,
            },
        })

        return ArbitratedView(
            asof=meeting.asof,
            drivers=drivers,
            disagreement=panel_disagreement(meeting, self.polarity),
            notes=(f"{_DESCRIPTION} Rate-axis projection {self._rate_projection(meeting):+.2f}."),
            trade=trade,
        )

    def why(self, parsed_or_raw) -> dict[str, str]:
        """Per-driver one-liners, for the run log — parity with ``LLMPM.why``."""
        try:
            parsed = (json.loads(parsed_or_raw) if isinstance(parsed_or_raw, str)
                      else parsed_or_raw) or {}
        except Exception:  # noqa: BLE001
            return {}
        return {str(it["driver"]): str(it.get("why", ""))
                for it in parsed.get("drivers", []) if isinstance(it, dict) and it.get("driver")}
