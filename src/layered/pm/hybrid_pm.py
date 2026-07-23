"""v1 — the mechanical + LLM hybrid PM ("anchor and adjust").

The split of labour that the notebook argues for (§7.11, `docs/decisions.md` 2026-07-23):
combining analyst views into a sized trade is *precise arithmetic* the LLM is bad at
(numbers-only ≈ full; the prose is not load-bearing when the LLM does the weighting),
while judging *when this month is unusual* is exactly what reading the reports is for. So:

  * the **mechanical** ``RelevancePM`` sets a walk-forward baseline weight per analyst
    (its demonstrated trade relevance) — the precise part;
  * the **LLM** may nudge each baseline weight by a bounded multiplier in [0.5, 2.0],
    and *only* with a reason pointed at this month's report — the generative part.

This makes the report load-bearing **by construction**: the model can act only through
report-justified adjustments to the mechanical prior; it cannot re-weight from scratch,
flip a sign, or add a driver. The decisive experiment is v1-vs-v0: if the adjustments beat
the pure mechanical baseline, the reports finally earn their cost; if v1 ≈ v0, they don't
and the cheap mechanical PM stands.

Inherits the walk-forward weight schedule, the trade construction, and the JSONL schema
from ``RelevancePM``/``MechanicalPM``; overrides only the projection (to apply the
multipliers) and ``arbitrate`` (to make the one bounded LLM call).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from src.layered.contracts import ArbitratedView
from src.layered.pm.board import Meeting
from src.layered.pm.brief import render_brief
from src.layered.pm.disagreement import oriented
from src.layered.pm.mandate import render_mandate
from src.layered.pm.mechanical_pm import MechanicalPM
from src.layered.pm.relevance_pm import POD_DIR, RelevancePM

_MULT_LO, _MULT_HI = 0.5, 2.0

_ADJUST = (
    "You are the discretionary layer of a hybrid PM. A mechanical combiner has already set a "
    "baseline weight for each analyst — how much that analyst's view has historically mattered "
    "for THIS trade (its walk-forward trade relevance). Do NOT re-weight from scratch and do NOT "
    "restate the mechanical view. Nudge a baseline weight ONLY where this month's report gives a "
    "concrete reason the history has not captured yet: a stale read, a regime shift, a one-off "
    "catalyst, or a report that flags its driver is about to matter more or less than usual. "
    "Return a multiplier in [0.5, 2.0] on each analyst's baseline weight (1.0 = leave it), each "
    "with a one-line reason pointing at that analyst's report. Default to 1.0 — only deviate with "
    "a report-grounded reason. You cannot change a sign, drop a driver, or add one."
)


def submit_adjustments_tool(drivers: list[str]) -> dict:
    return {
        "name": "submit_adjustments",
        "description": "Nudge the mechanical baseline weights based on this month's reports.",
        "input_schema": {
            "type": "object",
            "properties": {
                "adjustments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "driver": {"type": "string", "enum": list(drivers)},
                            "multiplier": {"type": "number",
                                           "description": "in [0.5, 2.0]; 1.0 leaves the baseline weight"},
                            "why": {"type": "string", "description": "one line, from this driver's report"},
                        },
                        "required": ["driver", "multiplier", "why"],
                    },
                },
                "notes": {"type": "string"},
            },
            "required": ["adjustments"],
        },
    }


class HybridPM(RelevancePM):
    """RelevancePM baseline + a bounded per-analyst LLM multiplier."""

    def __init__(self, pod: str, config: dict, llm=None, **kw):
        super().__init__(pod, config, **kw)
        self.llm = llm
        self.use_memory = False
        self._current_mult: dict[str, float] = {}
        self.last_adjust: dict = {}

    @classmethod
    def from_pod(cls, pod: str, llm=None, pod_dir: Optional[Path] = None, **kw) -> "HybridPM":
        import yaml
        path = (pod_dir or POD_DIR) / f"{pod}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"no pod spec for {pod!r} at {path}")
        return cls(pod=pod, config=yaml.safe_load(path.read_text()) or {}, llm=llm, **kw)

    # ── baseline (v0) weights for a meeting, or equal during warm-up ──────────
    def _baseline(self, meeting: Meeting) -> dict[str, float]:
        o = oriented(meeting, self.polarity)
        w0 = self._weights.get(pd.Timestamp(meeting.asof), {})
        if any(abs(w0.get(d, 0.0)) > 0 for d in o):
            return {d: w0.get(d, 0.0) for d in o}
        return {d: 1.0 for d in o}                              # warm-up: equal

    def _system_prompt(self) -> str:
        return f"{render_mandate(self.config)}\n\n{_ADJUST}"

    # ── the one bounded LLM call → per-analyst multipliers ────────────────────
    def _ask_multipliers(self, meeting: Meeting):
        base = self._baseline(meeting)
        tot = sum(abs(v) for v in base.values()) or 1.0
        lines = "\n".join(f"  {d}: baseline weight {base[d] / tot:+.2f}"
                          for d in sorted(base, key=lambda d: -abs(base[d])))
        user = (f"{render_brief(meeting, drivers=self.reads)}\n\n"
                f"Mechanical baseline weights (walk-forward trade relevance; sum of |weights| = 1):\n"
                f"{lines}\n\nReturn a multiplier in [0.5, 2.0] on each analyst's baseline weight — "
                f"1.0 unless this month's report gives a reason to move it.")
        raw = self.llm.complete(system=self._system_prompt(), user=user,
                                tool=submit_adjustments_tool(self.listens_to))
        parsed = json.loads(raw, strict=False)
        mult, why = {}, {}
        for a in (parsed.get("adjustments") or []):
            d = a.get("driver")
            if d in self.listens_to:
                mult[d] = min(_MULT_HI, max(_MULT_LO, float(a.get("multiplier", 1.0))))
                why[d] = str(a.get("why", ""))
        return base, mult, why, str(parsed.get("notes", "")), raw

    # ── projection applies baseline × multiplier ─────────────────────────────
    def _rate_projection(self, m: Meeting) -> float:
        o = oriented(m, self.polarity)
        if not o:
            return 0.0
        base = self._baseline(m)
        aw = {d: base.get(d, 0.0) * self._current_mult.get(d, 1.0) for d in o}
        num = sum(aw[d] * o[d] for d in o)
        den = sum(abs(aw[d]) for d in o)
        return float(num / den) if den > 0 else float(sum(o.values()) / len(o))

    def arbitrate(self, meeting: Meeting) -> ArbitratedView:
        if self.llm is None:
            raise RuntimeError(f"{self.pod}: HybridPM needs an llm client for the adjustment call.")
        try:
            base, mult, why, notes, raw = self._ask_multipliers(meeting)
        except Exception as e:  # noqa: BLE001 — one bad call falls back to pure v0, never crashes
            base, mult, why, notes, raw = self._baseline(meeting), {}, {}, \
                f"[adjustment failed: {type(e).__name__}: {e}]", None
        self._current_mult = mult
        self.last_adjust = {"multipliers": mult, "why": why, "notes": notes, "llm_raw": raw}
        # MechanicalPM.arbitrate builds the driver block + the trade; the trade goes through
        # our overridden _rate_projection, so it is the multiplier-adjusted projection. It also
        # sets self.last_raw to the trade-shaped reply trade_pnl reads — leave that as-is.
        return MechanicalPM.arbitrate(self, meeting)

    def why(self, _parsed_or_raw=None) -> dict[str, str]:
        return {d: w for d, w in (self.last_adjust.get("why") or {}).items()}
