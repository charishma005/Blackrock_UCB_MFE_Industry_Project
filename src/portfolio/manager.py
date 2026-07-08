"""LLM Portfolio Manager (modification #1).

Two modes, so you can run the comparison experiment "mechanical blend vs LLM
synthesis":

  * mode="mechanical" — the risk-adjusted blended weights pass through
    unchanged. Deterministic, free, the baseline.
  * mode="llm" — an LLM sees the per-agent signals (with reasoning), the macro
    regime, and the risk manager's diagnostics, then outputs a final target
    weight per instrument. This is where an LLM can do things a linear blend
    can't: resolve disagreements contextually, respect the regime, or refuse
    to size into a name the risk layer flagged.

The LLM's output is still passed BACK through the risk manager afterwards, so
it can never violate the vol target or leverage cap — the LLM proposes, the
deterministic risk layer disposes. That ordering is the whole safety argument.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

import pandas as pd

PMMode = Literal["mechanical", "llm"]


@dataclass
class PMConfig:
    mode: PMMode = "mechanical"


PM_SYSTEM_PROMPT = """You are the portfolio manager of a multi-asset fund.
You receive: (1) signals from several investor agents, each with a signal,
confidence, and reasoning; (2) the current macro regime; (3) risk-manager
diagnostics. Your job is to output a final target weight per instrument.

Rules:
- Weights are fractions of the portfolio; positive = long, negative = short.
- Respect the macro regime — do not fight it without strong bottom-up conviction.
- When agents disagree, weight the more credible/specific reasoning, not just
  the higher confidence number.
- Do not exceed the per-name and gross limits the risk manager reports; the
  risk layer will re-clip you anyway, so proposing wild sizes just wastes the
  budget.
- Return ONLY JSON: {"weights": {"SYMBOL": float, ...}, "reasoning": "<=80 words"}"""


class PortfolioManager:
    def __init__(self, config: PMConfig | None = None, llm_client=None):
        self.config = config or PMConfig()
        self.llm = llm_client

    def decide(
        self,
        risk_adjusted_weights: pd.Series,
        agent_signals: dict[str, dict[str, dict]],
        regime: dict,
        risk_diagnostics: dict,
    ) -> tuple[pd.Series, str]:
        """Return (final_weights, reasoning). Mechanical mode is a pass-through."""
        if self.config.mode == "mechanical" or self.llm is None:
            return risk_adjusted_weights, "mechanical pass-through (no LLM PM)"

        instruments = list(risk_adjusted_weights.index)
        payload = {
            "instruments": instruments,
            "risk_adjusted_starting_weights": {k: round(float(v), 4) for k, v in risk_adjusted_weights.items()},
            "macro_regime": regime,
            "risk_diagnostics": risk_diagnostics,
            "agent_signals": {
                agent: {sym: {"signal": s["signal"], "confidence": s["confidence"], "reasoning": s.get("reasoning", "")[:200]}
                        for sym, s in sigs.items()}
                for agent, sigs in agent_signals.items()
            },
        }
        try:
            raw = self.llm.complete(system=PM_SYSTEM_PROMPT, user=json.dumps(payload, default=str))
            parsed = json.loads(raw)
            w = pd.Series(parsed.get("weights", {}), dtype=float).reindex(instruments).fillna(0.0)
            return w, parsed.get("reasoning", "")
        except Exception as e:  # noqa: BLE001 — fall back to mechanical on any failure
            return risk_adjusted_weights, f"LLM PM failed ({e}); fell back to mechanical blend"
