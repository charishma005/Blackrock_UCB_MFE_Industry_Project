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
- Do NOT let any single |weight| exceed risk_limits.max_weight_per_name, and do
  NOT let sum(|weight|) exceed risk_limits.max_gross_leverage. These numeric
  bounds are in the payload; the risk layer will re-clip you anyway, so
  proposing sizes beyond them just wastes budget.
- Return ONLY JSON: {"weights": {"SYMBOL": float, ...}, "reasoning": "<=80 words"}"""


class PortfolioManager:
    def __init__(self, config: PMConfig | None = None, llm_client=None):
        self.config = config or PMConfig()
        self.llm = llm_client
        # count silent LLM fallbacks so a backtest can report how often the LLM
        # PM actually failed instead of the failures vanishing into mechanical.
        self.llm_calls = 0
        self.llm_failures = 0

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
            # explicit numeric bounds the prompt refers to (falls back to the
            # RiskConfig defaults if the risk layer was disabled this run).
            "risk_limits": {
                "max_weight_per_name": risk_diagnostics.get("max_weight_per_name", 0.25),
                "max_gross_leverage": risk_diagnostics.get("max_gross_leverage", 1.5),
            },
            "agent_signals": {
                agent: {sym: {"signal": s["signal"], "confidence": s["confidence"], "reasoning": s.get("reasoning", "")[:200]}
                        for sym, s in sigs.items()}
                for agent, sigs in agent_signals.items()
            },
        }

        self.llm_calls += 1
        user = json.dumps(payload, default=str)
        last_err: Exception | None = None
        # one retry: on a parse failure, tell the model exactly what broke and
        # re-ask, since a malformed response is often fixed by a nudge.
        for attempt in range(2):
            system = PM_SYSTEM_PROMPT if attempt == 0 else (
                PM_SYSTEM_PROMPT
                + f"\n\nYour previous reply could not be parsed ({last_err}). "
                  "Return ONLY the JSON object, no prose, no code fences."
            )
            try:
                raw = self.llm.complete(system=system, user=user)
                parsed = json.loads(raw)
                w = pd.Series(parsed.get("weights", {}), dtype=float).reindex(instruments).fillna(0.0)
                return w, parsed.get("reasoning", "")
            except Exception as e:  # noqa: BLE001 — retry boundary, then fall back
                last_err = e

        self.llm_failures += 1
        return risk_adjusted_weights, f"LLM PM failed ({last_err}); fell back to mechanical blend"
