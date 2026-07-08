"""Base class for all investor agents.

Pattern (borrowed from virattt/ai-hedge-fund, generalized to multi-asset):
  Phase 1: fetch data + deterministic Python analysis  -> facts dict
  Phase 2: one LLM call, persona-conditioned           -> InvestorSignal per instrument

Persona conditioning (modification #4): each agent loads a distilled persona
spec from agents/personas/<name>.yaml — beliefs, decision rules, vocabulary —
generated offline from primary sources (letters, interviews). The spec text is
injected into the system prompt.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from src.instruments import AssetClass, Instrument

PERSONA_DIR = Path(__file__).parent / "personas"


class InvestorSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float  # 0-100
    reasoning: str


class BaseInvestorAgent(ABC):
    name: str = "base"
    covers: set[AssetClass] = set()

    def __init__(self, llm_client=None):
        self.llm = llm_client
        self.persona = self._load_persona()

    def _load_persona(self) -> dict:
        path = PERSONA_DIR / f"{self.name}.yaml"
        if path.exists():
            return yaml.safe_load(path.read_text())
        return {}

    def instruments_for(self, universe: list[Instrument]) -> list[Instrument]:
        return [i for i in universe if i.asset_class in self.covers]

    # ── Phase 1: pure Python ────────────────────────────────────────────
    @abstractmethod
    def compute_facts(self, instrument: Instrument, data: dict) -> dict:
        """Deterministic analysis. Return a JSON-serializable facts dict."""

    # ── Phase 2: LLM judgment ───────────────────────────────────────────
    def system_prompt(self) -> str:
        p = self.persona
        parts = [f"You are {p.get('display_name', self.name)}."]
        if p.get("core_beliefs"):
            parts.append("Core beliefs:\n" + "\n".join(f"- {b}" for b in p["core_beliefs"]))
        if p.get("decision_rules"):
            parts.append("Decision rules:\n" + "\n".join(f"- {r}" for r in p["decision_rules"]))
        if p.get("vocabulary"):
            parts.append("Use this vocabulary naturally: " + ", ".join(p["vocabulary"]))
        parts.append(
            "Given the computed facts, return ONLY JSON: "
            '{"signal": "bullish"|"bearish"|"neutral", "confidence": 0-100, '
            '"reasoning": "<=100 words"}'
        )
        return "\n\n".join(parts)

    def judge(self, instrument: Instrument, facts: dict) -> InvestorSignal:
        """One LLM call. Falls back to neutral on any failure."""
        if self.llm is None:
            return InvestorSignal(signal="neutral", confidence=0, reasoning="No LLM configured")
        try:
            raw = self.llm.complete(
                system=self.system_prompt(),
                user=f"Instrument: {instrument.symbol} ({instrument.name})\n\nFacts:\n{json.dumps(facts, indent=2, default=str)}",
            )
            return InvestorSignal(**json.loads(raw))
        except Exception as e:  # noqa: BLE001
            return InvestorSignal(signal="neutral", confidence=0, reasoning=f"LLM error: {e}")

    # ── Entry point ─────────────────────────────────────────────────────
    def run(self, universe: list[Instrument], data_by_symbol: dict[str, dict]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for inst in self.instruments_for(universe):
            if not inst.tradeable:
                continue
            facts = self.compute_facts(inst, data_by_symbol.get(inst.symbol, {}))
            out[inst.symbol] = self.judge(inst, facts).model_dump()
        return out
