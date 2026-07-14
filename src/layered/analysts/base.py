"""Single-driver analyst — isolated expertise, one force in the world.

"The specialty is deliberately narrow: not 'macro' but a single economic driver
— inflation, unemployment, the Fed's balance sheet ... — or a single point on a
curve." An analyst sees the world ONLY through its driver and is asked to be
right about that one thing, not to trade and not to agree with anyone else.
Isolation is deliberate: it is what keeps each view expert and uncontaminated.

The two-phase pattern is borrowed from ``src/agents/base.py`` but the output is
a ``DriverView`` (a claim about a driver), not an ``InvestorSignal`` (a call on
an instrument) — the analyst never names a trade. That is the PM's job.

    Phase 1  read()   deterministic reading of the driver from its own data only
    Phase 2  judge()  optional LLM refinement of direction/conviction/reasoning

With no LLM configured the Phase-1 reading is returned verbatim, so the whole
layer runs offline and deterministically.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import yaml

from src.layered.contracts import DriverView
from src.layered.timeline import AsOf

PERSONA_DIR = Path(__file__).parent / "personas"


class SingleDriverAnalyst(ABC):
    """Base class for an isolated single-driver analyst.

    Subclasses declare ``driver`` (the name of their one force) and the FRED /
    market series they are allowed to read, then implement ``read`` to turn that
    data into a ``DriverView``. The base class enforces isolation by handing the
    subclass an ``AsOf`` and nothing else — an analyst has no route to another
    driver's data or to any other analyst's view.
    """

    driver: str = "base"
    horizon_days: int = 63           # default view horizon (~one quarter)
    inputs: tuple[str, ...] = ()     # series/symbols this analyst is allowed to read

    def __init__(self, llm_client=None):
        self.llm = llm_client
        self.persona = self._load_persona()

    def _load_persona(self) -> dict:
        path = PERSONA_DIR / f"{self.driver}.yaml"
        if path.exists():
            return yaml.safe_load(path.read_text()) or {}
        return {}

    # ── Phase 1: deterministic reading of the driver (pure Python) ──────────
    @abstractmethod
    def read(self, world: AsOf) -> DriverView:
        """Measure the driver from this analyst's own inputs and return a view.

        Must depend ONLY on ``self.inputs`` accessed through ``world`` (the AsOf
        gate), so the analyst stays isolated and free of look-ahead.
        """

    # ── Phase 2: optional LLM refinement ────────────────────────────────────
    def _system_prompt(self) -> str:
        p = self.persona
        parts = [
            f"You are a specialist analyst covering exactly one driver: "
            f"{p.get('display_name', self.driver)}. You have no view on anything "
            f"else and you never name a trade — you only report on your driver."
        ]
        if p.get("mandate"):
            parts.append("Mandate:\n" + "\n".join(f"- {m}" for m in p["mandate"]))
        parts.append(
            "You are given a deterministic reading of your driver. Refine it into "
            "your best current view. Return ONLY JSON: "
            '{"direction": "up"|"down"|"flat", "conviction": 0.0-1.0, '
            '"reasoning": "<=60 words"}'
        )
        return "\n\n".join(parts)

    def _refine(self, view: DriverView, reading: dict) -> DriverView:
        """One LLM call. Falls back to the Phase-1 view on any failure."""
        if self.llm is None:
            return view
        try:
            raw = self.llm.complete(
                system=self._system_prompt(),
                user=(
                    f"Driver: {self.driver}\n"
                    f"Deterministic reading (as of {view.asof.date()}):\n"
                    f"{json.dumps(reading, indent=2, default=str)}"
                ),
            )
            parsed = json.loads(raw)
            return view.model_copy(update={
                "direction": parsed.get("direction", view.direction),
                "conviction": float(parsed.get("conviction", view.conviction)),
                "reasoning": parsed.get("reasoning", view.reasoning),
            })
        except Exception as e:  # noqa: BLE001 — never let one bad call crash the meeting
            return view.model_copy(update={
                "reasoning": f"{view.reasoning} (LLM refine failed: {e})"
            })

    # ── Entry point ─────────────────────────────────────────────────────────
    def form_view(self, world: AsOf) -> DriverView:
        """Full analyst pass: Phase-1 reading, then optional Phase-2 refinement."""
        view = self.read(world)
        return self._refine(view, view.model_dump(mode="json"))
