"""Offline check for the input-modality experiment (no network, no keys, no cost).

The comparison is only valid if the arms differ in ONE thing — the LLM's input.
A recording StubLLM captures the exact prompt each arm builds; a tiny in-memory
FOMC corpus provides point-in-time text. We assert:

  * vector      : prompt carries the numeric reading, no FOMC text
  * text        : prompt carries the FOMC text, not the numeric reading
  * text+vector : prompt carries both
  * point-in-time: the text served is the latest with release_date <= asof, and a
                   future-dated document is NEVER served (no lookahead)
  * the SYSTEM prompt is identical across arms (only the input differs)
"""
from __future__ import annotations

import json
import re
import sys

import pandas as pd

from src.layered.analysts.macro_rates import macro_rates_analysts
from src.layered.synthetic import generate
from src.layered.timeline import AsOf

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name}" + (f" — {detail}" if detail and not cond else ""))


class RecordingStub:
    """Fake LLM: records the last (system, user) it was handed, echoes a valid view."""

    def __init__(self) -> None:
        self.last_system: str | None = None
        self.last_user: str | None = None

    def complete(self, system: str, user: str) -> str:
        self.last_system, self.last_user = system, user
        m = re.search(r'"direction":\s*"(\w+)"', user)
        direction = m.group(1) if m and m.group(1) in ("up", "down", "flat") else "flat"
        return json.dumps({"direction": direction, "conviction": 0.5, "reasoning": "stub"})


class FakeCorpus:
    """In-memory stand-in with the FomcCorpus interface: as_of() + doc_type."""

    doc_type = "statement"

    def __init__(self, items: list[tuple[str, str]]) -> None:
        self._items = sorted((pd.Timestamp(r), t) for r, t in items)

    def as_of(self, asof) -> str | None:
        asof = pd.Timestamp(asof)
        hit = None
        for r, t in self._items:
            if r <= asof:
                hit = t
            else:
                break
        return hit


# One synthetic world at a fixed asof (no network).
macro, prices = generate("2022-01-01", "2022-06-30", regime="hawkish")
asof = prices.index[-1]
world = AsOf(asof=asof, macro=macro, prices=prices)

NEW = "NEW-STATEMENT-TOKEN"
OLD = "OLD-STATEMENT-TOKEN"
FUTURE = "FUTURE-STATEMENT-TOKEN"
corpus = FakeCorpus([
    (asof - pd.Timedelta(days=10), f"{NEW} most recent statement"),
    (asof - pd.Timedelta(days=400), f"{OLD} stale statement"),
    (asof + pd.Timedelta(days=5), f"{FUTURE} not yet released"),
])


def user_prompt_for(mode: str) -> str:
    stub = RecordingStub()
    analyst = macro_rates_analysts(stub, input_mode=mode, text_source=corpus)[0]  # InflationAnalyst
    analyst.form_view(world)
    return stub.last_user or ""


uv = user_prompt_for("vector")
ut = user_prompt_for("text")
utv = user_prompt_for("text+vector")

check("vector: carries numeric reading", "Deterministic reading" in uv and '"direction"' in uv)
check("vector: carries no FOMC text", NEW not in uv)
check("text: carries FOMC text", NEW in ut)
check("text: drops the numeric reading", "Deterministic reading" not in ut)
check("text+vector: carries both", NEW in utv and "Deterministic reading" in utv)
check("point-in-time: serves latest<=asof (new, not stale)", NEW in ut and OLD not in ut)
check("no lookahead: future document never served", FUTURE not in ut and FUTURE not in utv)

# System prompt must be identical across arms — only the input may differ.
s_v, s_t = RecordingStub(), RecordingStub()
macro_rates_analysts(s_v, input_mode="vector", text_source=corpus)[0].form_view(world)
macro_rates_analysts(s_t, input_mode="text", text_source=corpus)[0].form_view(world)
check("system prompt identical across arms", s_v.last_system == s_t.last_system)

# Sanity: vector-arm prompt is byte-for-byte the legacy prompt (no text_source consulted).
legacy_stub = RecordingStub()
macro_rates_analysts(legacy_stub, input_mode="vector", text_source=None)[0].form_view(world)
check("vector arm unchanged whether or not a text_source is attached",
      legacy_stub.last_user == uv)

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
