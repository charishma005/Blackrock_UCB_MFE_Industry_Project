"""The output contract: declared gaps, memory bookkeeping, and back-compatibility.

No model is called — a stub returns the tool payload verbatim, so these exercise the
parsing and validation in ``form_view_from`` rather than the model's judgment.
"""
from __future__ import annotations

import json

import pandas as pd

from src.layered.analysts.llm_analyst import LLMAnalyst
from src.layered.contracts import DriverView, FeatureSet, SeriesFeature
from src.layered.evaluation.report_quality import conviction_response
from src.layered.text import TextContext

ASOF = pd.Timestamp("2023-06-01")


class _StubLLM:
    """Returns a canned tool payload; records nothing else."""

    def __init__(self, payload: dict):
        self.payload = payload

    def complete(self, system: str, user: str, tool: dict) -> str:  # noqa: ARG002
        return json.dumps(self.payload)


def _features() -> FeatureSet:
    return FeatureSet(
        driver="inflation", asof=ASOF,
        series=[SeriesFeature(name="headline_cpi_yoy", values=[2.0, 2.4])],
        level_feature="headline_cpi_yoy",
    )


def _payload(**over) -> dict:
    base = {
        "report": "Core momentum is firming across the recent prints.",
        "key_evidence": ["headline_cpi_yoy"],
        "falsifier": "core 3m annualized falls below 2%",
        "missing_inputs": [],
        "direction": "up",
        "conviction": 0.6,
    }
    base.update(over)
    return base


def _analyst(payload: dict, use_memory: bool = False) -> LLMAnalyst:
    return LLMAnalyst.from_persona("inflation", llm=_StubLLM(payload),
                                   text_selector=None, use_memory=use_memory)


def _form(analyst: LLMAnalyst) -> DriverView:
    return analyst.form_view_from(_features(), TextContext(driver="inflation", available=False))


# ── declared gaps ───────────────────────────────────────────────────────────
def test_valid_missing_input_survives():
    v = _form(_analyst(_payload(missing_inputs=[
        {"driver": "labor_tightness", "why": "wage pass-through"}])))
    assert [m.driver for m in v.missing_inputs] == ["labor_tightness"]
    assert v.missing_inputs[0].why == "wage pass-through"


def test_unknown_driver_is_dropped_not_degraded():
    """Grounded the same mechanical way key_evidence is — a bad citation costs the
    citation, never the whole view."""
    v = _form(_analyst(_payload(missing_inputs=[
        {"driver": "oil_prices", "why": "energy pass-through"},
        {"driver": "labor_tightness", "why": "wage pass-through"}])))
    assert not v.degraded
    assert [m.driver for m in v.missing_inputs] == ["labor_tightness"]


def test_own_driver_is_dropped():
    """"I lack my own data" is not a link a PM can route on."""
    v = _form(_analyst(_payload(missing_inputs=[{"driver": "inflation", "why": "more data"}])))
    assert v.missing_inputs == []


def test_malformed_entries_do_not_crash_the_view():
    v = _form(_analyst(_payload(missing_inputs=["labor_tightness", None, 42])))
    assert not v.degraded and v.missing_inputs == []


# ── memory bookkeeping ──────────────────────────────────────────────────────
def test_memory_holds_the_last_formed_view():
    a = _analyst(_payload(), use_memory=True)
    assert a.memory is None
    v = _form(a)
    assert a.memory is not None and a.memory.direction == v.direction


def test_degraded_view_never_becomes_the_memory():
    """Same rule as the carry-forward cache: a failed call is retried at the next
    release, not frozen and replayed back at the model as its own view."""
    a = _analyst(_payload(), use_memory=True)
    good = _form(a)
    a.llm = _StubLLM(_payload(direction="sideways"))   # invalid → degraded
    bad = _form(a)
    assert bad.degraded
    assert a.memory.conviction == good.conviction and a.memory.direction == good.direction


# ── back-compatibility ──────────────────────────────────────────────────────
def test_pre_existing_view_without_missing_inputs_still_loads():
    """Every run file on disk predates the field; a required field would break them all."""
    v = DriverView.model_validate({
        "driver": "inflation", "asof": ASOF, "direction": "up",
        "conviction": 0.5, "horizon_days": 30,
    })
    assert v.missing_inputs == []


# ── conviction response ─────────────────────────────────────────────────────
def _rec(direction: str, conviction: float, level: float, **over) -> dict:
    view = {"driver": "inflation", "asof": str(ASOF.date()), "direction": direction,
            "conviction": conviction, "horizon_days": 30, "level": level}
    view.update(over)
    return {"view": view}


def test_conviction_response_separates_hits_from_misses(tmp_path):
    # up@0.8 → level rises (right), next call up@0.9  → d_conv +0.1 after right
    # up@0.9 → level falls (wrong), next call up@0.4  → d_conv -0.5 after wrong
    path = tmp_path / "run.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in [
        _rec("up", 0.8, 2.0),
        _rec("up", 0.9, 2.5),
        _rec("up", 0.4, 2.1),
    ]) + "\n")

    out = conviction_response(str(path))
    assert out["n_transitions"] == 2 and out["n_right"] == 1 and out["n_wrong"] == 1
    assert out["d_conv_after_right"] == 0.1
    assert out["d_conv_after_wrong"] == -0.5


def test_conviction_response_skips_carried_and_degraded(tmp_path):
    path = tmp_path / "run.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in [
        _rec("up", 0.8, 2.0),
        _rec("up", 0.8, 2.5, carried=True),
        _rec("up", 0.1, 2.4, degraded=True),
    ]) + "\n")
    assert conviction_response(str(path))["n_transitions"] == 0
