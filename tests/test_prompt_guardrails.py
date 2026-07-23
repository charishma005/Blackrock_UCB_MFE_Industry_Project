"""The prompt guardrails the redesign exists to hold.

The old suite passed with all three defects present: a direction in the prompt, a
date in the prompt, and identical bytes across analysts. These check the opposite —
cheaply, with no model call.
"""
from __future__ import annotations

import re

import pandas as pd

from src.data.fred_local import load_bundle
from src.layered.analysts.carry_forward import CarryForward
from src.layered.analysts.llm_analyst import LLMAnalyst
from src.layered.contracts import DriverView
from src.layered.text.selector import scrub_dates
from src.layered.timeline import AsOf

# The drivers whose inputs are in the default vendored data (no fetch needed).
OFFLINE_DRIVERS = ["inflation", "labor_tightness", "curve_slope", "term_premium"]
ASOF = pd.Timestamp("2023-06-01")
_DATE = re.compile(r"\b(19|20)\d{2}\b")   # any absolute year


def _evidence(driver: str, use_memory: bool = False):
    analyst = LLMAnalyst.from_persona(driver, llm=None, text_selector=None,
                                      use_memory=use_memory)
    macro = load_bundle(list(analyst.inputs))
    features, text = analyst.build_inputs(AsOf(asof=ASOF, macro=macro, prices=pd.DataFrame()))
    return analyst, features, text


def _user_prompt(driver: str) -> str:
    analyst, features, text = _evidence(driver)
    return analyst._user_prompt(features, text)


def _view(driver: str, direction: str = "up", conviction: float = 0.6) -> DriverView:
    return DriverView(driver=driver, asof=ASOF, direction=direction, conviction=conviction,
                      horizon_days=30, falsifier="core 3m annualized falls below 2% in 2024")


def test_no_answer_leaked_into_the_prompt():
    for driver in OFFLINE_DRIVERS:
        p = _user_prompt(driver).lower()
        assert "conviction" not in p
        assert "direction" not in p


def test_no_absolute_date_in_the_prompt():
    for driver in OFFLINE_DRIVERS:
        assert not _DATE.search(_user_prompt(driver)), f"{driver}: a year leaked into the prompt"


def test_prompts_are_pairwise_distinct():
    prompts = [_user_prompt(d) for d in OFFLINE_DRIVERS]
    assert len(set(prompts)) == len(prompts), "two analysts received identical bytes"


def test_scrub_dates_removes_calendar_tokens():
    scrubbed = scrub_dates("Released February 01, 2023 at 2:00 p.m. EST; see 2022 data.")
    assert not _DATE.search(scrubbed)


# ── the memory arm ──────────────────────────────────────────────────────────
# Replaying the previous view is only safe because it adds no data to the prompt.
# These pin the two properties that make that true and would fail silently otherwise.

def test_memory_block_carries_no_date():
    """The replayed view must not date the prompt — including via the falsifier, which
    the model wrote and which the fixture deliberately seeds with a year."""
    for driver in OFFLINE_DRIVERS:
        analyst, features, text = _evidence(driver, use_memory=True)
        prompt = analyst._user_prompt(features, text, _view(driver))
        assert "Your previous view" in prompt, f"{driver}: memory block missing"
        assert not _DATE.search(prompt), f"{driver}: a year leaked in via the memory block"


def test_memory_is_absent_on_the_first_release():
    """No placeholder before there is a view to replay — the first prompt of a run is
    byte-identical to the memory-off arm."""
    analyst, features, text = _evidence("inflation", use_memory=True)
    assert analyst.memory is None
    assert analyst._user_prompt(features, text, analyst.memory) == \
        analyst._user_prompt(features, text)


def test_memory_off_reproduces_the_control_exactly():
    """With the arm off the view is held but never rendered, so the control arm is
    unchanged rather than merely similar."""
    analyst, features, text = _evidence("inflation", use_memory=False)
    analyst._memory = _view("inflation")
    assert analyst.memory is None
    assert "Your previous view" not in analyst._user_prompt(features, text, analyst.memory)


def test_carry_forward_fingerprint_ignores_the_replayed_view():
    """The load-bearing one. If the memory reached the evidence hash it would differ at
    every release, the cache would never hit again, and the phantom revisions that
    CarryForward exists to prevent would come straight back."""
    analyst, features, text = _evidence("inflation", use_memory=True)
    runner = CarryForward(analyst)

    analyst._memory = _view("inflation", "up", 0.6)
    first = runner._evidence_key(features, text)
    analyst._memory = _view("inflation", "down", 0.1)
    assert runner._evidence_key(features, text) == first, \
        "the replayed view leaked into the evidence fingerprint"
