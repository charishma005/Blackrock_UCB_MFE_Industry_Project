"""The prompt guardrails the redesign exists to hold.

The old suite passed with all three defects present: a direction in the prompt, a
date in the prompt, and identical bytes across analysts. These check the opposite —
cheaply, with no model call.
"""
from __future__ import annotations

import re

import pandas as pd

from src.data.fred_local import load_bundle
from src.layered.analysts.llm_analyst import LLMAnalyst
from src.layered.text.selector import scrub_dates
from src.layered.timeline import AsOf

# The drivers whose inputs are in the default vendored data (no fetch needed).
OFFLINE_DRIVERS = ["inflation", "labor_tightness", "curve_slope", "term_premium"]
ASOF = pd.Timestamp("2023-06-01")
_DATE = re.compile(r"\b(19|20)\d{2}\b")   # any absolute year


def _user_prompt(driver: str) -> str:
    analyst = LLMAnalyst.from_persona(driver, llm=None, text_selector=None)
    macro = load_bundle(list(analyst.inputs))
    features, text = analyst.build_inputs(AsOf(asof=ASOF, macro=macro, prices=pd.DataFrame()))
    return analyst._user_prompt(features, text)


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
