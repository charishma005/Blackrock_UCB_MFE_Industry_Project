"""The vendored international series load cleanly and stay point-in-time.

Mirror of test_equity_local for the INTL_ family: loader hygiene, the
no-lookahead slice, and the six international personas' isolation + guardrail
coverage (they load via equity_local.load_any_bundle, which dispatches INTL_
ids to intl_local).
"""
from __future__ import annotations

import re

import pandas as pd
import yaml

from src.data.equity_local import load_any_bundle
from src.data.intl_local import available, load_series
from src.layered.analysts.llm_analyst import PERSONA_DIR, LLMAnalyst
from src.layered.features import FeatureEngine, from_persona
from src.layered.timeline import AsOf

INTL_DRIVERS = ["ea_rates", "uk_rates", "jp_rates",
                "ea_equity", "uk_equity", "jp_equity"]
_DATE = re.compile(r"\b(19|20)\d{2}\b")
ASOF = pd.Timestamp("2023-06-02")


def test_expected_series_are_present():
    got = set(available())
    assert {"INTL_DE10Y", "INTL_UK10Y", "INTL_JP10Y", "INTL_US10Y",
            "INTL_SXXGV", "INTL_UKX", "INTL_MSCIJP",
            "INTL_MSCIJP_RVOL13"} <= got


def test_series_are_friday_stamped_monotone_and_clean():
    for sid in ("INTL_DE10Y", "INTL_UKX_RVOL13"):
        s = load_series(sid)
        assert not s.empty
        assert s.index.is_monotonic_increasing
        assert not s.isna().any()
        # Every observation is a decision Friday (dayofweek 4).
        assert set(s.index.dayofweek) == {4}


def test_asof_slice_never_looks_ahead():
    full = load_series("INTL_DE10Y")
    sliced = load_series("INTL_DE10Y", end="2020-01-03")
    asof = pd.Timestamp("2020-01-03")
    assert sliced.equals(full.loc[:asof])
    assert sliced.index.max() <= asof


def test_every_intl_persona_reads_only_declared_inputs():
    for driver in INTL_DRIVERS:
        persona = yaml.safe_load((PERSONA_DIR / f"{driver}.yaml").read_text())
        engine = FeatureEngine(from_persona(driver, persona))
        declared = set(engine.inputs)
        assert declared and all(s.startswith("INTL_") for s in declared)
        macro = load_any_bundle(list(engine.inputs))
        fs = engine.compute(AsOf(asof=ASOF, macro=macro, prices=pd.DataFrame()))
        assert set(fs.sources_read) <= declared
        assert fs.level is not None, f"{driver}: level_feature produced no value"


def test_no_absolute_date_leaks_into_intl_prompts():
    for driver in INTL_DRIVERS:
        analyst = LLMAnalyst.from_persona(driver, llm=None, text_selector=None)
        macro = load_any_bundle(list(analyst.inputs))
        features, text = analyst.build_inputs(
            AsOf(asof=ASOF, macro=macro, prices=pd.DataFrame()))
        prompt = analyst._user_prompt(features, text)
        assert not _DATE.search(prompt), f"{driver}: a year leaked into the prompt"
