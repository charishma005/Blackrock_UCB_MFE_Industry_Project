"""The vendored equity series load cleanly and stay point-in-time.

Also folds in the equity personas' isolation + guardrail coverage (the FRED
suites in test_isolation / test_prompt_guardrails are scoped to FRED drivers and
load via fred_local, which can't read EQ_* — so equity coverage lives here and
loads via equity_local.load_any_bundle).
"""
from __future__ import annotations

import re

import pandas as pd
import yaml

from src.data.equity_local import available, load_any_bundle, load_series
from src.layered.analysts.llm_analyst import PERSONA_DIR, LLMAnalyst
from src.layered.features import FeatureEngine, from_persona
from src.layered.timeline import AsOf

EQUITY_DRIVERS = ["sector_breadth", "vol_regime", "positioning", "risk_appetite"]
_DATE = re.compile(r"\b(19|20)\d{2}\b")
ASOF = pd.Timestamp("2023-06-02")


def test_expected_series_are_present():
    got = set(available())
    assert {"EQ_VIX", "EQ_BREADTH_PCT", "EQ_AMGR_Z", "EQ_SLOPE_BP"} <= got


def test_series_are_friday_stamped_monotone_and_clean():
    s = load_series("EQ_VIX")
    assert not s.empty
    assert s.index.is_monotonic_increasing
    assert not s.isna().any()
    # Every observation is a decision Friday (dayofweek 4).
    assert set(s.index.dayofweek) == {4}


def test_asof_slice_never_looks_ahead():
    full = load_series("EQ_VIX")
    sliced = load_series("EQ_VIX", end="2020-01-03")
    asof = pd.Timestamp("2020-01-03")
    assert sliced.equals(full.loc[:asof])
    assert sliced.index.max() <= asof


def test_every_equity_persona_reads_only_declared_inputs():
    for driver in EQUITY_DRIVERS:
        persona = yaml.safe_load((PERSONA_DIR / f"{driver}.yaml").read_text())
        engine = FeatureEngine(from_persona(driver, persona))
        declared = set(engine.inputs)
        assert declared and all(s.startswith("EQ_") for s in declared)
        macro = load_any_bundle(list(engine.inputs))
        fs = engine.compute(AsOf(asof=ASOF, macro=macro, prices=pd.DataFrame()))
        assert set(fs.sources_read) <= declared
        assert fs.level is not None, f"{driver}: level_feature produced no value"


def test_no_absolute_date_leaks_into_equity_prompts():
    for driver in EQUITY_DRIVERS:
        analyst = LLMAnalyst.from_persona(driver, llm=None, text_selector=None)
        macro = load_any_bundle(list(analyst.inputs))
        features, text = analyst.build_inputs(
            AsOf(asof=ASOF, macro=macro, prices=pd.DataFrame()))
        prompt = analyst._user_prompt(features, text)
        assert not _DATE.search(prompt), f"{driver}: a year leaked into the prompt"
