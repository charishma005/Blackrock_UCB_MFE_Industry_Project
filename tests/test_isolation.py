"""Input isolation — an analyst can only ever touch the series its spec declares."""
from __future__ import annotations

import pandas as pd
import yaml

from src.data.fred_local import load_bundle
from src.layered.analysts.llm_analyst import PERSONA_DIR
from src.layered.features import FeatureEngine, from_persona
from src.layered.timeline import AsOf

OFFLINE_DRIVERS = ["inflation", "labor_tightness", "curve_slope", "term_premium"]


def test_sources_read_are_a_subset_of_declared_inputs():
    for driver in OFFLINE_DRIVERS:
        persona = yaml.safe_load((PERSONA_DIR / f"{driver}.yaml").read_text())
        engine = FeatureEngine(from_persona(driver, persona))
        declared = set(engine.inputs)
        macro = load_bundle(list(engine.inputs))
        fs = engine.compute(AsOf(asof=pd.Timestamp("2023-06-01"), macro=macro, prices=pd.DataFrame()))
        assert set(fs.sources_read) <= declared, (
            f"{driver}: read {set(fs.sources_read) - declared} outside its declared inputs"
        )


def test_engine_inputs_match_the_spec_contract():
    persona = yaml.safe_load((PERSONA_DIR / "inflation.yaml").read_text())
    spec = from_persona("inflation", persona)
    assert tuple(FeatureEngine(spec).inputs) == spec.declared_inputs
