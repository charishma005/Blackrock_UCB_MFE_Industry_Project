"""The time-integrity guarantee: nothing an analyst reads postdates ``asof``."""
from __future__ import annotations

import pandas as pd
import yaml

from src.data.fred_local import load_bundle
from src.layered.analysts.llm_analyst import PERSONA_DIR
from src.layered.features import FeatureEngine, from_persona
from src.layered.timeline import AsOf


def test_asof_slices_strictly_to_the_past():
    idx = pd.date_range("2020-01-31", "2020-12-31", freq="ME")
    s = pd.Series(range(len(idx)), index=idx, dtype=float)
    w = AsOf(asof=pd.Timestamp("2020-06-30"), macro={"X": s}, prices=pd.DataFrame())
    got = w.series("X")
    assert len(got) == 6
    assert (got.index <= pd.Timestamp("2020-06-30")).all()


def test_feature_engine_ignores_the_future():
    """Computing at t against full history must equal computing against history
    truncated at t — i.e. data after t changes nothing, so nothing leaks."""
    persona = yaml.safe_load((PERSONA_DIR / "inflation.yaml").read_text())
    engine = FeatureEngine(from_persona("inflation", persona))
    macro = load_bundle(list(engine.inputs))
    asof = pd.Timestamp("2018-06-30")

    full = engine.compute(AsOf(asof=asof, macro=macro, prices=pd.DataFrame()))
    truncated = {k: v.loc[:asof] for k, v in macro.items()}
    trunc = engine.compute(AsOf(asof=asof, macro=truncated, prices=pd.DataFrame()))

    assert full.level == trunc.level
    assert {f.name: f.value for f in full.scalars} == {f.name: f.value for f in trunc.scalars}
