"""The seam contract — the shapes the rest of the fund merges against."""
from __future__ import annotations

import json

import pandas as pd

from src.layered.contracts import (
    ArbitratedView,
    DiscountedAnalyst,
    DriverView,
    FeatureSet,
    Risk,
    ScalarFeature,
    SeriesFeature,
    StrategyTrade,
)

_BASE = dict(driver="x", asof=pd.Timestamp("2020-01-01"), horizon_days=31)


def test_signed_conviction_folds_direction_in():
    assert DriverView(direction="up", conviction=0.6, **_BASE).signed_conviction == 0.6
    assert DriverView(direction="down", conviction=0.6, **_BASE).signed_conviction == -0.6
    assert DriverView(direction="flat", conviction=0.6, **_BASE).signed_conviction == 0.0


def test_featureset_level_resolves_from_level_feature():
    fs = FeatureSet(
        driver="x", asof=pd.Timestamp("2020-01-01"),
        series=[SeriesFeature(name="path", values=[1.0, 2.0, 3.0])],
        scalars=[ScalarFeature(name="now", value=9.0)],
        level_feature="path",
    )
    assert fs.level == 3.0                 # last value of the level series
    assert fs.names == {"path", "now"}


def test_arbitrated_view_roundtrips_through_json():
    """The PM run files are reloaded by ``evaluation/pm_runs.py``, and ``asof`` needs
    the same Timestamp coercion on the way back in that ``DriverView`` does — the
    contract sets ``arbitrary_types_allowed``, so pydantic checks the instance rather
    than parsing the ISO string."""
    av = ArbitratedView(asof=pd.Timestamp("2023-06-30"),
                        drivers={"inflation": 0.4, "curve_slope": -0.2},
                        disagreement=0.31, notes="n")
    dumped = av.model_dump(mode="json")
    back = ArbitratedView.model_validate({**dumped, "asof": pd.Timestamp(dumped["asof"])})
    assert back.drivers == av.drivers and back.disagreement == av.disagreement


def test_arbitrated_view_accepts_an_empty_panel():
    """The degraded shape: no drivers scored, so nothing is graded."""
    assert ArbitratedView(asof=pd.Timestamp("2023-06-30"), drivers={}).drivers == {}


def test_render_carries_no_absolute_date():
    fs = FeatureSet(
        driver="x", asof=pd.Timestamp("2023-02-14"),
        series=[SeriesFeature(name="path", values=[1.0, 2.0], unit="%")],
        level_feature="path",
    )
    assert "2023" not in fs.render()


def test_an_arbitrated_view_carrying_a_trade_survives_a_round_trip():
    """Dump-then-load, the replay path. `StrategyTrade.asof` is a pd.Timestamp under
    arbitrary_types_allowed, so without ArbitratedView's own coercion a saved view
    would write cleanly and refuse to load — a bug that only ever surfaces on replay."""
    av = ArbitratedView(
        asof=pd.Timestamp("2023-06-30"), drivers={"inflation": 0.4},
        leaned_on=["inflation"], falsifier="f", confidence=0.7,
        risks=[Risk(text="r", tag="curve")],
        discounted=[DiscountedAnalyst(driver="curve_slope", why="stale")],
        trade=StrategyTrade(strategy="curve", asof=pd.Timestamp("2023-06-30"),
                            legs={"DGS2": -0.5, "DGS10": 0.5}, conviction=0.6,
                            rationale="r"))
    dumped = json.loads(json.dumps(av.model_dump(mode="json")))
    back = ArbitratedView.model_validate({**dumped, "asof": pd.Timestamp(dumped["asof"])})
    assert back.trade.legs == {"DGS2": -0.5, "DGS10": 0.5}
    assert back.trade.asof == pd.Timestamp("2023-06-30")
    assert back.confidence == 0.7 and back.risks[0].tag == "curve"
    assert [d.driver for d in back.discounted] == ["curve_slope"]
