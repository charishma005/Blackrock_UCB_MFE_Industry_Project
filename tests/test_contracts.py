"""The seam contract — the shapes the rest of the fund merges against."""
from __future__ import annotations

import pandas as pd

from src.layered.contracts import DriverView, FeatureSet, ScalarFeature, SeriesFeature

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


def test_render_carries_no_absolute_date():
    fs = FeatureSet(
        driver="x", asof=pd.Timestamp("2023-02-14"),
        series=[SeriesFeature(name="path", values=[1.0, 2.0], unit="%")],
        level_feature="path",
    )
    assert "2023" not in fs.render()
