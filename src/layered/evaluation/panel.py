"""Replay a feature spec across history into a signal matrix.

Runs ``FeatureEngine`` at every release date through the ``AsOf`` gate and flattens
each ``FeatureSet`` into one row, giving a ``(release date × feature)`` frame where
every column is a candidate signal.

This is the free, pre-LLM check. Before spending anything on the analyst it settles
a question that has to come first: **is this driver predictable at all from
measurements available at the time?** If nothing in the feature block carries any
information coefficient against the next print, a weak result from the model is a
fact about the problem rather than about the model — worth knowing before
interpreting a single report.

One distinction keeps this honest. Measuring a feature's IC to understand the
problem is diagnosis. Feeding that IC back to the analyst, or picking features
because they scored well here, would convert a measurement into a fitted signal and
reintroduce exactly the anchoring the redesign removed. **This module informs the
researcher; it must never inform the prompt.**
"""
from __future__ import annotations

import pandas as pd

from src.layered.features import FeatureEngine
from src.layered.timeline import AsOf


def release_dates(macro: dict[str, pd.Series], series_id: str,
                  start: str | None = None, end: str | None = None,
                  freq: str | None = None) -> pd.DatetimeIndex:
    """The clock a driver actually moves on — its target series' release dates.

    Series are release-dated on load, so these are the moments the number became
    known, not the months they describe.

    ``freq`` resamples the clock for **market drivers**. A Treasury or breakeven
    series updates every business day, so its raw index would grade a next-day move
    — noise — and daily spacing would break the non-overlapping-observation basis of
    the t-statistic. Passing ``freq="ME"`` samples the last value known by each
    month end instead, so the driver is graded on a ~monthly, non-overlapping clock
    just like a monthly release. Monthly-release drivers pass ``freq=None``.
    """
    s = macro.get(series_id)
    if s is None or s.empty:
        raise ValueError(f"series {series_id!r} is not in the macro bundle")
    s = s.dropna().sort_index()
    if freq:
        s = s.resample(freq).last().dropna()
    idx = s.index
    if start is not None:
        idx = idx[idx >= pd.Timestamp(start)]
    if end is not None:
        idx = idx[idx <= pd.Timestamp(end)]
    return pd.DatetimeIndex(idx)


class FeaturePanel:
    """Computes one driver's features at every date on its release clock."""

    def __init__(self, engine: FeatureEngine):
        self.engine = engine

    @property
    def clock_series(self) -> str:
        """Default clock — the first declared input, i.e. the driver's own target."""
        inputs = self.engine.inputs
        if not inputs:
            raise ValueError(f"{self.engine.spec.driver}: spec declares no raw inputs")
        return inputs[0]

    def build(self, macro: dict[str, pd.Series], dates: pd.DatetimeIndex) -> pd.DataFrame:
        """``(date × feature)``. A series feature contributes its latest value.

        Every row goes through ``AsOf``, so a feature at date *t* is computed from
        data available at *t* and nothing later.
        """
        empty = pd.DataFrame()
        rows: dict[pd.Timestamp, dict[str, float]] = {}
        for asof in dates:
            fs = self.engine.compute(AsOf(asof=asof, macro=macro, prices=empty))
            row = {f.name: (f.values[-1] if f.values else float("nan")) for f in fs.series}
            row.update({f.name: f.value for f in fs.scalars})
            rows[asof] = row
        return pd.DataFrame.from_dict(rows, orient="index").sort_index()

    def level(self, panel: pd.DataFrame) -> pd.Series:
        """The driver's level column — what the IC evaluator grades against."""
        name = self.engine.spec.level_feature
        if name is None:
            raise ValueError(f"{self.engine.spec.driver}: spec has no level_feature")
        if name not in panel.columns:
            raise ValueError(
                f"level_feature {name!r} is missing from the panel — not enough "
                f"history at these dates? (columns: {list(panel.columns)})"
            )
        return panel[name]
