"""The closed vocabulary of measurement operations.

This module is the mechanism behind the layer's central rule: **features are
measurements, never signals.** The analyst is fed the metrics an economist would
put on their dashboard — levels, changes, moving averages, spreads — and every
act of judgment happens in the model that reads them.

Keeping the vocabulary closed is what makes that rule structural rather than a
convention someone has to police in review. There is no operation here that fits
a parameter, standardizes over a full sample, or scores a direction, so a feature
spec *cannot express* a forecast. The test each operation satisfies: computable at
time ``t`` from data available at ``t``, with no parameter chosen by looking at
outcomes.

Every operation maps one or two ``pd.Series`` to a ``pd.Series``, so a spec can
take either the trajectory (last N values) or the current reading (last value)
from the same definition.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def level(s: pd.Series) -> pd.Series:
    """The series itself — for drivers whose level is the measurement (e.g. NFCI)."""
    return s


def diff(s: pd.Series, window: int = 1) -> pd.Series:
    """Absolute change over ``window`` observations."""
    return s.diff(window)


def pct_change(s: pd.Series, window: int = 1) -> pd.Series:
    """Percent change over ``window`` observations."""
    return s.pct_change(window) * 100.0


def yoy(s: pd.Series, periods: int = 12) -> pd.Series:
    """Year-over-year percent change (``periods`` = observations per year)."""
    return s.pct_change(periods) * 100.0


def pct_change_annualized(s: pd.Series, window: int, periods_per_year: int = 12) -> pd.Series:
    """Change over ``window`` observations, expressed at an annual rate.

    The standard way to see whether recent momentum is diverging from the
    trailing year — a 3-month annualized rate turns before year-over-year does.
    """
    return ((s / s.shift(window)) ** (periods_per_year / window) - 1.0) * 100.0


def moving_average(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window).mean()


def spread(a: pd.Series, b: pd.Series) -> pd.Series:
    """``a − b``, aligned on the union of their indices and forward-filled.

    Forward-fill is needed when the two series publish on different clocks (a
    monthly print against a daily one); it only ever carries a value *forward*,
    so it cannot introduce look-ahead.
    """
    joined = pd.concat([a, b], axis=1, sort=False).sort_index().ffill()
    return joined.iloc[:, 0] - joined.iloc[:, 1]


def distance_from_reference(s: pd.Series, reference: float) -> pd.Series:
    """Distance from a fixed, externally-given reference (e.g. a 2% target).

    The reference is a stated policy constant, not an estimate — it is never
    fitted from the data.
    """
    return s - reference


def lag(s: pd.Series, periods: int = 1) -> pd.Series:
    """The value ``periods`` observations ago.

    Enables base-effect measurement: a year-over-year rate is a rolling window, so
    the observation about to *leave* that window is already known today and
    mechanically determines part of the next reading. Looking backwards, never
    forwards — ``periods`` is required to be positive.
    """
    if periods < 1:
        raise ValueError("lag periods must be >= 1 (a negative lag would look ahead)")
    return s.shift(periods)


def ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    """``a / b``, aligned and forward-filled. A zero denominator yields NaN."""
    joined = pd.concat([a, b], axis=1, sort=False).sort_index().ffill()
    return joined.iloc[:, 0] / joined.iloc[:, 1].replace(0.0, np.nan)


def rolling_min(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window).min()


def rolling_max(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window).max()


# name -> (function, number of series inputs, allowed parameter names)
REGISTRY: dict[str, tuple] = {
    "level":                  (level,                  1, set()),
    "diff":                   (diff,                   1, {"window"}),
    "pct_change":             (pct_change,             1, {"window"}),
    "yoy":                    (yoy,                    1, {"periods"}),
    "pct_change_annualized":  (pct_change_annualized,  1, {"window", "periods_per_year"}),
    "moving_average":         (moving_average,         1, {"window"}),
    "spread":                 (spread,                 2, set()),
    "ratio":                  (ratio,                  2, set()),
    "lag":                    (lag,                    1, {"periods"}),
    "distance_from_reference":(distance_from_reference,1, {"reference"}),
    "rolling_min":            (rolling_min,            1, {"window"}),
    "rolling_max":            (rolling_max,            1, {"window"}),
}


def apply(op: str, inputs: list[pd.Series], params: dict) -> pd.Series:
    """Look up and run one operation, validating arity and parameter names."""
    if op not in REGISTRY:
        raise ValueError(f"unknown op {op!r}. Allowed: {', '.join(sorted(REGISTRY))}")
    fn, arity, allowed = REGISTRY[op]
    if len(inputs) != arity:
        raise ValueError(f"op {op!r} takes {arity} series input(s), got {len(inputs)}")
    unknown = set(params) - allowed
    if unknown:
        raise ValueError(f"op {op!r} does not accept {sorted(unknown)}; allowed: {sorted(allowed)}")
    return fn(*inputs, **params)
