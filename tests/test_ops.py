"""The closed feature vocabulary — a spec cannot express an operation it shouldn't."""
from __future__ import annotations

import pandas as pd
import pytest

from src.layered.features import ops

S = pd.Series([1.0, 2.0, 3.0, 4.0], index=pd.date_range("2020-01-01", periods=4))


def test_unknown_op_rejected():
    with pytest.raises(ValueError):
        ops.apply("inflation_signal", [S], {})


def test_wrong_arity_rejected():
    with pytest.raises(ValueError):
        ops.apply("spread", [S], {})          # spread needs two series


def test_unknown_param_rejected():
    with pytest.raises(ValueError):
        ops.apply("diff", [S], {"periods": 1})  # diff takes `window`, not `periods`


def test_lag_refuses_to_look_ahead():
    with pytest.raises(ValueError):
        ops.apply("lag", [S], {"periods": -1})
