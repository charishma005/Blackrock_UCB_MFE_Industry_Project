"""ALFRED vintage release dates — the rigorous alternative to the fixed per-series lag.

Covers: the pure reduction from a full revision history to each observation's first
release (no network needed), that ``fred_local.load_series`` prefers a vendored
vintage file over the fixed lag when one exists, that it falls back to the fixed lag
unchanged for a series with no vintage file (the regression that matters most — every
series without vintage data must behave exactly as before), partial-coverage
fallback, and the monotonicity guard against a corrupted vintage file.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.data import fred_local, fred_vintage
from src.data.markets import PUBLICATION_LAG_DAYS, _first_release_dates_from_observations


# ── the pure reduction (markets._first_release_dates_from_observations) ─────────────
def test_first_release_takes_the_earliest_vintage_per_observation():
    """A value revised twice after first publication must not overwrite the release
    date with a later revision's realtime_start."""
    obs = [
        {"date": "2024-01-01", "value": "300.0", "realtime_start": "2024-02-14"},
        {"date": "2024-01-01", "value": "300.1", "realtime_start": "2024-03-14"},  # revision
        {"date": "2024-02-01", "value": "301.0", "realtime_start": "2024-03-13"},
    ]
    out = _first_release_dates_from_observations(obs)
    assert out[pd.Timestamp("2024-01-01")] == pd.Timestamp("2024-02-14")
    assert out[pd.Timestamp("2024-02-01")] == pd.Timestamp("2024-03-13")


def test_first_release_skips_the_missing_value_sentinel():
    obs = [{"date": "2024-01-01", "value": ".", "realtime_start": "2024-02-14"}]
    assert _first_release_dates_from_observations(obs).empty


# ── fred_local.load_series preferring vintage data ──────────────────────────────────
def _write_observation_csv(path, series_id, rows):
    df = pd.DataFrame(rows, columns=["observation_date", series_id])
    df.to_csv(path / f"{series_id}.csv", index=False)


def _write_vintage_csv(path, series_id, rows):
    df = pd.DataFrame(rows, columns=["observation_date", "first_release_date"])
    df.to_csv(path / f"{series_id}.csv", index=False)


def test_load_series_prefers_true_release_dates_when_vendored(tmp_path, monkeypatch):
    fred_dir = tmp_path / "fred"
    vintage_dir = tmp_path / "fred_vintage"
    fred_dir.mkdir()
    vintage_dir.mkdir()
    monkeypatch.setenv("FRED_CSV_DIR", str(fred_dir))
    monkeypatch.setenv("FRED_VINTAGE_CSV_DIR", str(vintage_dir))

    _write_observation_csv(fred_dir, "CPIAUCSL",
                           [("2024-01-01", 300.0), ("2024-02-01", 301.0)])
    # True releases: 20/17 days out — deliberately NOT the fixed 14-day lag, so a
    # pass-through bug (still using PUBLICATION_LAG_DAYS) would be caught.
    _write_vintage_csv(vintage_dir, "CPIAUCSL",
                       [("2024-01-01", "2024-01-21"), ("2024-02-01", "2024-02-18")])

    s = fred_local.load_series("CPIAUCSL")
    assert list(s.index) == [pd.Timestamp("2024-01-21"), pd.Timestamp("2024-02-18")]
    assert list(s.values) == [300.0, 301.0]


def test_load_series_falls_back_to_fixed_lag_with_no_vintage_file(tmp_path, monkeypatch):
    """The regression that matters most: a series with no vendored vintage data must
    behave exactly as it did before this feature existed."""
    fred_dir = tmp_path / "fred"
    vintage_dir = tmp_path / "fred_vintage"     # exists, but no CSV for this series
    fred_dir.mkdir()
    vintage_dir.mkdir()
    monkeypatch.setenv("FRED_CSV_DIR", str(fred_dir))
    monkeypatch.setenv("FRED_VINTAGE_CSV_DIR", str(vintage_dir))

    _write_observation_csv(fred_dir, "CPIAUCSL", [("2024-01-01", 300.0)])
    s = fred_local.load_series("CPIAUCSL")
    assert s.index[0] == pd.Timestamp("2024-01-01") + pd.Timedelta(
        days=PUBLICATION_LAG_DAYS["CPIAUCSL"])


def test_load_series_falls_back_per_row_for_partial_vintage_coverage(tmp_path, monkeypatch):
    """A vintage file that only covers part of history must not drop or leave
    observation-dated the rows it doesn't cover — each missing row uses the fixed lag
    on its own, the rest still use the true release date."""
    fred_dir = tmp_path / "fred"
    vintage_dir = tmp_path / "fred_vintage"
    fred_dir.mkdir()
    vintage_dir.mkdir()
    monkeypatch.setenv("FRED_CSV_DIR", str(fred_dir))
    monkeypatch.setenv("FRED_VINTAGE_CSV_DIR", str(vintage_dir))

    _write_observation_csv(fred_dir, "UNRATE",
                           [("2024-01-01", 3.7), ("2024-02-01", 3.8)])
    _write_vintage_csv(vintage_dir, "UNRATE", [("2024-01-01", "2024-01-05")])  # Feb missing

    s = fred_local.load_series("UNRATE")
    assert s.loc[pd.Timestamp("2024-01-05")] == 3.7
    assert s.index[1] == pd.Timestamp("2024-02-01") + pd.Timedelta(
        days=PUBLICATION_LAG_DAYS["UNRATE"])


def test_out_of_order_vintage_dates_raise_rather_than_silently_reorder(tmp_path, monkeypatch):
    """A later observation released before an earlier one is a corrupted vendored
    file, not a valid vintage history — every rolling/diff feature op depends on this
    index staying in reference-period order, so this must fail loudly, not resort."""
    fred_dir = tmp_path / "fred"
    vintage_dir = tmp_path / "fred_vintage"
    fred_dir.mkdir()
    vintage_dir.mkdir()
    monkeypatch.setenv("FRED_CSV_DIR", str(fred_dir))
    monkeypatch.setenv("FRED_VINTAGE_CSV_DIR", str(vintage_dir))

    _write_observation_csv(fred_dir, "WALCL",
                           [("2024-01-01", 7000.0), ("2024-02-01", 7100.0)])
    # February's release date is BEFORE January's — impossible for a real series.
    _write_vintage_csv(vintage_dir, "WALCL",
                       [("2024-01-01", "2024-03-01"), ("2024-02-01", "2024-01-05")])

    with pytest.raises(ValueError, match="not increasing"):
        fred_local.load_series("WALCL")


def test_fred_vintage_available_and_load_release_dates(tmp_path, monkeypatch):
    vintage_dir = tmp_path / "fred_vintage"
    vintage_dir.mkdir()
    monkeypatch.setenv("FRED_VINTAGE_CSV_DIR", str(vintage_dir))
    assert fred_vintage.available() == set()

    _write_vintage_csv(vintage_dir, "NFCI", [("2024-01-03", "2024-01-11")])
    assert fred_vintage.available() == {"NFCI"}
    dates = fred_vintage.load_release_dates("NFCI")
    assert dates[pd.Timestamp("2024-01-03")] == pd.Timestamp("2024-01-11")


def test_load_release_dates_raises_for_an_unvendored_series(tmp_path, monkeypatch):
    monkeypatch.setenv("FRED_VINTAGE_CSV_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        fred_vintage.load_release_dates("NOPE")
