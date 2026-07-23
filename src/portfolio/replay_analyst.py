"""Replay the macro-llm r7 equity signals as DriverViews — no LLM, $0.

The four equity drivers (`sector_breadth`, `vol_regime`, `positioning`,
`risk_appetite`) were run for 752 weeks in macro-llm's round 7; their validated
**vector-arm** outputs are vendored at ``data/equity_replay/<driver>_vector.csv``
(columns ``date,pos,conviction,reasoning``). This module replays them
point-in-time so the PM pods receive 14 years of real driver signals without an
API key — the historical counterpart to a live ``LLMAnalyst``.

Point-in-time contract: ``view_asof(asof)`` uses the latest CSV row with
``date <= asof``. It never reads a future row, so a truncated CSV produces
identical views up to its last date (the no-lookahead property the tests check).

Direction semantics (flagged for team sign-off — see data/equity_replay/README):
``pos`` is the r7 analyst's desired **S&P 500 position** (a market call), mapped

    pos > +0.15 → up    pos < −0.15 → down    else flat

with the ±0.15 threshold frozen in the r7 prereg. A live persona instead emits a
**driver-direction** call (will the driver's own level rise?). Consumers
disambiguate via ``source`` (``replay:<driver>`` vs ``llm:<driver>``).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.layered.contracts import DriverView

# Frozen in the r7 preregistration (EXPERIMENTS.md); strict inequality, so a
# position of exactly +0.15 maps to flat.
POS_FLAT_THRESHOLD = 0.15

# .../src/portfolio/replay_analyst.py -> parents[2] == repo root
_REPLAY_DIR = Path(__file__).resolve().parents[2] / "data" / "equity_replay"

EQUITY_DRIVERS: tuple[str, ...] = (
    "sector_breadth", "vol_regime", "positioning", "risk_appetite",
)


def _direction(pos: float) -> str:
    if pos > POS_FLAT_THRESHOLD:
        return "up"
    if pos < -POS_FLAT_THRESHOLD:
        return "down"
    return "flat"


class CsvReplayAnalyst:
    """One driver's cached r7 signal, replayed as point-in-time DriverViews."""

    def __init__(self, driver: str, csv_path: str | Path,
                 horizon_days: int = 7, max_age_days: int = 14):
        self.driver = driver
        self.csv_path = Path(csv_path)
        self.horizon_days = horizon_days
        # A view older than this at ``asof`` is treated as stale → no view. Guards
        # the pre-2012 / post-2026-06 edges where the replay simply has no data.
        self.max_age_days = max_age_days
        df = pd.read_csv(self.csv_path, parse_dates=["date"])
        self._df = df.sort_values("date").reset_index(drop=True)

    def view_asof(self, asof: pd.Timestamp) -> DriverView | None:
        """The latest signal knowable at ``asof`` as a DriverView, or None if the
        replay has no (recent enough) row on/before ``asof``."""
        asof = pd.Timestamp(asof)
        past = self._df[self._df["date"] <= asof]
        if past.empty:
            return None
        row = past.iloc[-1]
        row_date = pd.Timestamp(row["date"])
        if (asof - row_date).days > self.max_age_days:
            return None
        pos = float(row["pos"])
        conviction = min(1.0, max(0.0, float(row["conviction"])))
        return DriverView(
            driver=self.driver,
            asof=asof,
            direction=_direction(pos),
            conviction=conviction,
            horizon_days=self.horizon_days,
            level=pos,                       # the replayed S&P position, for scoring
            reasoning=str(row.get("reasoning", "")),
            source=f"replay:{self.driver}",
            carried=row_date < asof,         # re-used from an earlier week
        )


def load_replay_analysts(drivers: tuple[str, ...] = EQUITY_DRIVERS,
                         replay_dir: Path | None = None
                         ) -> dict[str, CsvReplayAnalyst]:
    """Build one CsvReplayAnalyst per driver whose ``<driver>_vector.csv`` exists."""
    d = replay_dir or _REPLAY_DIR
    out: dict[str, CsvReplayAnalyst] = {}
    for driver in drivers:
        path = d / f"{driver}_vector.csv"
        if path.exists():
            out[driver] = CsvReplayAnalyst(driver, path)
    return out
