"""Run the meeting weekly and separate the two kinds of attribution.

One ``run_meeting`` is a single sitting; this loop stands it up as a schedule and
produces the two distinct records the thesis insists be kept apart:

    * research attribution — were the analysts' driver calls right?  (ResearchScorer)
    * arbitrage/fund P&L   — did the netted book actually make money?  (metrics)

Time integrity is preserved by handing each meeting an ``AsOf`` over the FULL
macro/price history: the AsOf gate slices every read to ``<= asof``, so no
meeting sees the future even though the loop holds all the data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.backtest.metrics import summary
from src.layered.meeting import Fund, MeetingResult
from src.layered.scoring import ResearchScorer
from src.layered.timeline import AsOf


@dataclass
class LayeredBacktestResult:
    values: pd.Series                                  # fund equity curve
    book_over_time: pd.DataFrame                        # (date x instrument) daily weights
    meetings: dict[pd.Timestamp, MeetingResult] = field(default_factory=dict)
    research_scorecard: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics: dict = field(default_factory=dict)


def run_weekly(
    fund: Fund,
    macro: dict[str, pd.Series],
    prices: pd.DataFrame,
    start: str,
    end: str,
    freq: str = "W-FRI",
    initial_cash: float = 100_000.0,
) -> LayeredBacktestResult:
    rets = prices.pct_change().dropna(how="all")

    # snap rebalance dates to real trading days at or before each target
    targets = pd.date_range(start, end, freq=freq)
    snapped = sorted({prices.index[prices.index <= d][-1] for d in targets
                      if len(prices.index[prices.index <= d])})
    rebalance_dates = pd.DatetimeIndex(snapped)

    scorer = ResearchScorer()
    books: dict[pd.Timestamp, pd.Series] = {}
    meetings: dict[pd.Timestamp, MeetingResult] = {}
    # per-strategy paper returns accumulated so the fund's diversification term
    # has something to chew on once several strategies are running.
    strat_paper: dict[str, pd.Series] = {}

    for asof in rebalance_dates:
        world = AsOf(asof=asof, macro=macro, prices=prices)  # AsOf slices to <= asof
        result = fund.run_meeting(world, strat_returns=strat_paper or None)
        scorer.record(result.views)
        books[asof] = result.book
        meetings[asof] = result

    # daily weight matrix, forward-filled between meetings, applied to NEXT day's
    # return (shift(1) — no lookahead).
    book_df = pd.DataFrame(books).T.reindex(prices.index).ffill().fillna(0.0)
    common = book_df.columns.intersection(rets.columns)
    lagged = book_df[common].shift(1)
    port_rets = (lagged * rets[common]).sum(axis=1).fillna(0.0)
    values = (1.0 + port_rets).cumprod() * initial_cash

    return LayeredBacktestResult(
        values=values,
        book_over_time=book_df,
        meetings=meetings,
        research_scorecard=scorer.scorecard(),
        metrics=summary(values, book_df),
    )
