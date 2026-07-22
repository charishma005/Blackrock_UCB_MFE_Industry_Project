"""Backtest engine — the daily loop that turns PM-pod signals into a real track
record with rolling attribution and dynamic (or equal) weighting.

Pipeline (see slides/architecture.html):

    analysts (DriverView) → PM pods → agent_signals → ensemble → risk → book

Each PM pod is treated by the ensemble as one "agent": the pod's per-instrument
calls are scored, weighted, and fired exactly the way individual agents used to
be. The three pods (relative_value, equities_topdown, trend_follower) live in
``src.portfolio.pods``; the analyst→pod adapter is ``src.portfolio.analyst_feed``.

Portfolio construction: at each rebalance date every pod's signals are encoded as
signed exposures (see ensemble/attribution.encode_signal), normalized to unit
gross exposure per pod, then blended by that pod's current weight (equal or
performance-based) into one target weight vector. Target weights are held constant
until the next rebalance and applied to next-day returns (shift(1) — no lookahead).

NOTE: the PM pods and the analyst feed are currently DUMMY placeholders that emit
neutral signals, so a run with no API key exercises the full plumbing end-to-end
without producing real views. Swap in the real pods / live analyst feed later;
nothing in this engine needs to change, because the agent_signals contract is
already what it consumes.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from src.backtest.metrics import RISK_FREE_ANNUAL, TRADING_DAYS, summary, turnover
from src.data.markets import daily_returns, fetch_macro_bundle, fetch_prices
from src.ensemble.attribution import AttributionTracker, encode_signal
from src.ensemble.weights import AgentPolicy, WeightManager
from src.instruments import DEFAULT_UNIVERSE, AssetClass
from src.portfolio.analyst_feed import analyst_views_asof
from src.portfolio.manager import PMConfig, PortfolioManager
from src.portfolio.pods import PMPods
from src.risk.manager import RiskConfig, RiskManager

WeightingMode = Literal["equal", "performance"]


@dataclass
class BacktestConfig:
    start: str
    end: str
    rebalance_freq: str = "W-FRI"     # weekly, Friday close
    initial_cash: float = 100_000.0
    macro_lookback_days: int = 730
    model: str = "claude-haiku-4-5-20251001"
    weighting: WeightingMode = "performance"
    weight_temperature: float = 0.75
    use_risk_manager: bool = True     # apply vol targeting + correlation limits
    pm_mode: str = "mechanical"       # final synthesis: "mechanical" or "llm"


@dataclass
class BacktestResult:
    values: pd.Series
    weights_over_time: pd.DataFrame          # (date x instrument) target weights, daily
    agent_weights_history: pd.DataFrame       # (rebalance_date x pod) weights
    scorecards_history: dict[pd.Timestamp, pd.DataFrame] = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    fired_agents: set[str] = field(default_factory=set)       # pods hard-fired by the ensemble
    static_signals: dict[str, dict[str, dict]] = field(default_factory=dict)   # reserved (unused: pods are recomputed per date)
    rebalance_signals: dict[pd.Timestamp, dict[str, dict]] = field(default_factory=dict)  # {date: {pod: {sym: signal}}}
    risk_diag_history: dict[pd.Timestamp, dict] = field(default_factory=dict)   # per-rebalance risk-layer diagnostics
    pm_reasoning_history: dict[pd.Timestamp, str] = field(default_factory=dict) # per-rebalance final-PM reasoning text
    attribution: object | None = None          # AttributionTracker, for per-pod paper P&L in diagnostics
    asset_returns: pd.DataFrame | None = None  # (date x symbol) daily returns used for attribution


def _blend_target_weights(
    agent_signals_by_agent: dict[str, dict[str, dict]],
    agent_weights: dict[str, float],
    instruments: list[str],
) -> pd.Series:
    """One rebalance date -> blended target weight per instrument."""
    blended = pd.Series(0.0, index=instruments)
    for agent, w in agent_weights.items():
        sigs = agent_signals_by_agent.get(agent, {})
        expo = pd.Series(
            {sym: encode_signal(s.get("signal", "neutral"), s.get("confidence", 0.0)) for sym, s in sigs.items()},
            index=instruments,
        ).fillna(0.0)
        gross = expo.abs().sum()
        if gross > 0:
            expo = expo / gross  # each pod's own book is unit gross exposure
        blended += w * expo
    return blended


def run_backtest(config: BacktestConfig, llm_client=None) -> BacktestResult:
    universe = DEFAULT_UNIVERSE
    all_syms = [i.symbol for i in universe if i.tradeable]
    fred_ids = [i.symbol for i in universe if i.asset_class == AssetClass.MACRO]

    # Price history for ALL tradeable instruments.
    prices = fetch_prices(all_syms, config.start, config.end)
    rets = daily_returns(prices)

    macro_start = (pd.Timestamp(config.start) - pd.Timedelta(days=config.macro_lookback_days)).strftime("%Y-%m-%d")
    macro_full = fetch_macro_bundle(fred_ids, macro_start, config.end)

    # The three PM pods. Each is one "agent" from the ensemble's point of view.
    pods = PMPods()

    rebalance_dates = pd.date_range(config.start, config.end, freq=config.rebalance_freq)
    # snap each target date to the nearest prior trading day actually in the price index
    trading_index = prices.index
    snapped = []
    for d in rebalance_dates:
        prior = trading_index[trading_index <= d]
        if len(prior) > 0:
            snapped.append(prior[-1])
    rebalance_dates = pd.DatetimeIndex(sorted(set(snapped)))

    tracker = AttributionTracker()
    wm = WeightManager(
        temperature=config.weight_temperature,
        # The trend follower is the macro/slow pod — give it a longer evaluation
        # window and a floor so it is not fired on short-window noise during a
        # quiet regime (the old ray_dalio policy, re-pointed at the pod).
        policies={"trend_follower": AgentPolicy(window=120, floor=0.05, fire_after=5)},
    )

    weights_at_rebalance: dict[pd.Timestamp, pd.Series] = {}
    agent_weight_rows: dict[pd.Timestamp, dict[str, float]] = {}
    scorecards_history: dict[pd.Timestamp, pd.DataFrame] = {}
    rebalance_signals_history: dict[pd.Timestamp, dict[str, dict]] = {}
    agents_seen: set[str] = set(pods.names)

    risk_mgr = RiskManager(RiskConfig()) if config.use_risk_manager else None
    pm = PortfolioManager(PMConfig(mode=config.pm_mode), llm_client=llm_client)
    risk_diag_history: dict[pd.Timestamp, dict] = {}
    pm_reasoning_history: dict[pd.Timestamp, str] = {}

    n_reb = len(rebalance_dates)
    if n_reb:
        print(f"[backtest:{config.weighting}] {n_reb} rebalance dates "
              f"{rebalance_dates[0].date()} → {rebalance_dates[-1].date()}",
              file=sys.stderr, flush=True)

    for i, asof in enumerate(rebalance_dates, start=1):
        print(f"[backtest:{config.weighting}] rebalance {i}/{n_reb}  {asof.date()}",
              file=sys.stderr, flush=True)
        # No-lookahead slice: only data up to `asof` is visible to the analysts.
        macro_asof = {sid: s.loc[:asof] for sid, s in macro_full.items()}

        # analysts → pods → agent_signals dict, keyed by pod name.
        analyst_views = analyst_views_asof(asof, macro_asof)
        day_signals = pods.run(analyst_views, universe)
        rebalance_signals_history[asof] = day_signals
        for agent, sigs in day_signals.items():
            tracker.record(agent, asof, sigs)

        rets_so_far = rets.loc[:asof]
        # Score each pod on its OWN policy window rather than one global window,
        # so a slow/macro pod's longer intended horizon is respected.
        eval_windows = {a: wm.policy(a).window for a in agents_seen}
        scorecard = tracker.scorecard(rets_so_far, windows=eval_windows)
        scorecards_history[asof] = scorecard

        # Always update fire/strike state from the scorecard, regardless of
        # blending mode, so equal-weight respects the same hard-fire logic.
        performance_weights = wm.update(scorecard)

        if config.weighting == "performance":
            agent_weights = performance_weights
        else:  # equal-weight among survivors of the SAME firing logic
            active = [a for a in agents_seen if a not in wm.fired]
            agent_weights = {a: 1.0 / len(active) for a in active} if active else {}

        agent_weight_rows[asof] = agent_weights
        blended = _blend_target_weights(day_signals, agent_weights, all_syms)

        # ── risk layer: vol targeting + correlation limits (deterministic) ──
        risk_diag: dict = {}
        if risk_mgr is not None:
            blended, risk_diag = risk_mgr.apply(blended, rets_so_far)
            risk_diag_history[asof] = risk_diag

        # ── final PM synthesis: mechanical pass-through or LLM synthesis ──
        # Regime is a placeholder now that the macro agent is gone; the real
        # regime read can come from a dedicated analyst (e.g. financial_conditions).
        regime: dict = {}
        final_weights, pm_reason = pm.decide(blended, day_signals, regime, risk_diag)
        pm_reasoning_history[asof] = pm_reason

        # LLM PM output is re-clipped by the risk layer so it can't breach limits
        if risk_mgr is not None and config.pm_mode == "llm":
            final_weights, _ = risk_mgr.apply(final_weights, rets_so_far)

        weights_at_rebalance[asof] = final_weights

    # Build daily weight matrix via forward-fill between rebalance dates
    weights_df = pd.DataFrame(weights_at_rebalance).T.reindex(prices.index).ffill().fillna(0.0)
    agent_weights_history = pd.DataFrame(agent_weight_rows).T.fillna(0.0)

    # Apply weights to NEXT day's return (shift(1) — no lookahead)
    common = weights_df.columns.intersection(rets.columns)
    lagged = weights_df[common].shift(1)
    asset_pnl = (lagged * rets[common]).sum(axis=1)
    # Uninvested cash earns the risk-free rate (keeps the return series
    # self-consistent with the rf charged in the Sharpe downstream).
    cash_weight = 1.0 - lagged.sum(axis=1)
    cash_pnl = cash_weight * (RISK_FREE_ANNUAL / TRADING_DAYS)
    port_rets = (asset_pnl + cash_pnl).fillna(0.0)
    values = (1 + port_rets).cumprod() * config.initial_cash

    return BacktestResult(
        values=values,
        weights_over_time=weights_df,
        agent_weights_history=agent_weights_history,
        scorecards_history=scorecards_history,
        metrics=summary(values, weights_df),
        fired_agents=wm.fired,
        static_signals={},
        rebalance_signals=rebalance_signals_history,
        risk_diag_history=risk_diag_history,
        pm_reasoning_history=pm_reasoning_history,
        attribution=tracker,
        asset_returns=rets,
    )


def run_benchmark(config: BacktestConfig) -> BacktestResult:
    """Equal-weight buy-and-hold across all tradeable instruments — no pods,
    no rebalancing. The naive baseline the pod ensemble should beat."""
    universe = DEFAULT_UNIVERSE
    all_syms = [i.symbol for i in universe if i.tradeable]
    prices = fetch_prices(all_syms, config.start, config.end)
    rets = daily_returns(prices)

    n = len(rets.columns)
    static_weight = pd.Series(1.0 / n, index=rets.columns)
    weights_df = pd.DataFrame([static_weight] * len(prices.index), index=prices.index)

    port_rets = rets.mean(axis=1).fillna(0.0)  # equal weight, static -> just the cross-sectional mean
    values = (1 + port_rets).cumprod() * config.initial_cash

    return BacktestResult(
        values=values,
        weights_over_time=weights_df,
        agent_weights_history=pd.DataFrame(),
        metrics=summary(values, weights_df),
    )


def three_way_comparison(config: BacktestConfig, llm_client=None) -> pd.DataFrame:
    """The headline experiment: equal-weighted pods vs performance-weighted
    pods vs a naive equal-weight buy-and-hold benchmark."""
    eq_cfg = BacktestConfig(**{**config.__dict__, "weighting": "equal"})
    pw_cfg = BacktestConfig(**{**config.__dict__, "weighting": "performance"})

    equal_result = run_backtest(eq_cfg, llm_client=llm_client)
    weighted_result = run_backtest(pw_cfg, llm_client=llm_client)
    benchmark_result = run_benchmark(config)

    rows = {
        "equal_weighted_pods": equal_result.metrics,
        "performance_weighted_pods": weighted_result.metrics,
        "benchmark_equal_weight_buy_hold": benchmark_result.metrics,
    }
    return pd.DataFrame(rows).T
