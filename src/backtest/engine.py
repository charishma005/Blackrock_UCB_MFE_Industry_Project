"""Backtest engine — the daily loop that turns one-off agent signals into a
real track record with rolling attribution and dynamic (or equal) weighting.

Two equity data modes, set via BacktestConfig.equity_data_source:

  "yfinance" (default, free) — no point-in-time history. Buffett/Damodaran's
  facts are IDENTICAL at every rebalance date, so their signal is computed
  ONCE and held fixed for the whole run — LOOK-AHEAD BIASED (uses today's
  fundamentals throughout the backtest period). Fine for testing plumbing;
  do not cite Sharpe ratios from this mode.

  "financialdatasets" (paid, ~$20 one-time credits) — genuine point-in-time
  data via report_period_lte. Equity signals are recomputed at each
  rebalance using ONLY financials reported by that date. To control cost,
  the LLM is only re-queried when the underlying filing's report_period
  actually changed since the last rebalance (equities file quarterly, so
  weekly rebalancing would otherwise waste ~12 of 13 LLM calls per quarter
  on an unchanged filing) — see `_get_equity_signals_pointintime` below.

Ray Dalio's macro/price inputs genuinely change week to week regardless of
mode, so he always recomputes at every rebalance.

Portfolio construction: at each rebalance date, each agent's signals are
encoded as signed exposures (see ensemble/attribution.encode_signal),
normalized to unit gross exposure per agent, then blended by that agent's
current weight (equal or performance-based) into one target weight vector.
Target weights are held constant until the next rebalance and applied to
next-day returns (shift(1) — no lookahead).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from src.agents.aswath_damodaran import AswathDamodaranAgent
from src.agents.ray_dalio import RayDalioAgent
from src.agents.warren_buffett import WarrenBuffettAgent
from src.backtest.metrics import RISK_FREE_ANNUAL, TRADING_DAYS, summary, turnover
from src.data.equities import get_equity_facts_bundle as get_equity_facts_pointintime
from src.data.equities_wrds import get_equity_facts_bundle as get_equity_facts_wrds
from src.data.equities_yfinance import get_equity_facts_bundle as get_equity_facts_yfinance
from src.data.markets import daily_returns, fetch_macro_bundle, fetch_prices
from src.ensemble.attribution import AttributionTracker, encode_signal
from src.ensemble.weights import AgentPolicy, WeightManager
from src.instruments import DEFAULT_UNIVERSE, AssetClass, Instrument
from src.portfolio.manager import PMConfig, PortfolioManager
from src.risk.manager import RiskConfig, RiskManager

WeightingMode = Literal["equal", "performance"]
EquityDataSource = Literal["yfinance", "financialdatasets", "wrds"]


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
    pm_mode: str = "mechanical"       # "mechanical" or "llm" portfolio manager
    equity_data_source: EquityDataSource = "yfinance"  # "yfinance" (free, look-ahead biased)
                                                        # or "financialdatasets" (paid, point-in-time)


@dataclass
class BacktestResult:
    values: pd.Series
    weights_over_time: pd.DataFrame          # (date x instrument) target weights, daily
    agent_weights_history: pd.DataFrame       # (rebalance_date x agent) weights
    scorecards_history: dict[pd.Timestamp, pd.DataFrame] = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    fired_agents: set[str] = field(default_factory=set)
    static_signals: dict[str, dict[str, dict]] = field(default_factory=dict)   # Buffett/Damodaran, full reasoning
    rebalance_signals: dict[pd.Timestamp, dict[str, dict]] = field(default_factory=dict)  # Dalio per date, full reasoning
    risk_diag_history: dict[pd.Timestamp, dict] = field(default_factory=dict)   # per-rebalance risk-layer diagnostics
    pm_reasoning_history: dict[pd.Timestamp, str] = field(default_factory=dict) # per-rebalance PM reasoning text


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
            expo = expo / gross  # each agent's own book is unit gross exposure
        blended += w * expo
    return blended


def run_backtest(config: BacktestConfig, llm_client=None) -> BacktestResult:
    universe = DEFAULT_UNIVERSE
    all_syms = [i.symbol for i in universe if i.tradeable]
    equity_syms = [i.symbol for i in universe if i.asset_class == AssetClass.EQUITY]
    fred_ids = [i.symbol for i in universe if i.asset_class == AssetClass.MACRO]

    # Price history for ALL tradeable instruments (yfinance covers equities,
    # ETF proxies, and futures fine — separate concern from fundamentals source).
    prices = fetch_prices(all_syms, config.start, config.end)
    rets = daily_returns(prices)

    macro_start = (pd.Timestamp(config.start) - pd.Timedelta(days=config.macro_lookback_days)).strftime("%Y-%m-%d")
    macro_full = fetch_macro_bundle(fred_ids, macro_start, config.end)

    buffett = WarrenBuffettAgent(llm_client=llm_client)
    damo = AswathDamodaranAgent(llm_client=llm_client)

    point_in_time = config.equity_data_source in ("financialdatasets", "wrds")
    pit_fetcher = {"financialdatasets": get_equity_facts_pointintime, "wrds": get_equity_facts_wrds}.get(config.equity_data_source)
    static_signals: dict[str, dict[str, dict]] = {}
    equity_signal_cache: dict[tuple[str, str | None, str], dict] = {}  # (ticker, report_period, agent) -> signal dict
    equity_fetch_cache: dict[tuple[str, tuple[int, int]], dict] = {}   # (ticker, (year, quarter)) -> raw data bundle

    if not point_in_time:
        # LOOK-AHEAD BIASED PATH: yfinance has no point-in-time history, so
        # every rebalance date would see IDENTICAL (today's) fundamentals.
        # Computed once and held fixed — do not cite Sharpe from this mode.
        equity_data = {sym: get_equity_facts_yfinance(sym) for sym in equity_syms}
        static_signals = {
            "warren_buffett": buffett.run(universe, equity_data),
            "aswath_damodaran": damo.run(universe, equity_data),
        }

    def get_equity_signals_pointintime(asof_str: str) -> dict[str, dict[str, dict]]:
        """Recompute Buffett/Damodaran using ONLY financials reported by
        `asof_str`. Two-level caching, since financialdatasets.ai charges
        per request regardless of whether the answer changed:
          1. DATA fetch cached by (ticker, calendar quarter) — filings are
             quarterly, so fetching weekly would burn ~4x the API credits
             needed for no new information.
          2. LLM call cached by (ticker, report_period, agent) — belt and
             suspenders in case the API's report_period changes mid-quarter
             for some ticker (irregular fiscal calendars).
        """
        asof_ts = pd.Timestamp(asof_str)
        quarter_key = (asof_ts.year, asof_ts.quarter)
        out: dict[str, dict[str, dict]] = {"warren_buffett": {}, "aswath_damodaran": {}}
        for sym in equity_syms:
            fetch_key = (sym, quarter_key)
            if fetch_key not in equity_fetch_cache:
                equity_fetch_cache[fetch_key] = pit_fetcher(sym, asof_str)
            bundle = equity_fetch_cache[fetch_key]
            report_period = bundle.get("latest_report_period")
            for agent_name, agent in (("warren_buffett", buffett), ("aswath_damodaran", damo)):
                cache_key = (sym, report_period, agent_name)
                if cache_key not in equity_signal_cache:
                    facts = agent.compute_facts(Instrument(sym, AssetClass.EQUITY), bundle)
                    equity_signal_cache[cache_key] = agent.judge(Instrument(sym, AssetClass.EQUITY), facts).model_dump()
                out[agent_name][sym] = equity_signal_cache[cache_key]
        return out

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
        policies={"ray_dalio": AgentPolicy(window=120, floor=0.05, fire_after=5)},
    )

    weights_at_rebalance: dict[pd.Timestamp, pd.Series] = {}
    agent_weight_rows: dict[pd.Timestamp, dict[str, float]] = {}
    scorecards_history: dict[pd.Timestamp, pd.DataFrame] = {}
    rebalance_signals_history: dict[pd.Timestamp, dict[str, dict]] = {}
    agents_seen: set[str] = {"warren_buffett", "aswath_damodaran"}

    risk_mgr = RiskManager(RiskConfig()) if config.use_risk_manager else None
    pm = PortfolioManager(PMConfig(mode=config.pm_mode), llm_client=llm_client)
    risk_diag_history: dict[pd.Timestamp, dict] = {}
    pm_reasoning_history: dict[pd.Timestamp, str] = {}

    for asof in rebalance_dates:
        # No-lookahead slices: only data up to `asof` is visible to Dalio.
        macro_asof = {sid: s.loc[:asof] for sid, s in macro_full.items()}
        prices_asof = prices.loc[:asof]

        dalio = RayDalioAgent(llm_client=llm_client, macro_data=macro_asof, prices=prices_asof)
        dalio_signals = dalio.run(universe, data_by_symbol={})
        agents_seen.add("ray_dalio")

        equity_signals_today = static_signals if not point_in_time else get_equity_signals_pointintime(asof.strftime("%Y-%m-%d"))
        day_signals = {**equity_signals_today, "ray_dalio": dalio_signals}
        rebalance_signals_history[asof] = day_signals
        for agent, sigs in day_signals.items():
            tracker.record(agent, asof, sigs)

        rets_so_far = rets.loc[:asof]
        # Score each agent on its OWN policy window (e.g. Dalio's 120d) rather
        # than one global 60d window — otherwise a macro agent's longer intended
        # evaluation horizon is silently ignored and it gets fired on short noise.
        eval_windows = {a: wm.policy(a).window for a in agents_seen}
        scorecard = tracker.scorecard(rets_so_far, windows=eval_windows)
        scorecards_history[asof] = scorecard

        if config.weighting == "performance":
            agent_weights = wm.update(scorecard)
        else:  # equal-weight, still respects hard-fire logic for a fair comparison
            active = [a for a in agents_seen if a not in wm.fired]
            agent_weights = {a: 1.0 / len(active) for a in active} if active else {}

        agent_weight_rows[asof] = agent_weights
        blended = _blend_target_weights(day_signals, agent_weights, all_syms)

        # ── risk layer: vol targeting + correlation limits (deterministic) ──
        risk_diag: dict = {}
        if risk_mgr is not None:
            blended, risk_diag = risk_mgr.apply(blended, rets_so_far)
            risk_diag_history[asof] = risk_diag

        # ── portfolio manager: mechanical pass-through or LLM synthesis ──
        regime = dalio.regime()
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
    # Uninvested cash earns the risk-free rate. Without this, a book that runs a
    # cash sleeve (net exposure < 1) earns 0% on that sleeve while metrics.py
    # STILL charges the full rf in the Sharpe's excess-return term — a phantom
    # drag that dominates the ratio at low realized vol. The cash weight is
    # 1 - net exposure (short proceeds add to cash, matching a margin account),
    # which makes the return series self-consistent with the rf charged downstream.
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
        static_signals=static_signals,
        rebalance_signals=rebalance_signals_history,
        risk_diag_history=risk_diag_history,
        pm_reasoning_history=pm_reasoning_history,
    )


def run_benchmark(config: BacktestConfig) -> BacktestResult:
    """Equal-weight buy-and-hold across all tradeable instruments — no agents,
    no rebalancing. The naive baseline the agent ensembles should beat."""
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
    """The headline experiment: equal-weighted agents vs performance-weighted
    agents vs a naive equal-weight buy-and-hold benchmark."""
    eq_cfg = BacktestConfig(**{**config.__dict__, "weighting": "equal"})
    pw_cfg = BacktestConfig(**{**config.__dict__, "weighting": "performance"})

    equal_result = run_backtest(eq_cfg, llm_client=llm_client)
    weighted_result = run_backtest(pw_cfg, llm_client=llm_client)
    benchmark_result = run_benchmark(config)

    rows = {
        "equal_weighted_agents": equal_result.metrics,
        "performance_weighted_agents": weighted_result.metrics,
        "benchmark_equal_weight_buy_hold": benchmark_result.metrics,
    }
    return pd.DataFrame(rows).T
