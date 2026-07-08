"""Orchestrator — one rebalance cycle of the multi-asset, weighted-agent fund.

Flow per rebalance date:
  1. fetch data (prices, macro, equity fundamentals)
  2. run each agent -> signals per instrument
  3. record signals in AttributionTracker
  4. WeightManager.update(scorecard) -> per-agent weights (the firing mechanism)
  5. RiskManager -> per-instrument position limits
  6. PortfolioManager (LLM) -> final orders, given WEIGHTED signals + regime context

Run:  python -m src.main --start 2024-01-01 --end 2024-12-31
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from src.agents.aswath_damodaran import AswathDamodaranAgent
from src.agents.ray_dalio import RayDalioAgent
from src.agents.warren_buffett import WarrenBuffettAgent
from src.data.equities_yfinance import get_equity_facts_bundle  # free; financialdatasets.ai now requires a paid key
from src.data.markets import daily_returns, fetch_macro_bundle, fetch_prices
from src.ensemble.attribution import AttributionTracker
from src.ensemble.weights import AgentPolicy, WeightManager
from src.instruments import DEFAULT_UNIVERSE, AssetClass

# TODO(port): bring over warren_buffett + aswath_damodaran from the original repo
#   - wrap each in a BaseInvestorAgent subclass (Phase 1 code ports almost verbatim)
#   - keep MIT attribution in file headers
# TODO(build): risk/manager.py multi-asset version (vol + cross-asset correlation limits)
# TODO(build): portfolio/manager.py LLM call consuming weighted signals + Dalio regime()
# TODO(build): backtest/engine.py daily loop calling this cycle + marking P&L


def run_cycle(start: str, end: str, macro_lookback_days: int = 730, model: str = "claude-haiku-4-5-20251001") -> None:
    """macro_lookback_days: extra history fetched *before* `start` so YoY /
    trend calcs (CPI YoY needs 13mo, unemployment trend needs ~12mo) have
    enough data even when the trading window itself is short.
    """
    universe = DEFAULT_UNIVERSE
    market_syms = [i.symbol for i in universe if i.data_source == "yfinance"]
    fred_ids = [i.symbol for i in universe if i.data_source == "fred"]

    macro_start = (pd.Timestamp(start) - pd.Timedelta(days=macro_lookback_days)).strftime("%Y-%m-%d")

    prices = fetch_prices(market_syms, start, end)
    macro = fetch_macro_bundle(fred_ids, macro_start, end)  # wider window than `start`
    rets = daily_returns(prices)

    # ── equity fundamentals (yfinance, free; no point-in-time filtering — see
    #    data/equities_yfinance.py docstring re: look-ahead bias for backtests) ──
    equity_syms = [i.symbol for i in universe if i.asset_class == AssetClass.EQUITY]
    data_by_symbol: dict[str, dict] = {sym: get_equity_facts_bundle(sym) for sym in equity_syms}

    # ── LLM client (leave ANTHROPIC_API_KEY unset to run signals as neutral) ──
    llm_client = None
    if os.environ.get("ANTHROPIC_API_KEY"):
        from src.llm.anthropic_client import AnthropicClient
        llm_client = AnthropicClient(model=model)
    else:
        print("[warn] ANTHROPIC_API_KEY not set — agents will emit neutral/0% placeholder signals.\n")

    # ── agents ──
    agents = [
        RayDalioAgent(llm_client=llm_client, macro_data=macro, prices=prices),
        WarrenBuffettAgent(llm_client=llm_client),
        AswathDamodaranAgent(llm_client=llm_client),
    ]

    tracker = AttributionTracker()
    wm = WeightManager(
        temperature=0.75,
        policies={
            # macro agents: longer window, floor weight, harder to fire (regime-aware)
            "ray_dalio": AgentPolicy(window=120, floor=0.05, fire_after=5),
        },
    )

    asof = prices.index[-1]
    all_signals: dict[str, dict[str, dict]] = {}
    for agent in agents:
        sigs = agent.run(universe, data_by_symbol=data_by_symbol)
        all_signals[agent.name] = sigs
        tracker.record(agent.name, asof, sigs)

    scorecard = tracker.scorecard(rets)
    weights = wm.update(scorecard)

    print("Regime:", agents[0].regime().get("quadrant"))
    print("\nAgent scorecard:\n", scorecard.round(3))
    print("\nAgent weights:", {k: round(v, 3) for k, v in weights.items()})
    print("Fired agents:", wm.fired or "none")
    print("\nSignals:")
    for agent, sigs in all_signals.items():
        for sym, s in sigs.items():
            print(f"  {agent:12s} {sym:6s} {s['signal']:8s} {s['confidence']:.0f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument(
        "--model", default="claude-haiku-4-5-20251001",
        help="Anthropic model string. Cheap/fast default for testing; "
             "use claude-sonnet-5 or claude-opus-4-8 for higher-quality real runs.",
    )
    args = ap.parse_args()
    run_cycle(args.start, args.end, model=args.model)
