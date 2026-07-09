"""CLI entry point for the headline experiment: equal-weighted vs
performance-weighted agent ensemble vs naive buy-and-hold benchmark.

Run:  python3 -m src.run_backtest --start 2024-01-01 --end 2024-12-31

Cost note: with weekly rebalancing over a year (~52 dates), Ray Dalio alone
accounts for ~52 * 7 = 364 LLM calls (he recomputes every rebalance since his
macro inputs evolve). Buffett and Damodaran are computed ONCE regardless of
date range (see backtest/engine.py docstring), so they add a fixed ~6 calls
total. On Haiku this is cheap; budget accordingly if you switch to Sonnet/Opus
for a real headline run.
"""
from __future__ import annotations

import argparse
import os

from src.backtest.engine import BacktestConfig, three_way_comparison, run_backtest

import pandas as pd

# Don't let pandas wrap the metrics table when output is redirected to a file
# (no terminal width -> it otherwise splits columns into a "\ ...continued" block).
pd.set_option("display.width", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


def print_diagnostics(cfg: BacktestConfig, llm_client) -> None:
    """Full transparency dump: agent weight evolution, fired agents, and every
    signal's actual LLM reasoning text — not just the numeric confidence."""
    result = run_backtest(cfg, llm_client=llm_client)
    point_in_time = cfg.equity_data_source == "financialdatasets"

    print("=" * 70)
    print("AGENT WEIGHT EVOLUTION (performance-weighted run)")
    print("=" * 70)
    print(result.agent_weights_history.round(3))
    print(f"\nFired agents: {result.fired_agents or 'none'}")

    print("\n" + "=" * 70)
    if point_in_time:
        print("EQUITY SIGNALS — Buffett & Damodaran, point-in-time, first and last rebalance")
        print("(recomputed only when the underlying quarterly filing changed)")
    else:
        print("STATIC EQUITY SIGNALS — Buffett & Damodaran (computed ONCE from today's "
              "fundamentals — LOOK-AHEAD BIASED, see engine docstring)")
    print("=" * 70)
    dates = sorted(result.rebalance_signals.keys())
    equity_snapshots = (
        [("FIRST", dates[0]), ("LAST", dates[-1])] if point_in_time and dates
        else [("(single computation)", dates[0])] if dates else []
    )
    for label, d in equity_snapshots:
        print(f"\n--- {label}: {d.date()} ---")
        for agent in ("warren_buffett", "aswath_damodaran"):
            sigs = result.static_signals.get(agent) or result.rebalance_signals[d].get(agent, {})
            print(f"  [{agent}]")
            for sym, s in sigs.items():
                print(f"    {sym}: {s['signal'].upper()} ({s['confidence']:.0f}%)")
                print(f"      reasoning: {s['reasoning']}")

    print("\n" + "=" * 70)
    print("RAY DALIO SIGNALS — first and last rebalance date (macro evolves over time)")
    print("=" * 70)
    if dates:
        for label, d in [("FIRST", dates[0]), ("LAST", dates[-1])]:
            print(f"\n--- {label} rebalance: {d.date()} ---")
            dalio_sigs = result.rebalance_signals[d].get("ray_dalio", {})
            for sym, s in dalio_sigs.items():
                print(f"  {sym}: {s['signal'].upper()} ({s['confidence']:.0f}%)")
                print(f"    reasoning: {s['reasoning']}")

    print("\n" + "=" * 70)
    print("FINAL METRICS")
    print("=" * 70)
    print(result.metrics)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--rebalance-freq", default="W-FRI", help="pandas offset alias, e.g. W-FRI, ME, 2W-FRI")
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--no-risk-manager", action="store_true",
                     help="Disable the vol-targeting/correlation risk layer (on by default).")
    ap.add_argument("--pm-mode", default="mechanical", choices=["mechanical", "llm"],
                     help="'mechanical' = risk-adjusted blend passes through; "
                          "'llm' = LLM portfolio manager synthesizes final weights (costs extra calls).")
    ap.add_argument("--equity-data-source", default="yfinance",
                     choices=["yfinance", "financialdatasets", "wrds"],
                     help="'yfinance' (default, free) uses TODAY's fundamentals for the whole backtest — "
                          "look-ahead biased, do not cite Sharpe from this mode. "
                          "'financialdatasets' uses point-in-time data via report_period_lte "
                          "(needs FINANCIAL_DATASETS_API_KEY, ~$20 one-time credits). "
                          "'wrds' uses Compustat point-in-time via the rdq field (needs a WRDS account "
                          "with Compustat access, e.g. a university subscription) — the most rigorous option.")
    ap.add_argument("--cache-dir", default=None,
                     help="Directory for the on-disk LLM response cache. Set it and reruns of the "
                          "same window/model become free and deterministic (temperature is pinned to 0). "
                          "The equal- and performance-weighted runs query identical signals, so caching "
                          "roughly halves the cost of the 3-way comparison. Delete the dir to force a re-query.")
    ap.add_argument("--verbose", action="store_true",
                     help="Print full agent weight evolution, fired agents, and LLM reasoning text "
                          "(single performance-weighted run, not the 3-way comparison table).")
    args = ap.parse_args()

    llm_client = None
    if os.environ.get("ANTHROPIC_API_KEY"):
        from src.llm.anthropic_client import AnthropicClient
        llm_client = AnthropicClient(model=args.model, cache_dir=args.cache_dir)
    else:
        print("[warn] ANTHROPIC_API_KEY not set — all signals will be neutral/0%, "
              "so this run only validates plumbing, not real results.\n")

    if args.equity_data_source == "financialdatasets" and not os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        print("[warn] --equity-data-source financialdatasets but FINANCIAL_DATASETS_API_KEY is not set — "
              "requests will 401. Get a key (paid tier, ~$20 one-time credits) at financialdatasets.ai.\n")
    if args.equity_data_source == "wrds" and not os.environ.get("WRDS_USERNAME"):
        print("[warn] --equity-data-source wrds but WRDS_USERNAME is not set — the first WRDS query "
              "will prompt interactively for credentials (or read ~/.pgpass). Set WRDS_USERNAME to "
              "skip the username prompt.\n")

    cfg = BacktestConfig(
        start=args.start, end=args.end, rebalance_freq=args.rebalance_freq,
        use_risk_manager=not args.no_risk_manager, pm_mode=args.pm_mode,
        equity_data_source=args.equity_data_source,
    )
    print(f"Running: {args.start} to {args.end}, rebalance={args.rebalance_freq}, "
          f"model={args.model}, risk_manager={not args.no_risk_manager}, pm_mode={args.pm_mode}, "
          f"equity_data_source={args.equity_data_source}\n")

    if args.verbose:
        print_diagnostics(cfg, llm_client)
    else:
        comparison = three_way_comparison(cfg, llm_client=llm_client)
        print(comparison.round(4))


if __name__ == "__main__":
    main()