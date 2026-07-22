"""CLI entry point for the headline experiment: equal-weighted vs
performance-weighted PM-pod ensemble vs naive buy-and-hold benchmark.

Run:  python3 -m src.run_backtest --start 2024-01-01 --end 2024-12-31

The three PM pods (relative_value, equities_topdown, trend_follower) are
currently DUMMY placeholders that emit neutral signals, and the analyst feed is
a placeholder too — so a run with no ANTHROPIC_API_KEY validates the full
plumbing (analysts → pods → ensemble → risk → book) end-to-end without spending
anything. Swap in the real pods / live analyst feed later; this CLI does not
change.
"""
from __future__ import annotations

import argparse
import os

from src.backtest.engine import BacktestConfig, three_way_comparison, run_backtest

import pandas as pd

# Don't let pandas wrap the metrics table when output is redirected to a file.
pd.set_option("display.width", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


def print_diagnostics(cfg: BacktestConfig, llm_client) -> None:
    """Full transparency dump: pod weight evolution, fired pods, and every
    pod signal's reasoning text at the first and last rebalance."""
    result = run_backtest(cfg, llm_client=llm_client)

    print("=" * 70)
    print("POD WEIGHT EVOLUTION (performance-weighted run)")
    print("=" * 70)
    print(result.agent_weights_history.round(3))
    print(f"\nFired pods: {result.fired_agents or 'none'}")

    dates = sorted(result.rebalance_signals.keys())
    if dates:
        print("\n" + "=" * 70)
        print("POD SIGNALS — first and last rebalance date")
        print("=" * 70)
        for label, d in [("FIRST", dates[0]), ("LAST", dates[-1])]:
            print(f"\n--- {label} rebalance: {d.date()} ---")
            for pod, sigs in result.rebalance_signals[d].items():
                print(f"  [{pod}]")
                for sym, s in sigs.items():
                    print(f"    {sym}: {s['signal'].upper()} ({s['confidence']:.0f}%)")
                    if s.get("reasoning"):
                        print(f"      reasoning: {s['reasoning']}")

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
    ap.add_argument("--weighting", default="performance", choices=["equal", "performance"],
                     help="Pod ensemble weighting for the single --verbose run. 'performance' "
                          "(default) = dynamic weights from rolling attribution; 'equal' = fixed "
                          "1/N across active pods. Ignored by the 3-way comparison, which runs both.")
    ap.add_argument("--pm-mode", default="mechanical", choices=["mechanical", "llm"],
                     help="Final PM synthesis: 'mechanical' = risk-adjusted blend passes through; "
                          "'llm' = LLM portfolio manager synthesizes final weights (costs extra calls).")
    ap.add_argument("--cache-dir", default=None,
                     help="Directory for the on-disk LLM response cache. Set it and reruns of the "
                          "same window/model become free and deterministic (temperature pinned to 0).")
    ap.add_argument("--verbose", action="store_true",
                     help="Print full pod weight evolution, fired pods, and signal reasoning "
                          "(single performance-weighted run, not the 3-way comparison table).")
    args = ap.parse_args()

    llm_client = None
    if os.environ.get("ANTHROPIC_API_KEY"):
        from src.backtest.anthropic_client import AnthropicClient
        llm_client = AnthropicClient(model=args.model, cache_dir=args.cache_dir)
    else:
        print("[warn] ANTHROPIC_API_KEY not set — pods and analysts are dummy/neutral, "
              "so this run only validates plumbing, not real results.\n")

    cfg = BacktestConfig(
        start=args.start, end=args.end, rebalance_freq=args.rebalance_freq,
        weighting=args.weighting,
        use_risk_manager=not args.no_risk_manager, pm_mode=args.pm_mode,
    )
    print(f"Running: {args.start} to {args.end}, rebalance={args.rebalance_freq}, "
          f"model={args.model}, weighting={args.weighting}, "
          f"risk_manager={not args.no_risk_manager}, pm_mode={args.pm_mode}\n")

    if args.verbose:
        print_diagnostics(cfg, llm_client)
    else:
        comparison = three_way_comparison(cfg, llm_client=llm_client)
        print(comparison.round(4))


if __name__ == "__main__":
    main()
