"""Run the layered agent fund — the analyst-PM meeting as a standing architecture.

Two things this prints:
  1. ONE meeting in full — every analyst's driver view, the PM's arbitration and
     the relative-value trade it expresses, the fund's allocation, and the final
     netted book. This is the worked example from the thesis made concrete.
  2. A weekly run over the window — the fund's P&L AND the research scorecard,
     shown side by side to make the point that research quality (were the driver
     calls right?) is attributed separately from arbitrage quality (did the book
     make money?).

Data:
  --source synthetic (default)  fabricated, internally-consistent series — runs
                                offline with no keys. --regime hawkish|dovish.
  --source fred                 real FRED macro + yfinance prices (needs
                                FRED_API_KEY and network).

LLM (optional): set ANTHROPIC_API_KEY to let each analyst's Phase-2 refine its
deterministic reading. Unset → pure deterministic Phase-1, still fully runnable.

Run:  python3 -m src.run_layered
      python3 -m src.run_layered --source fred --start 2022-01-01 --end 2024-12-31
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from src.layered.backtest import run_weekly
from src.layered.meeting import macro_rates_fund
from src.layered.pm.macro_rates import RATES_UNIVERSE
from src.layered.timeline import AsOf


def _load_data(args):
    if args.source == "fred":
        from src.data.markets import fetch_macro_bundle, fetch_prices
        from src.layered.analysts.macro_rates import MACRO_RATES_SERIES

        syms = [i.symbol for i in RATES_UNIVERSE]
        prices = fetch_prices(syms, args.start, args.end)
        # widen the macro window so YoY / trend calcs have runway before `start`
        macro_start = (pd.Timestamp(args.start) - pd.Timedelta(days=1095)).strftime("%Y-%m-%d")
        macro = fetch_macro_bundle(list(MACRO_RATES_SERIES), macro_start, args.end)
        return macro, prices
    from src.layered.synthetic import generate
    return generate(args.start, args.end, regime=args.regime)


def _print_meeting(result) -> None:
    print("=" * 74)
    print(f"ONE MEETING — as of {result.asof.date()}")
    print("=" * 74)

    print("\n[1] ANALYST LAYER — isolated single-driver views")
    for v in result.views:
        lvl = f"level={v.level}" if v.level is not None else ""
        print(f"  {v.driver:16s} {v.direction:4s}  conv={v.conviction:.2f}  {lvl}")
        print(f"        {v.reasoning}")

    print("\n[2] PM LAYER — arbitrate + express")
    for t in result.trades:
        print(f"  strategy: {t.strategy}   conviction={t.conviction:.2f}")
        print(f"    {t.rationale}")
        print(f"    trade legs (unit gross): {t.legs}")
        print(f"    risk: structure={t.risk['structure']}, net_duration={t.risk['net_duration']} "
              f"(isolates {t.risk['isolates']}, hedges {t.risk['hedges']}), "
              f"disagreement={t.risk['disagreement']}")

    print("\n[3] UNIFYING LAYER — net, size, feed back down")
    a = result.allocation
    print(f"  capital multipliers: {a.capital}")
    for k in ("diversification", "base_size", "pre_scale_vol", "post_scale_vol", "netted_gross", "netted_book"):
        if k in a.diagnostics:
            print(f"    {k}: {a.diagnostics[k]}")

    print("\n[4] FINAL FUND BOOK (netted, sized)")
    for sym, w in result.book.items():
        if abs(w) > 1e-9:
            print(f"    {sym:6s} {w:+.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "fred"])
    ap.add_argument("--regime", default="hawkish", choices=["hawkish", "dovish"],
                    help="synthetic source only: which textbook regime to fabricate")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--freq", default="W-FRI", help="rebalance frequency (pandas offset alias)")
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    args = ap.parse_args()

    llm_client = None
    if os.environ.get("ANTHROPIC_API_KEY"):
        from src.llm.anthropic_client import AnthropicClient
        llm_client = AnthropicClient(model=args.model)
        try:
            llm_client.validate()  # fail fast on a bad key/model
        except Exception as e:  # noqa: BLE001
            print(f"[error] LLM preflight failed — check ANTHROPIC_API_KEY and --model.\n"
                  f"        {type(e).__name__}: {e}")
            raise SystemExit(1)
    else:
        print("[warn] ANTHROPIC_API_KEY not set — analysts use deterministic Phase-1 "
              "readings only (no LLM refinement). Fully runnable; just not LLM-driven.\n")

    macro, prices = _load_data(args)
    fund = macro_rates_fund(llm_client)

    # (1) one meeting, in full, as of the last available day
    asof = prices.index[-1]
    world = AsOf(asof=asof, macro=macro, prices=prices)
    _print_meeting(fund.run_meeting(world))

    # (2) weekly run: P&L and research scorecard, attributed separately
    print("\n" + "=" * 74)
    print(f"WEEKLY RUN {args.start} → {args.end}  (source={args.source}"
          + (f", regime={args.regime}" if args.source == "synthetic" else "") + ")")
    print("=" * 74)
    bt = run_weekly(fund, macro, prices, args.start, args.end, freq=args.freq)

    print("\nRESEARCH SCORECARD — were the analysts' driver calls right? "
          "(separate from P&L)")
    print(bt.research_scorecard.to_string())

    print("\nFUND P&L — did the netted book make money? (arbitrage + risk quality)")
    for k in ("total_return", "annualized_return", "annualized_vol", "sharpe", "sortino", "max_drawdown", "calmar"):
        if k in bt.metrics:
            print(f"    {k:20s} {bt.metrics[k]:.4f}")


if __name__ == "__main__":
    main()
