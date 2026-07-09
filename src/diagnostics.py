"""Diagnostics runner — surface the internal state the engine computes but the
headline metrics table hides, so we can localize *which* component is driving a
bad Sharpe instead of reading every module top-to-bottom.

Run:  python3 -m src.diagnostics --start 2024-01-01 --end 2024-12-31

It answers, in order:
  1. Is the Sharpe dominated by the risk-free/cash drag rather than bad picks?
     (decompose sharpe into an "excess vs rf" term and see how much is the
     uninvested-cash sleeve earning 0 while rf is charged.)
  2. Is the risk layer actually reaching its vol target, or leaving the book
     near-cash? (final_gross, vol_scale, breadth_haircut, ex-ante vs realized vol.)
  3. Are the agents emitting real, varied signals or mostly neutral?
  4. Which agent's paper portfolio is actually losing (scorecard IC/Sharpe)?
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestConfig, run_backtest
from src.backtest.metrics import TRADING_DAYS, RISK_FREE_ANNUAL, daily_returns
from src.ensemble.attribution import encode_signal


def _rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def cash_drag_decomposition(values: pd.Series) -> None:
    """Show how much of the (negative) Sharpe is the rf drag on a near-cash book
    vs the strategy's own active P&L. The engine credits 0% on uninvested cash
    but metrics.py charges the full rf in the excess-return term."""
    _rule("1. SHARPE DECOMPOSITION — is this a cash/rf-drag artifact?")
    rets = daily_returns(values)
    ann_ret = float((values.iloc[-1] / values.iloc[0]) ** (TRADING_DAYS / max(len(rets), 1)) - 1)
    ann_vol = float(rets.std() * np.sqrt(TRADING_DAYS))
    # Sharpe as currently computed (excess vs rf), and a self-financing variant
    # (no rf charge) — the gap is the phantom cash drag.
    excess = rets - RISK_FREE_ANNUAL / TRADING_DAYS
    sharpe_rf = float(np.sqrt(TRADING_DAYS) * excess.mean() / excess.std()) if excess.std() else float("nan")
    sharpe_self = float(np.sqrt(TRADING_DAYS) * rets.mean() / rets.std()) if rets.std() else float("nan")
    print(f"  annualized return        : {ann_ret:+.2%}")
    print(f"  annualized vol           : {ann_vol:.2%}")
    print(f"  risk-free charged        : {RISK_FREE_ANNUAL:.2%}  (metrics.py RISK_FREE_ANNUAL)")
    print(f"  Sharpe (excess vs rf)    : {sharpe_rf:+.2f}   <- what the report shows")
    print(f"  Sharpe (self-financing)  : {sharpe_self:+.2f}   <- if cash earned rf / rf not charged")
    print(f"  => rf drag accounts for  : {sharpe_rf - sharpe_self:+.2f} of the Sharpe")
    if abs(sharpe_rf - sharpe_self) > abs(sharpe_self):
        print("  VERDICT: the negative Sharpe is DOMINATED by the rf/cash-drag inconsistency,")
        print("           not by the strategy's active bets. Fix the cash accrual first.")
    else:
        print("  VERDICT: the strategy's own P&L is the main driver, not the rf charge.")


def exposure_and_risk_layer(result) -> None:
    """How much of NAV is actually deployed, and whether the risk layer reaches
    its vol target or leaves the book near-cash."""
    _rule("2. RISK LAYER — is the book near-cash? does vol targeting reach target?")
    w = result.weights_over_time
    gross = w.abs().sum(axis=1)
    net = w.sum(axis=1)
    print(f"  avg gross exposure (sum|w|) : {gross.mean():.3f}   (1.0 = fully invested, cash = 1 - gross)")
    print(f"  avg net  exposure (sum w)   : {net.mean():+.3f}")
    print(f"  avg cash sleeve (~1 - gross): {max(0.0, 1 - gross.mean()):.1%}")
    print(f"  max gross over run          : {gross.max():.3f}")

    hist = result.risk_diag_history
    if not hist:
        print("  (no risk diagnostics — risk manager disabled?)")
        return
    rows = []
    for d in sorted(hist):
        diag = hist[d]
        rows.append({
            "date": d.date(),
            "pre_vol": diag.get("pre_scale_vol"),
            "vol_scale": diag.get("vol_scale"),
            "breadth": diag.get("effective_breadth"),
            "n_pos": diag.get("n_positions"),
            "haircut": diag.get("breadth_haircut"),
            "final_scale": diag.get("final_scale"),
            "final_gross": diag.get("final_gross"),
            "post_vol": diag.get("post_scale_vol"),
        })
    df = pd.DataFrame(rows).set_index("date")
    print("\n  per-rebalance risk-layer scaling (first / last 5 shown):")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df.head().to_string())
        if len(df) > 10:
            print("  ...")
            print(df.tail().to_string())

    # target vol lives on RiskConfig default (0.10 unless changed)
    post = df["post_vol"].dropna()
    if len(post):
        print(f"\n  ex-ante post-scale vol (avg): {post.mean():.3f}  <- what the risk layer THINKS it sized to")
    realized = daily_returns(result.values)
    print(f"  realized annualized vol     : {realized.std() * np.sqrt(TRADING_DAYS):.3f}  <- what actually happened")
    print("  (large gap => cov estimate stale, or net exposure cancels out day-to-day)")


def signal_distribution(result) -> None:
    """Are agents emitting real, varied signals, or mostly neutral/zero — which
    would flatten the book regardless of the risk layer."""
    _rule("3. SIGNAL DISTRIBUTION — are agents actually taking views?")
    # collect every (agent -> signal) across all rebalance dates
    counts: dict[str, dict[str, int]] = {}
    conf_sum: dict[str, float] = {}
    conf_n: dict[str, int] = {}
    for d, day in result.rebalance_signals.items():
        for agent, sigs in day.items():
            c = counts.setdefault(agent, {"bullish": 0, "bearish": 0, "neutral": 0})
            for sym, s in sigs.items():
                sig = s.get("signal", "neutral")
                c[sig] = c.get(sig, 0) + 1
                conf_sum[agent] = conf_sum.get(agent, 0.0) + float(s.get("confidence", 0.0))
                conf_n[agent] = conf_n.get(agent, 0) + 1
    for agent, c in counts.items():
        total = sum(c.values()) or 1
        avg_conf = conf_sum.get(agent, 0.0) / max(conf_n.get(agent, 1), 1)
        print(f"  {agent:18s} bull={c['bullish']:4d} bear={c['bearish']:4d} "
              f"neutral={c['neutral']:4d}  ({c['neutral']/total:.0%} neutral, avg conf {avg_conf:.0f}%)")
    if not counts:
        print("  (no signals recorded)")


def scorecard_and_weights(result) -> None:
    """Which agent is the actual drag: per-agent paper-portfolio quality and the
    weight the ensemble assigned it."""
    _rule("4. PER-AGENT ATTRIBUTION — which agent is the drag?")
    dates = sorted(result.scorecards_history)
    if dates:
        last = result.scorecards_history[dates[-1]]
        print(f"  final scorecard (as of {dates[-1].date()}):")
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(last.round(3).to_string())
    print("\n  agent weight evolution (first / last 3 rebalances):")
    aw = result.agent_weights_history
    if len(aw):
        with pd.option_context("display.width", 200):
            print(aw.round(3).head(3).to_string())
            if len(aw) > 6:
                print("  ...")
                print(aw.round(3).tail(3).to_string())
    print(f"\n  fired agents: {result.fired_agents or 'none'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--rebalance-freq", default="W-FRI")
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--equity-data-source", default="yfinance",
                    choices=["yfinance", "financialdatasets", "wrds"])
    ap.add_argument("--no-risk-manager", action="store_true")
    ap.add_argument("--pm-mode", default="mechanical", choices=["mechanical", "llm"])
    args = ap.parse_args()

    llm_client = None
    if os.environ.get("ANTHROPIC_API_KEY"):
        from src.llm.anthropic_client import AnthropicClient
        llm_client = AnthropicClient(model=args.model)
    else:
        print("[warn] ANTHROPIC_API_KEY not set — signals will be neutral/0%, so the "
              "book will be flat. Diagnostics still validate the plumbing, but section 3 "
              "will (correctly) show all-neutral.\n")

    cfg = BacktestConfig(
        start=args.start, end=args.end, rebalance_freq=args.rebalance_freq,
        model=args.model, use_risk_manager=not args.no_risk_manager,
        pm_mode=args.pm_mode, equity_data_source=args.equity_data_source,
    )
    print(f"Diagnostics: {args.start}..{args.end}, freq={args.rebalance_freq}, "
          f"source={args.equity_data_source}, risk_mgr={not args.no_risk_manager}, pm={args.pm_mode}")
    result = run_backtest(cfg, llm_client=llm_client)

    _rule("HEADLINE METRICS")
    for k, v in result.metrics.items():
        print(f"  {k:22s}: {v:+.4f}" if isinstance(v, float) else f"  {k:22s}: {v}")

    cash_drag_decomposition(result.values)
    exposure_and_risk_layer(result)
    signal_distribution(result)
    scorecard_and_weights(result)


if __name__ == "__main__":
    main()
