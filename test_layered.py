"""Offline smoke test for the layered agent fund — no API keys, no network, no cost.

Run: python3 test_layered.py

Exercises the whole architecture on synthetic data:
  1. Time integrity — the AsOf gate never leaks the future
  2. Analyst layer — isolated single-driver views on the hawkish regime
  3. PM layer — arbitrate + express: hawkish drivers → a 2s10s FLATTENER,
     DV01-neutral (net duration ≈ 0); dovish drivers → a STEEPENER
  4. Unifying layer — capital allocation + netted book, gross within cap
  5. Research scoring — grades driver calls separately from P&L
  6. Contract stability — DriverView / StrategyTrade round-trip
"""
import sys

import numpy as np
import pandas as pd

from src.layered.analysts.macro_rates import macro_rates_analysts
from src.layered.backtest import run_weekly
from src.layered.contracts import DriverView
from src.layered.meeting import macro_rates_fund
from src.layered.pm.macro_rates import MacroRatesPM
from src.layered.scoring import score_driver_views
from src.layered.synthetic import generate
from src.layered.timeline import AsOf

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    status = "PASS" if condition else "FAIL"
    show = (not condition) and detail != ""
    print(f"[{status}] {name}" + (f" — {detail}" if show else ""))


# ── 1: time integrity ───────────────────────────────────────────────────────
macro, prices = generate("2022-01-01", "2024-12-31", regime="hawkish")
mid = prices.index[len(prices) // 2]
world_mid = AsOf(asof=mid, macro=macro, prices=prices)
check("AsOf.price never sees the future", world_mid.price("IEF").index.max() <= mid)
check("AsOf.series never sees the future", world_mid.series("CPIAUCSL").index.max() <= mid)
check("AsOf.frame never sees the future",
      world_mid.frame().index.max() <= mid if len(world_mid.frame()) else True)

# ── 2: analyst layer (isolated, deterministic) ──────────────────────────────
asof = prices.index[-1]
world = AsOf(asof=asof, macro=macro, prices=prices)
analysts = macro_rates_analysts(llm_client=None)
views = [a.form_view(world) for a in analysts]
by = {v.driver: v for v in views}
check("Four single-driver analysts produced views", len(views) == 4, [v.driver for v in views])
check("Inflation analyst sees inflation rising (hawkish regime)",
      by["inflation"].direction == "up", by["inflation"].direction)
check("Labor analyst sees tightening (hawkish regime)",
      by["labor_tightness"].direction == "up", by["labor_tightness"].direction)
check("Balance-sheet analyst sees runoff/QT (hawkish regime)",
      by["balance_sheet"].direction == "down", by["balance_sheet"].direction)
check("All convictions in [0, 1]", all(0.0 <= v.conviction <= 1.0 for v in views))

# ── 3: PM layer — arbitrate + express ───────────────────────────────────────
pm = MacroRatesPM(llm_client=None)
trade = pm.decide(views, world)
check("Hawkish regime expresses a 2s10s FLATTENER",
      trade.risk["structure"] == "2s10s flattener", trade.risk["structure"])
check("Trade is DV01-neutral (net duration ≈ 0)",
      abs(trade.risk["net_duration"]) < 0.15, trade.risk["net_duration"])
check("Flattener is long the 10y leg (IEF) / short the 2y leg (SHY)",
      trade.legs.get("IEF", 0) > 0 and trade.legs.get("SHY", 0) < 0, trade.legs)
check("Trade legs normalized to unit gross", abs(trade.gross - 1.0) < 1e-6, trade.gross)
check("PM conviction in [0, 1]", 0.0 <= trade.conviction <= 1.0, trade.conviction)

# dovish regime should flip to a steepener
macro_d, prices_d = generate("2022-01-01", "2024-12-31", regime="dovish")
world_d = AsOf(asof=prices_d.index[-1], macro=macro_d, prices=prices_d)
views_d = [a.form_view(world_d) for a in macro_rates_analysts()]
trade_d = MacroRatesPM().decide(views_d, world_d)
check("Dovish regime expresses a 2s10s STEEPENER",
      trade_d.risk["structure"] == "2s10s steepener", trade_d.risk["structure"])

# ── 4: unifying layer ───────────────────────────────────────────────────────
fund = macro_rates_fund(llm_client=None)
result = fund.run_meeting(world)
check("Allocation grants the strategy capital", result.allocation.capital.get("macro_rates", 0) > 0,
      result.allocation.capital)
check("Netted fund book is non-empty", result.book.abs().sum() > 0, result.book.to_dict())
check("Netted gross within the fund's leverage cap",
      result.book.abs().sum() <= fund.allocator.config.max_gross_leverage + 1e-6,
      result.book.abs().sum())

# ── 5: research scoring (separate from P&L) ─────────────────────────────────
# a synthetic analyst that always calls "up" while its level keeps rising = 100% hit
rising = [DriverView(driver="x", asof=pd.Timestamp("2024-01-01") + pd.Timedelta(days=30 * i),
                     direction="up", conviction=0.8, horizon_days=30, level=float(i))
          for i in range(6)]
sc = score_driver_views(rising)
check("Research scorer credits a correct always-up analyst", sc["hit_rate"] == 1.0, sc)
falling = [v.model_copy(update={"direction": "up", "level": float(-i)}) for i, v in enumerate(rising)]
sc2 = score_driver_views(falling)
check("Research scorer penalizes a wrong always-up analyst", sc2["hit_rate"] == 0.0, sc2)

# ── 6: weekly backtest end to end ───────────────────────────────────────────
bt = run_weekly(fund, macro, prices, "2022-06-01", "2024-12-31")
check("Weekly run produced an equity curve", len(bt.values) > 100, len(bt.values))
check("Research scorecard has a row per driver", len(bt.research_scorecard) == 4, bt.research_scorecard.index.tolist())
check("Metrics include Sharpe", "sharpe" in bt.metrics)
check("Equity curve is finite", np.isfinite(bt.values.iloc[-1]), bt.values.iloc[-1])

# ── summary ──────────────────────────────────────────────────────────────
print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
