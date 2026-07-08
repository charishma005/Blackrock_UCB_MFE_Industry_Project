"""Offline smoke test — no API keys, no network calls, no cost.

Run: python3 test_agents.py

Exercises:
  1. Warren Buffett + Aswath Damodaran compute_facts() on synthetic data
  2. judge() fallback path (llm_client=None -> neutral, never crashes)
  3. Metrics module sanity (Sharpe on a known synthetic series)
  4. Attribution + weighting pipeline on synthetic signals
"""
import sys

from src.agents.aswath_damodaran import AswathDamodaranAgent
from src.agents.warren_buffett import WarrenBuffettAgent
from src.backtest.metrics import sharpe_ratio, summary
from src.ensemble.attribution import AttributionTracker
from src.ensemble.weights import AgentPolicy, WeightManager
from src.instruments import AssetClass, Instrument

import numpy as np
import pandas as pd

PASS, FAIL = [], []


def check(name: str, condition: bool, detail=""):
    (PASS if condition else FAIL).append(name)
    status = "PASS" if condition else "FAIL"
    show_detail = (not condition) and detail is not None and (not isinstance(detail, str) or detail != "")
    print(f"[{status}] {name}" + (f" — {detail}" if show_detail else ""))


# ── 1 & 2: agent facts + judge fallback ─────────────────────────────────────
inst = Instrument("AAPL", AssetClass.EQUITY, "Apple")
fake_data = {
    "metrics": [
        {"return_on_equity": 1.5, "return_on_invested_capital": 0.35, "beta": 1.1,
         "enterprise_value_to_ebitda_ratio": 25, "debt_to_equity": 1.6},
        {"return_on_equity": 1.4, "enterprise_value_to_ebitda_ratio": 22},
        {"return_on_equity": 1.45, "enterprise_value_to_ebitda_ratio": 20},
        {"return_on_equity": 1.3, "revenue": 350e9},
        {"revenue": 300e9},
    ],
    "line_items": [
        {"net_income": 95e9, "depreciation_and_amortization": 11e9, "capital_expenditure": -10e9,
         "change_in_working_capital": 2e9, "free_cash_flow": 100e9, "outstanding_shares": 15e9,
         "issuance_or_purchase_of_equity_shares": -90e9, "dividends_and_other_cash_distributions": -15e9},
    ],
    "market_cap": 3.8e12,
}

buffett = WarrenBuffettAgent(llm_client=None)
damo = AswathDamodaranAgent(llm_client=None)

facts_b = buffett.compute_facts(inst, fake_data)
facts_d = damo.compute_facts(inst, fake_data)

check("Buffett owner_earnings computed", facts_b.get("owner_earnings") is not None, facts_b)
check("Buffett moat_index in [0,1]", facts_b.get("moat_index_0to1") is not None and 0 <= facts_b["moat_index_0to1"] <= 1, facts_b)
check("Damodaran cost_of_equity ~ CAPM (0.09-0.10)", facts_d.get("cost_of_equity") is not None and 0.08 < facts_d["cost_of_equity"] < 0.11, facts_d)
check("Damodaran margin_of_safety computed", facts_d.get("margin_of_safety") is not None, facts_d)

sig_b = buffett.judge(inst, facts_b)
sig_d = damo.judge(inst, facts_d)
check("Buffett judge() falls back cleanly with no LLM", sig_b.signal == "neutral" and sig_b.confidence == 0)
check("Damodaran judge() falls back cleanly with no LLM", sig_d.signal == "neutral" and sig_d.confidence == 0)

# ── 3: metrics sanity ───────────────────────────────────────────────────────
np.random.seed(0)
dates = pd.bdate_range("2024-01-01", periods=252)
# known process: mean daily return 0.0006, vol 0.01 -> Sharpe should be roughly
# (0.0006*252 - 0.0434) / (0.01*sqrt(252)) ~ 0.85, wide tolerance since it's random
rets = pd.Series(np.random.normal(0.0006, 0.01, 252), index=dates)
vals = (1 + rets).cumprod() * 100_000
s = sharpe_ratio(rets)
check("Sharpe ratio in plausible range", -1 < s < 3, f"got {s:.2f}")
summ = summary(vals)
check("summary() returns all expected keys", {"sharpe", "sortino", "max_drawdown", "calmar"} <= summ.keys())

# ── 4: attribution + weighting pipeline ─────────────────────────────────────
asset_rets = pd.DataFrame(np.random.normal(0.0004, 0.012, (150, 2)), index=pd.bdate_range("2024-01-01", periods=150), columns=["AAPL", "TLT"])
tracker = AttributionTracker()
for d in asset_rets.index[::20]:
    tracker.record("good", d, {"AAPL": {"signal": "bullish", "confidence": 80}})
    tracker.record("bad", d, {"AAPL": {"signal": "bearish", "confidence": 90}})
scorecard = tracker.scorecard(asset_rets)
check("Scorecard has one row per agent", len(scorecard) == 2, scorecard)

wm = WeightManager(temperature=0.5, policies={"bad": AgentPolicy(fire_threshold=-0.2, fire_after=1)})
weights = wm.update(scorecard)
check("Weights sum to ~1 across surviving agents", weights and abs(sum(weights.values()) - 1.0) < 1e-6, weights)

# ── summary ──────────────────────────────────────────────────────────────
print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
