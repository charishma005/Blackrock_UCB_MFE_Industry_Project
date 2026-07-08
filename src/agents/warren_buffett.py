"""Warren Buffett agent, reimplemented on BaseInvestorAgent.

Deliberately different mechanics from the upstream reference implementation:
  - Owner earnings computed directly (net income + D&A - capex), no 3-way
    maintenance-capex estimation or median blending.
  - No multi-stage DCF. Instead: owner-earnings YIELD (owner earnings /
    market cap) compared against a hurdle rate — closer to how Buffett
    himself has described thinking about it in interviews ("what's my
    return if I bought the whole business today").
  - Moat durability measured as ROE consistency (1 - coefficient of
    variation) rather than a 6-function additive scorecard.
  - Signal decision is left to the LLM (persona-conditioned), matching the
    spirit of the original design choice for this specific agent — but the
    facts it's judging are computed differently.
"""
from __future__ import annotations

from src.agents.base import BaseInvestorAgent
from src.instruments import AssetClass, Instrument

HURDLE_RATE = 0.07  # required owner-earnings yield for "wonderful business at a fair price"


class WarrenBuffettAgent(BaseInvestorAgent):
    name = "warren_buffett"
    covers = {AssetClass.EQUITY}

    def compute_facts(self, instrument: Instrument, data: dict) -> dict:
        metrics: list[dict] = data.get("metrics", [])
        line_items: list[dict] = data.get("line_items", [])
        market_cap: float | None = data.get("market_cap")

        if not metrics or not line_items:
            return {"error": "insufficient data"}

        latest = metrics[0]
        latest_li = line_items[0]

        # Owner earnings (simple form): net income + D&A - capex - Δworking capital
        net_income = latest_li.get("net_income")
        dep = latest_li.get("depreciation_and_amortization")
        capex = latest_li.get("capital_expenditure")
        wc_change = latest_li.get("change_in_working_capital", 0) or 0
        owner_earnings = None
        if net_income is not None and dep is not None and capex is not None:
            owner_earnings = net_income + dep - abs(capex) - wc_change

        owner_earnings_yield = (
            owner_earnings / market_cap if owner_earnings and market_cap else None
        )

        # Moat durability index: ROE consistency across available periods
        roes = [m.get("return_on_equity") for m in metrics if m.get("return_on_equity") is not None]
        moat_index, moat_detail = None, "insufficient ROE history"
        if len(roes) >= 3:
            avg_roe = sum(roes) / len(roes)
            std_roe = (sum((r - avg_roe) ** 2 for r in roes) / len(roes)) ** 0.5
            cv = std_roe / avg_roe if avg_roe else 1.0
            moat_index = max(0.0, min(1.0, 1 - cv)) if avg_roe > 0.15 else 0.0
            moat_detail = f"avg ROE {avg_roe:.1%}, coefficient of variation {cv:.2f}"

        # Management quality: buybacks + dividends (binary, same spirit, compact)
        issuance = latest_li.get("issuance_or_purchase_of_equity_shares")
        dividends = latest_li.get("dividends_and_other_cash_distributions")
        buybacks = bool(issuance and issuance < 0)
        pays_dividend = bool(dividends and dividends < 0)

        # Leverage sanity check (Buffett tolerates leverage only if cash-rich)
        debt_to_equity = latest.get("debt_to_equity")

        return {
            "owner_earnings": owner_earnings,
            "owner_earnings_yield": round(owner_earnings_yield, 4) if owner_earnings_yield else None,
            "hurdle_rate": HURDLE_RATE,
            "moat_index_0to1": round(moat_index, 2) if moat_index is not None else None,
            "moat_detail": moat_detail,
            "buybacks": buybacks,
            "pays_dividend": pays_dividend,
            "debt_to_equity": debt_to_equity,
            "market_cap": market_cap,
        }
