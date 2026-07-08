"""Aswath Damodaran agent, reimplemented on BaseInvestorAgent.

Deliberately different mechanics from the upstream reference implementation:
  - No 10-year growth-fade DCF. Instead: a steady-state Gordon Growth model
    on FCFF, with the growth rate capped so it never approaches the discount
    rate (avoids the fade-schedule complexity, trades precision for a
    single transparent number Damodaran could sanity-check on a napkin).
  - Reinvestment efficiency expressed as a SPREAD (ROIC - cost of capital)
    rather than a binary "ROIC > 10%" threshold — the spread is what
    Damodaran's own teaching materials emphasize as the value-creation signal.
  - Relative valuation cross-check uses EV/EBITDA vs the company's own
    3-year average (not P/E vs a 5-year median) — a different multiple and
    a different lookback.
  - CAPM cost of equity computation is a standard formula (not proprietary
    to any implementation) and is used the same way; documented here.
"""
from __future__ import annotations

from src.agents.base import BaseInvestorAgent
from src.instruments import AssetClass, Instrument

RISK_FREE_RATE = 0.04
EQUITY_RISK_PREMIUM = 0.05


class AswathDamodaranAgent(BaseInvestorAgent):
    name = "aswath_damodaran"
    covers = {AssetClass.EQUITY}

    def compute_facts(self, instrument: Instrument, data: dict) -> dict:
        metrics: list[dict] = data.get("metrics", [])
        line_items: list[dict] = data.get("line_items", [])
        market_cap: float | None = data.get("market_cap")

        if not metrics or not line_items:
            return {"error": "insufficient data"}

        latest = metrics[0]
        latest_li = line_items[0]

        beta = latest.get("beta") or 1.0
        cost_of_equity = RISK_FREE_RATE + beta * EQUITY_RISK_PREMIUM

        # Reinvestment efficiency spread
        roic = latest.get("return_on_invested_capital")
        spread = (roic - cost_of_equity) if roic is not None else None

        # Revenue CAGR informs the reinvestment story, but perpetuity growth
        # must stay well below the discount rate — Gordon Growth is acutely
        # sensitive as g -> r, so we cap the TERMINAL growth rate at a
        # conservative long-run ceiling (not near cost of equity). Recent
        # revenue CAGR is reported separately as color on the story, not fed
        # directly into the perpetuity.
        revs = [m.get("revenue") for m in reversed(metrics) if m.get("revenue")]
        cagr = None
        if len(revs) >= 2 and revs[0] > 0:
            cagr = (revs[-1] / revs[0]) ** (1 / (len(revs) - 1)) - 1
        TERMINAL_GROWTH_CEILING = 0.03  # long-run GDP-ish growth, standard perpetuity practice
        growth = min(cagr, TERMINAL_GROWTH_CEILING) if cagr is not None else 0.02
        growth = max(growth, -0.01)  # floor so terminal value doesn't blow up

        # Steady-state Gordon growth on FCFF
        fcff = latest_li.get("free_cash_flow")
        shares = latest_li.get("outstanding_shares")
        intrinsic_value, intrinsic_per_share, margin_of_safety = None, None, None
        if fcff and shares and cost_of_equity > growth:
            intrinsic_value = fcff * (1 + growth) / (cost_of_equity - growth)
            intrinsic_per_share = intrinsic_value / shares
            if market_cap:
                margin_of_safety = (intrinsic_value - market_cap) / market_cap

        # Relative valuation: current EV/EBITDA vs own 3yr average
        ev_ebitdas = [m.get("enterprise_value_to_ebitda_ratio") for m in metrics[:3] if m.get("enterprise_value_to_ebitda_ratio")]
        relative_flag = None
        if len(ev_ebitdas) >= 2:
            current, avg = ev_ebitdas[0], sum(ev_ebitdas) / len(ev_ebitdas)
            relative_flag = "expensive vs own history" if current > 1.3 * avg else (
                "cheap vs own history" if current < 0.7 * avg else "in line with own history"
            )

        return {
            "beta": beta,
            "cost_of_equity": round(cost_of_equity, 4),
            "revenue_cagr": round(cagr, 4) if cagr is not None else None,
            "growth_assumption_used": round(growth, 4),
            "reinvestment_spread_roic_minus_coe": round(spread, 4) if spread is not None else None,
            "intrinsic_value_gordon_growth": intrinsic_value,
            "intrinsic_value_per_share": round(intrinsic_per_share, 2) if intrinsic_per_share else None,
            "margin_of_safety": round(margin_of_safety, 4) if margin_of_safety is not None else None,
            "relative_valuation_flag": relative_flag,
            "market_cap": market_cap,
        }
