"""Equity fundamentals via yfinance — free, no key required.

Drop-in alternative to data/equities.py (financialdatasets.ai version, which
now requires a paid key). Produces the SAME dict shape that
warren_buffett.py / aswath_damodaran.py's compute_facts() expect, so no
agent code changes are needed — only main.py's import line changes.

Known limitations vs financialdatasets.ai (be aware, especially before
citing results from this data source in a writeup):
  - Shorter history: yfinance gives ~4 annual periods, ~5 quarterly.
  - Row names occasionally shift between yfinance versions/tickers.
  - No `report_period_lte` filtering — always returns "most recent as of now",
    not as-of a historical `end_date`. Fine for live/near-term testing;
    NOT suitable for point-in-time backtesting (look-ahead bias risk).

Annual-comparable normalization (fixes the quarterly/annual mismatch bug):
When `use_quarterly=True`, income-statement / cash-flow items are FLOWS
reported per quarter. The agents compare them to ANNUAL thresholds (Buffett's
7% owner-earnings hurdle, Damodaran's Gordon-growth perpetuity on annual FCF),
so a raw quarterly figure understated everything ~4x and biased both agents
bearish. We therefore:
  - line_items flows -> trailing-4-quarter SUM (TTM), the standard way to make
    a quarterly flow annual-comparable for a valuation.
  - metrics ratio numerators (ROE/ROIC) and revenue -> annualized run-rate
    (single quarter x 4). Ratios/CAGR keep one point PER quarter (so the moat
    ROE-consistency calc still has >=3 observations) while comparing correctly
    to annual thresholds. Annual data (`use_quarterly=False`) passes through
    unchanged — both transforms are the identity.
"""
from __future__ import annotations

import yfinance as yf

QUARTERS_PER_YEAR = 4


def _safe_row(df, *names) -> list[float | None]:
    """Return the first matching row (by any of `names`) across all periods,
    oldest -> newest, or a list of None if not found."""
    for name in names:
        if name in df.index:
            row = df.loc[name]
            return [None if v != v else float(v) for v in row.iloc[::-1]]  # oldest first, NaN->None
    return []


def get_equity_facts_bundle(ticker: str, use_quarterly: bool = True) -> dict:
    """One-shot fetch -> {"metrics": [...], "line_items": [...], "market_cap": float}
    matching the shape financialdatasets.ai's fetchers produced, newest-first
    within each list (index 0 = most recent), same convention as the
    original agents expect.
    """
    t = yf.Ticker(ticker)
    info = t.info or {}

    fin = t.quarterly_financials if use_quarterly else t.financials
    bs = t.quarterly_balance_sheet if use_quarterly else t.balance_sheet
    cf = t.quarterly_cashflow if use_quarterly else t.cashflow

    revenue = _safe_row(fin, "Total Revenue", "Operating Revenue")
    net_income = _safe_row(fin, "Net Income", "Net Income Common Stockholders")
    ebit = _safe_row(fin, "EBIT")

    equity = _safe_row(bs, "Stockholders Equity", "Common Stock Equity")
    debt = _safe_row(bs, "Total Debt")
    shares = _safe_row(bs, "Ordinary Shares Number", "Share Issued")
    invested_capital = _safe_row(bs, "Invested Capital")

    fcf = _safe_row(cf, "Free Cash Flow")
    dep = _safe_row(cf, "Depreciation And Amortization", "Depreciation Amortization Depletion")
    capex = _safe_row(cf, "Capital Expenditure")
    wc_change = _safe_row(cf, "Change In Working Capital")
    buybacks = _safe_row(cf, "Repurchase Of Capital Stock")
    dividends = _safe_row(cf, "Cash Dividends Paid", "Common Stock Dividend Paid")

    n = max(len(revenue), len(net_income), 1)

    def pick(lst, i):
        """Point-in-time value at period i (used for balance-sheet STOCKS)."""
        return lst[i] if 0 <= i < len(lst) else None

    def ttm(lst, i):
        """Trailing-4-quarter SUM of a FLOW ending at period i (annual-comparable).
        Passes a single period through when annual. None if <4 quarters available."""
        if not use_quarterly:
            return pick(lst, i)
        lo = i - QUARTERS_PER_YEAR + 1
        if lo < 0:
            return None
        span = [lst[j] for j in range(lo, i + 1) if j < len(lst)]
        present = [v for v in span if v is not None]
        return sum(present) if len(present) == QUARTERS_PER_YEAR else None

    def annualized(val):
        """Scale a single-period FLOW to an annual run-rate (for ratios/CAGR,
        which keep one point per quarter). Identity for annual data."""
        if val is None:
            return None
        return val * QUARTERS_PER_YEAR if use_quarterly else val

    metrics, line_items = [], []
    for i in reversed(range(n)):  # newest-first, matches original convention
        ni_i, eq_i = pick(net_income, i), pick(equity, i)
        ebit_i, ic_i = pick(ebit, i), pick(invested_capital, i)
        roe = (annualized(ni_i) / eq_i) if ni_i is not None and eq_i else None
        roic = (annualized(ebit_i) / ic_i) if ebit_i is not None and ic_i else None
        metrics.append({
            "revenue": annualized(pick(revenue, i)),
            "return_on_equity": roe,
            "return_on_invested_capital": roic,
            "beta": info.get("beta"),
            "debt_to_equity": (pick(debt, i) / eq_i) if pick(debt, i) and eq_i else None,
            "enterprise_value_to_ebitda_ratio": info.get("enterpriseToEbitda"),
        })
        line_items.append({
            "net_income": ttm(net_income, i),
            "depreciation_and_amortization": ttm(dep, i),
            "capital_expenditure": ttm(capex, i),
            "change_in_working_capital": ttm(wc_change, i),
            "free_cash_flow": ttm(fcf, i),
            "outstanding_shares": pick(shares, i) or info.get("sharesOutstanding"),  # stock, point-in-time
            "issuance_or_purchase_of_equity_shares": ttm(buybacks, i),
            "dividends_and_other_cash_distributions": ttm(dividends, i),
        })

    market_cap = info.get("marketCap")
    return {"metrics": metrics, "line_items": line_items, "market_cap": market_cap}
