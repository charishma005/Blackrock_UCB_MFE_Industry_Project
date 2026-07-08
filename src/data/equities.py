"""Equity fundamentals data — financialdatasets.ai free tier (no key needed for
AAPL/MSFT/NVDA/GOOGL/TSLA/AMZN/META).

Endpoint shapes reference virattt/ai-hedge-fund (MIT license) src/tools/api.py,
reimplemented standalone here (no shared cache/model classes, dict-based
returns) so this repo has no runtime dependency on the upstream package.
"""
from __future__ import annotations

import os
import time

import requests

BASE = "https://api.financialdatasets.ai"


def _get(url: str, api_key: str | None = None, max_retries: int = 3) -> dict:
    headers = {}
    key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
    if key:
        headers["X-API-KEY"] = key

    for attempt in range(max_retries + 1):
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code == 429 and attempt < max_retries:
            time.sleep(30 + 30 * attempt)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Failed after {max_retries} retries: {url}")


def get_financial_metrics(ticker: str, end_date: str, period: str = "ttm", limit: int = 5) -> list[dict]:
    url = f"{BASE}/financial-metrics/?ticker={ticker}&report_period_lte={end_date}&limit={limit}&period={period}"
    return _get(url).get("financial_metrics", [])


def search_line_items(ticker: str, items: list[str], end_date: str, limit: int = 8) -> list[dict]:
    url = f"{BASE}/financials/search/line-items"
    headers = {}
    key = os.environ.get("FINANCIAL_DATASETS_API_KEY")
    if key:
        headers["X-API-KEY"] = key
    r = requests.post(url, headers=headers, json={
        "tickers": [ticker], "line_items": items, "end_date": end_date, "limit": limit,
    }, timeout=30)
    r.raise_for_status()
    return r.json().get("search_results", [])


def get_market_cap(ticker: str, end_date: str) -> float | None:
    url = f"{BASE}/company/facts/?ticker={ticker}"
    data = _get(url).get("company_facts", {})
    metrics = get_financial_metrics(ticker, end_date, limit=1)
    if metrics:
        return metrics[0].get("market_cap")
    return data.get("market_cap")


EQUITY_LINE_ITEMS = [
    "net_income", "depreciation_and_amortization", "capital_expenditure",
    "change_in_working_capital", "free_cash_flow", "outstanding_shares",
    "issuance_or_purchase_of_equity_shares", "dividends_and_other_cash_distributions",
]


def get_equity_facts_bundle(ticker: str, as_of_date: str) -> dict:
    """Point-in-time equivalent of data/equities_yfinance.get_equity_facts_bundle.

    Every field is filtered to `report_period_lte=as_of_date`, so this is the
    fix for the equity look-ahead bias: at backtest date `asof`, the agent
    only sees financials that had actually been REPORTED by that date, not
    today's (2026) fundamentals. period="ttm" means each period's flow items
    are already annualized by the API, so no manual TTM aggregation is
    needed here (contrast with the yfinance version).

    `latest_report_period` lets the caller detect whether the underlying
    filing actually changed since the last rebalance — equities file
    quarterly, so re-querying the LLM every week on an unchanged filing
    wastes money. See backtest/engine.py's point-in-time mode.
    """
    metrics = get_financial_metrics(ticker, as_of_date, period="ttm", limit=5)
    line_items = search_line_items(ticker, EQUITY_LINE_ITEMS, as_of_date, limit=8)
    market_cap = get_market_cap(ticker, as_of_date)
    latest_report_period = metrics[0].get("report_period") if metrics else None
    return {
        "metrics": metrics,
        "line_items": line_items,
        "market_cap": market_cap,
        "periods_per_year": 1,  # already TTM/annual from the API — no aggregation needed
        "latest_report_period": latest_report_period,
    }


def get_insider_trades(ticker: str, end_date: str, limit: int = 50) -> list[dict]:
    url = f"{BASE}/insider-trades/?ticker={ticker}&filing_date_lte={end_date}&limit={limit}"
    return _get(url).get("insider_trades", [])
