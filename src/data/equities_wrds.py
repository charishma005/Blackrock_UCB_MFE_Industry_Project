"""Equity fundamentals via WRDS (Compustat), point-in-time via the `rdq` field.

Point-in-time correctness: Compustat's quarterly fundamentals table
(comp.fundq) carries BOTH `datadate` (the fiscal period the numbers describe)
and `rdq` (report date of quarterly earnings — when the filing actually
became public). Filtering `rdq <= as_of_date` is the standard academic-
finance method for eliminating look-ahead bias: it answers "what was known
publicly by this date," which is a stronger and more literal guarantee than
financialdatasets.ai's `report_period_lte` (which filters on the reported
period itself, trusting the vendor's own point-in-time bookkeeping).

Requires: pip install wrds, and a WRDS account with Compustat + CRSP access
(a university subscription, e.g. via Berkeley, covers this). First call to
wrds.Connection() prompts for username/password (or reads ~/.pgpass) and
this can take several seconds — WRDS is a remote Postgres-backed service,
not a lightweight REST API, so expect higher latency than financialdatasets.ai.

FIELD NAME CAVEAT: Compustat's exact variable list depends on your
institution's subscription tier. The columns below (niq, dpq, capxy, etc.)
are the standard Compustat quarterly fundamentals items and should be
present in any standard academic subscription, but verify with
`wrds.Connection().describe_table('comp', 'fundq')` before relying on this
in a real run — this module was written against documented Compustat
conventions, not tested against live WRDS credentials.
"""
from __future__ import annotations

import os
from functools import lru_cache

import pandas as pd

# Compustat quarterly fundamentals items used here (comp.fundq), TTM-summed
# where they are flow items:
#   niq    = net income (quarterly)
#   dpq    = depreciation & amortization (quarterly)
#   capxy  = capital expenditures (YTD, cumulative within fiscal year — see _ytd_to_quarterly)
#   oancfy = operating cash flow (YTD)
#   fincfy = financing activities cash flow (YTD) — buybacks/dividends live inside this
#   seqq   = stockholders' equity (quarterly, stock item, not summed)
#   dlttq + dlcq = long-term + current debt (stock item)
#   cshoq  = shares outstanding (stock item)
#   prccq  = quarterly close price (for market cap = prccq * cshoq)
#   epsfxq = diluted EPS excl. extra items (quarterly)
#   oibdpq = operating income before depreciation (quarterly) — EBITDA proxy


def _get_connection():
    """Open a WRDS connection without prompting on every run.

    Two prompts happen with a bare `wrds.Connection()`: username and password.
    We suppress both:

    - Username: read from $WRDS_USERNAME (the env var run_backtest.py already
      checks for). Passed as `wrds_username=` so wrds never prompts for it.
    - Password: wrds reads it from ~/.pgpass automatically if that file exists.
      Create it ONCE with:  python -c "import wrds; wrds.Connection(wrds_username='YOUR_USER').create_pgpass_file()"
      (it will prompt for the password that one time, then write ~/.pgpass so
      no future run prompts again).

    If $WRDS_USERNAME is unset we fall back to bare Connection() so interactive
    use still works.
    """
    import wrds

    username = os.environ.get("WRDS_USERNAME")
    if username:
        return wrds.Connection(wrds_username=username)
    return wrds.Connection()


def check_connection() -> str:
    """Preflight liveness check: open a WRDS connection AND run a trivial query
    to prove the session is actually usable, then close it.

    Why a query and not just `wrds.Connection()`: the connection object can be
    constructed while the backing Postgres session is unauthenticated or the
    subscription lacks query rights, so `SELECT 1` is what confirms the pipe is
    really alive. Call this ONCE up front (see run_backtest) so a bad
    connection fails in the first second with an actionable message, instead of
    hanging or erroring deep inside the rebalance loop after minutes of setup.

    Returns the WRDS username used on success; raises RuntimeError with guidance
    on any failure.
    """
    try:
        import wrds  # noqa: F401  (import error is itself a failure we want to report)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "`wrds` is not installed. Install with: pip install wrds "
            "(or `pip install \"multi-asset-fund[wrds]\"`)."
        ) from e

    try:
        db = _get_connection()
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Could not OPEN a WRDS connection. Check: WRDS_USERNAME is set, your "
            "password is in ~/.pgpass (create it once with "
            "`python -c \"import wrds; wrds.Connection(wrds_username='YOUR_USER')"
            ".create_pgpass_file()\"`), and that wrds.wharton.upenn.edu is "
            f"reachable from this network. Underlying error: {e}"
        ) from e

    try:
        db.raw_sql("SELECT 1")
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Opened a WRDS connection but a trivial `SELECT 1` FAILED — the "
            "session is not usable (authenticated but query layer or Compustat "
            f"subscription unavailable?). Underlying error: {e}"
        ) from e
    finally:
        try:
            db.close()
        except Exception:  # noqa: BLE001 — best-effort cleanup
            pass

    return os.environ.get("WRDS_USERNAME", "(interactive)")


@lru_cache(maxsize=None)
def _fetch_fundq_history(ticker: str) -> pd.DataFrame:
    """Full quarterly fundamentals history for a ticker, oldest cached once
    per process (WRDS round-trips are slow; don't re-query per rebalance).
    Filtering to `as_of_date` happens client-side on the cached frame.
    """
    db = _get_connection()
    # gvkey lookup via the ticker->gvkey linking table (comp.security), then
    # pull fundq filtered to that gvkey. tic = Compustat's own ticker field.
    query = f"""
        SELECT f.gvkey, f.datadate, f.rdq, f.fqtr, f.niq, f.dpq, f.capxy, f.oancfy,
               f.fincfy, f.seqq, f.dlttq, f.dlcq, f.cshoq, f.prccq, f.epsfxq,
               f.oibdpq, f.saleq
        FROM comp.fundq f
        JOIN comp.security s ON f.gvkey = s.gvkey
        WHERE s.tic = '{ticker}' AND f.indfmt = 'INDL' AND f.datafmt = 'STD'
              AND f.popsrc = 'D' AND f.consol = 'C'
        ORDER BY f.datadate ASC
    """
    df = db.raw_sql(query, date_cols=["datadate", "rdq"])
    db.close()
    return df


def _ytd_to_quarterly(df: pd.DataFrame, col: str) -> pd.Series:
    """Compustat *Y items (capxy, oancfy, fincfy) are YEAR-TO-DATE cumulative
    within each fiscal year, not per-quarter — Q2 - Q1 gives the true Q2
    flow. Resets at the first fiscal quarter of each year.

    Use Compustat's own fiscal-quarter field `fqtr`, NOT the calendar quarter
    of `datadate`. For a firm whose fiscal year doesn't end in December (e.g.
    Apple, fyr=9), fiscal Q1 ends in a calendar quarter other than Q1, so
    keying the reset off `datadate.dt.quarter` de-accumulates at the wrong
    rows and corrupts every YTD-derived flow (capex, OCF, financing CF, FCF).
    """
    is_q1 = df["fqtr"] == 1
    diffed = df[col].diff()
    diffed[is_q1] = df[col][is_q1]  # Q1's YTD figure IS the quarterly figure
    return diffed


def get_equity_facts_bundle(ticker: str, as_of_date: str) -> dict:
    """Point-in-time bundle, filtered by `rdq <= as_of_date`. Same output
    shape as data/equities.py and data/equities_yfinance.py so agents need
    zero changes to consume this source.
    """
    hist = _fetch_fundq_history(ticker)
    visible = hist[hist["rdq"] <= pd.Timestamp(as_of_date)].copy()
    if visible.empty:
        return {"metrics": [], "line_items": [], "market_cap": None,
                "periods_per_year": 4, "latest_report_period": None}

    visible = visible.sort_values("datadate")
    visible["capx_q"] = _ytd_to_quarterly(visible, "capxy")
    visible["oancf_q"] = _ytd_to_quarterly(visible, "oancfy")
    visible["fincf_q"] = _ytd_to_quarterly(visible, "fincfy")

    def ttm(series: pd.Series) -> pd.Series:
        return series.rolling(4, min_periods=4).sum()

    visible["net_income_ttm"] = ttm(visible["niq"])
    visible["dep_ttm"] = ttm(visible["dpq"])
    visible["capex_ttm"] = ttm(visible["capx_q"])
    # SIGN CONVENTION: Compustat's capxy is conventionally POSITIVE (dollar
    # amount spent), not negative — so FCF = operating cash flow MINUS capex.
    # VERIFY this against your actual WRDS pull before trusting FCF-derived
    # signals; sign conventions occasionally vary by data vintage/subscription.
    visible["fcf_ttm"] = visible["oancf_q"].rolling(4, min_periods=4).sum() - visible["capex_ttm"]
    visible["revenue_ttm"] = ttm(visible["saleq"])

    metrics, line_items, market_caps = [], [], []
    for _, row in visible.iloc[::-1].iterrows():  # newest first, matching convention
        # Normalize every extracted value to clean None/float FIRST. WRDS returns
        # pandas nullable dtypes (pd.NA), and `pd.NA` raises TypeError in any bare
        # boolean context (`x or default`, `if x`, `x and y`) — bool(pd.NA) is
        # deliberately ambiguous, unlike None or np.nan. Every value must go
        # through pd.notna() before it's used in a condition or arithmetic default.
        equity = float(row["seqq"]) if pd.notna(row["seqq"]) else None
        dlttq = float(row["dlttq"]) if pd.notna(row["dlttq"]) else 0.0
        dlcq = float(row["dlcq"]) if pd.notna(row["dlcq"]) else 0.0
        debt = dlttq + dlcq
        shares = row["cshoq"] * 1e6 if pd.notna(row["cshoq"]) else None  # Compustat reports shares in millions
        row_market_cap = (row["prccq"] * shares) if pd.notna(row["prccq"]) and shares is not None else None
        market_caps.append(row_market_cap)
        roe = (row["net_income_ttm"] * 1e6 / (equity * 1e6)) if pd.notna(row["net_income_ttm"]) and equity else None
        ebitda = row["oibdpq"] * 1e6 * 4 if pd.notna(row["oibdpq"]) else None  # rough annualization for EV/EBITDA

        metrics.append({
            "revenue": row["revenue_ttm"] * 1e6 if pd.notna(row["revenue_ttm"]) else None,
            "return_on_equity": roe,
            "return_on_invested_capital": None,  # Compustat doesn't give a direct ROIC field; compute from NOPAT if needed
            "beta": None,  # not in fundq; pull from CRSP beta suite separately if needed
            "debt_to_equity": (debt * 1e6 / (equity * 1e6)) if equity else None,
            "enterprise_value_to_ebitda_ratio": ((row_market_cap + debt * 1e6) / ebitda) if row_market_cap is not None and ebitda else None,
        })
        line_items.append({
            "net_income": row["net_income_ttm"] * 1e6 if pd.notna(row["net_income_ttm"]) else None,
            "depreciation_and_amortization": row["dep_ttm"] * 1e6 if pd.notna(row["dep_ttm"]) else None,
            "capital_expenditure": row["capex_ttm"] * 1e6 if pd.notna(row["capex_ttm"]) else None,
            "change_in_working_capital": None,  # not directly in fundq; derive from balance sheet deltas if needed
            "free_cash_flow": row["fcf_ttm"] * 1e6 if pd.notna(row["fcf_ttm"]) else None,
            "outstanding_shares": shares,
            "issuance_or_purchase_of_equity_shares": row["fincf_q"] * 1e6 if pd.notna(row["fincf_q"]) else None,  # net financing CF as a proxy; not buybacks-only
            "dividends_and_other_cash_distributions": None,  # separate Compustat item (dvy) — add if needed
        })

    return {
        "metrics": metrics,
        "line_items": line_items,
        "market_cap": market_caps[0] if market_caps else None,  # most recent row (list is newest-first)
        "periods_per_year": 1,  # already TTM-summed above
        "latest_report_period": visible.iloc[-1]["datadate"].strftime("%Y-%m-%d"),
    }