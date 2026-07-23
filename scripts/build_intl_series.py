"""Build the vendored INTL_* weekly series from the sister repo's Bloomberg panel.

Source: ``../berkeley-mfe-blackrock-2026/total_assets_weekly.csv`` (override with
the ``INTL_SOURCE_CSV`` env var) — a weekly Friday-close Bloomberg export,
observation-dated. A Friday close is observable on that same Friday, so the
observation date IS the decision date and the CSVs are written unshifted — the
same point-in-time contract as ``data/equity/`` (and unlike ``data/fred/``,
which is release-shifted on load). Never add INTL_ ids to PUBLICATION_LAG_DAYS.

The three ``*_RVOL13`` series are precomputed here because the persona op
vocabulary deliberately has no rolling-std (see src/layered/features/ops.py):
13-week rolling std of weekly percent returns, annualized with sqrt(52).

Usage:  python scripts/build_intl_series.py        # writes data/intl/*.csv
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
SOURCE = os.environ.get(
    "INTL_SOURCE_CSV",
    str(_REPO.parent / "berkeley-mfe-blackrock-2026" / "total_assets_weekly.csv"))
OUT_DIR = _REPO / "data" / "intl"

# Bloomberg column -> vendored series id.
COLUMN_MAP = {
    "GTDEM2Y Govt": "INTL_DE2Y",
    "GTDEM10Y Govt": "INTL_DE10Y",
    "GUKG2 Index": "INTL_UK2Y",
    "GUKG10 Index": "INTL_UK10Y",
    "GJGB2 Index": "INTL_JP2Y",
    "GJGB10 Index": "INTL_JP10Y",
    "USGG10YR Index": "INTL_US10Y",
    "EURUSD Curncy": "INTL_EURUSD",
    "GBPUSD Curncy": "INTL_GBPUSD",
    "JPYUSD Curncy": "INTL_JPYUSD",
    "SXXGV Index": "INTL_SXXGV",
    "UKX Index": "INTL_UKX",
    "GDDUJN Index": "INTL_MSCIJP",
}
# Equity indices that also get a precomputed 13-week annualized realized vol.
VOL_OF = {
    "INTL_SXXGV": "INTL_SXXGV_RVOL13",
    "INTL_UKX": "INTL_UKX_RVOL13",
    "INTL_MSCIJP": "INTL_MSCIJP_RVOL13",
}


def main() -> None:
    df = pd.read_csv(SOURCE)
    idx = pd.to_datetime(df["Dates"], format="%m/%d/%Y")
    if not (idx.dt.dayofweek == 4).all():
        bad = idx[idx.dt.dayofweek != 4]
        sys.exit(f"non-Friday rows in source: {list(bad.dt.date)[:5]} ...")

    missing = [c for c in COLUMN_MAP if c not in df.columns]
    if missing:
        sys.exit(f"source is missing expected columns: {missing}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for col, sid in COLUMN_MAP.items():
        s = (pd.Series(pd.to_numeric(df[col], errors="coerce").values,
                       index=idx, name=sid)
             .dropna().sort_index())
        s.rename_axis("date").to_csv(OUT_DIR / f"{sid}.csv")
        written.append((sid, len(s)))
        if sid in VOL_OF:
            vid = VOL_OF[sid]
            rv = (s.pct_change().rolling(13).std() * (52 ** 0.5)).dropna()
            rv.rename(vid).rename_axis("date").to_csv(OUT_DIR / f"{vid}.csv")
            written.append((vid, len(rv)))

    for sid, n in written:
        print(f"{sid:24s} {n} rows")
    print(f"{len(written)} series -> {OUT_DIR}")


if __name__ == "__main__":
    main()
