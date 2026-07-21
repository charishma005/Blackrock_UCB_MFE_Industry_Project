"""Fetch FRED series into the vendored ``data/fred/`` directory.

The analyst layer runs offline against the CSVs already in ``data/fred/``. Three
personas need series not in that default set — ``WALCL`` (balance_sheet), ``NFCI``
(financial_conditions), ``T10YIE`` (inflation_expectations) — and this is the one
step that needs a ``FRED_API_KEY`` (free: fred.stlouisfed.org).

    export FRED_API_KEY=...
    python scripts/fetch_fred.py                 # the three missing series
    python scripts/fetch_fred.py WALCL NFCI       # named series
    python scripts/fetch_fred.py --all            # refresh the full vendored set

CSVs are written **observation-dated** (``observation_date,<ID>``), exactly like the
existing files, so ``fred_local.load_series`` applies the publication lag uniformly
on load. ``fetch_fred`` returns release-dated data, so the lag is subtracted back out
before saving — storing the raw observation date, never a double-shifted one.
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.data.fred_local import available  # noqa: E402
from src.data.markets import PUBLICATION_LAG_DAYS, fetch_fred  # noqa: E402

OUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "fred"

# The series the fetch-dependent personas declare, absent from the default vendored set.
MISSING = ["WALCL", "NFCI", "T10YIE"]

START = "2000-01-01"   # before the 2005 analyst window, for 12-month feature warmup


def fetch_one(series_id: str, start: str, end: str) -> int:
    s = fetch_fred(series_id, start, end)              # release-dated
    lag = PUBLICATION_LAG_DAYS.get(series_id, 0)
    if lag:
        s.index = s.index - pd.Timedelta(days=lag)     # back to observation date for storage
    df = s.rename(series_id).rename_axis("observation_date").reset_index()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / f"{series_id}.csv", index=False)
    return len(df)


def main() -> None:
    args = sys.argv[1:]
    if args == ["--all"]:
        ids = sorted(set(available()) | set(MISSING))
    elif args:
        ids = args
    else:
        ids = MISSING

    end = str(pd.Timestamp.today().date())
    print(f"fetching {len(ids)} series into {OUT_DIR}  ({START} → {end})")
    for sid in ids:
        try:
            n = fetch_one(sid, START, end)
            print(f"  {sid:10s} {n} rows")
        except Exception as e:  # noqa: BLE001
            print(f"  {sid:10s} FAILED — {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
