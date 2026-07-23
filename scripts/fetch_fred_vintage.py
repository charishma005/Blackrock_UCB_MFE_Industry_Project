"""Fetch ALFRED vintage release dates into ``data/fred_vintage/`` — the rigorous
alternative to ``fred_local``'s fixed per-series lag shift.

Needs ``FRED_API_KEY``. Unlike ``scripts/fetch_fred.py`` (which vendors a series'
*values*), this vendors each observation's TRUE first-publication date — a second,
release-timing-only file that ``fred_local.load_series`` prefers over
``markets.PUBLICATION_LAG_DAYS`` whenever it exists for a series. Not run by default:
a series with no vintage file here keeps using the fixed-lag approximation, unchanged,
so running this incrementally (one series at a time, as a key and time allow) is safe.

    export FRED_API_KEY=...
    python scripts/fetch_fred_vintage.py                    # every series already in data/fred/
    python scripts/fetch_fred_vintage.py CPIAUCSL UNRATE     # named series
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.data.fred_local import available  # noqa: E402
from src.data.markets import fetch_fred_vintage  # noqa: E402

OUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "fred_vintage"

START = "2000-01-01"   # matches scripts/fetch_fred.py's warmup window


def fetch_one(series_id: str, start: str, end: str) -> int:
    release = fetch_fred_vintage(series_id, start, end)
    df = (release.rename("first_release_date")
                  .rename_axis("observation_date")
                  .reset_index())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / f"{series_id}.csv", index=False)
    return len(df)


def main() -> None:
    args = sys.argv[1:]
    ids = args if args else sorted(available())

    end = str(pd.Timestamp.today().date())
    print(f"fetching vintage release dates for {len(ids)} series into {OUT_DIR}  "
          f"({START} → {end})")
    for sid in ids:
        try:
            n = fetch_one(sid, START, end)
            print(f"  {sid:10s} {n} rows")
        except Exception as e:  # noqa: BLE001
            print(f"  {sid:10s} FAILED — {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
