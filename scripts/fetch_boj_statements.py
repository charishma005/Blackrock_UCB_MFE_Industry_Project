"""Fetch Bank of Japan monetary-policy statements -> data/boj/documents.jsonl.

Source: the BoJ's per-year decision lists at
``https://www.boj.or.jp/en/mopo/mpmdeci/mpr_{YYYY}/index.htm``, which link the
English statements ``k{yymmdd}[a]`` (titled "Statement on Monetary Policy" in
most years, "Announcement of the Monetary Policy Meeting Decisions" in the
mid-2000s). Format varies by era: HTML pages in 2005-06 and 2018+, **PDFs for
roughly 2007-2017** — both are handled (PDF text via pypdf; the statements
are machine-generated and extract cleanly). The release date is the decision
date encoded in the filename (statements publish the same day).

Coverage: 2005-01 -> present (matching the FOMC corpus window).
Usage:  uv run --with pypdf python scripts/fetch_boj_statements.py
"""
from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._cb_text import (  # noqa: E402
    fetch, fetch_bytes, html_to_text, make_doc, pdf_to_text, write_docs)

BASE = "https://www.boj.or.jp"
LIST = BASE + "/en/mopo/mpmdeci/mpr_{year}/index.htm"
OUT = Path(__file__).resolve().parents[1] / "data" / "boj" / "documents.jsonl"
START_YEAR = 2005

_LINK = re.compile(
    r'<a href="(/en/mopo/mpmdeci/(?:state|mpr)_\d{4}/k(\d{6})[a-z]?\.(?:htm|pdf))"'
    r'[^>]*>([^<]*)', re.I)
# The k{yymmdd} filename is the decision-statement marker; the title guard is
# a loose second check because statement titles vary ("Statement on Monetary
# Policy", "Announcement of the Monetary Policy Meeting Decisions", "New
# Framework for Strengthening Monetary Easing...", "Introduction of ...").
_TITLE_OK = re.compile(r"monetary|market operations|interest rate|easing", re.I)


def _iso(yymmdd: str) -> str:
    yy, mm, dd = yymmdd[:2], yymmdd[2:4], yymmdd[4:6]
    return f"20{yy}-{mm}-{dd}"


def main() -> None:
    docs = []
    for year in range(START_YEAR, date.today().year + 1):
        try:
            listing = fetch(LIST.format(year=year))
        except RuntimeError as e:
            print(f"[warn] {year}: {e}", file=sys.stderr)
            continue
        seen_dates = set()
        for href, yymmdd, raw_title in _LINK.findall(listing):
            title = re.sub(r"\s+", " ", raw_title).replace("\xa0", " ").strip()
            iso = _iso(yymmdd)
            if not _TITLE_OK.search(title) or iso in seen_dates:
                continue
            seen_dates.add(iso)
            if href.lower().endswith(".pdf"):
                text = pdf_to_text(fetch_bytes(BASE + href))
            else:
                text = html_to_text(fetch(BASE + href))
            docs.append(make_doc(f"boj_stmt_{iso}", iso, title, BASE + href, text))
        print(f"{year}: {len(seen_dates)} statements", file=sys.stderr)
    write_docs(OUT, docs)


if __name__ == "__main__":
    main()
