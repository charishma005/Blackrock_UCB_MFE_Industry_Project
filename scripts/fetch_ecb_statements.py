"""Fetch ECB press-conference monetary-policy statements -> data/ecb/documents.jsonl.

Source: the ECB's monetary-policy-statement archive. The human listing page is
JS-rendered, but it lazyloads static per-year fragments at
``.../monetary-policy-statement/{YYYY}/html/index_include.en.html`` whose
``<dt isoDate="...">`` entries carry the date and the statement link — that is
the discovery mechanism here (stable back to 1998). We keep docs titled
"Introductory statement" (pre-2021 name) or "Monetary policy statement",
which exist for every Governing Council monetary-policy meeting; the shorter
"Monetary policy decisions" press release is irregular pre-2012 and is NOT
what we fetch. Text includes the Q&A when the page carries it.

Coverage: 2005-01 -> present (matching the FOMC corpus window).
Usage:  python scripts/fetch_ecb_statements.py
"""
from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._cb_text import fetch, html_to_text, make_doc, write_docs  # noqa: E402

BASE = "https://www.ecb.europa.eu"
FRAGMENT = (BASE + "/press/press_conference/monetary-policy-statement"
            "/{year}/html/index_include.en.html")
OUT = Path(__file__).resolve().parents[1] / "data" / "ecb" / "documents.jsonl"
START_YEAR = 2005

_TITLE_OK = re.compile(r"introductory statement|monetary policy statement", re.I)
_ENTRY = re.compile(
    r'<dt isoDate="(\d{4}-\d{2}-\d{2})".*?'
    r'<div class="title"><a href="([^"]+\.en\.html)"[^>]*>(.*?)</a>',
    re.S)


def main() -> None:
    docs = []
    for year in range(START_YEAR, date.today().year + 1):
        try:
            frag = fetch(FRAGMENT.format(year=year))
        except RuntimeError as e:
            print(f"[warn] {year}: {e}", file=sys.stderr)
            continue
        seen_dates = set()
        for iso, href, raw_title in _ENTRY.findall(frag):
            title = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", raw_title)).strip()
            if not _TITLE_OK.search(title) or iso in seen_dates:
                continue
            seen_dates.add(iso)
            page = fetch(BASE + href if href.startswith("/") else href)
            text = html_to_text(page)
            docs.append(make_doc(f"ecb_stmt_{iso}", iso, title,
                                 BASE + href, text))
        print(f"{year}: {len(seen_dates)} statements", file=sys.stderr)
    write_docs(OUT, docs)


if __name__ == "__main__":
    main()
