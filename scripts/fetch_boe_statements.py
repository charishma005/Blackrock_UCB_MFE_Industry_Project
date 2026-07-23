"""Fetch Bank of England Monetary Policy Summaries -> data/boe/documents.jsonl.

Source: page URLs discovered through the BoE sitemap API
(``https://www.bankofengland.co.uk/_api/sitemap/getsitemap`` — the plain
/sitemap.xml redirects to an error page), filtered to
``/monetary-policy-summary-and-minutes/``. That yields every summary page back
to the first one (August 2015, when the BoE began publishing a summary with
each decision; before 2015 there was no statement when rates were unchanged —
the pre-2015 record is the minutes, deliberately out of scope).

The release date is parsed from the summary text itself ("...meeting ending
on 3 August 2016...") **plus one day**: the MPC announcement is published at
noon on the day after the meeting ends. The sidebar <time> tags on the page
are recent-news links, not the article date — do not use them.

Usage:  python scripts/fetch_boe_statements.py
"""
from __future__ import annotations

import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._cb_text import fetch, html_to_text, make_doc, write_docs  # noqa: E402

SITEMAP = "https://www.bankofengland.co.uk/_api/sitemap/getsitemap"
PREFIX = "https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes/"
OUT = Path(__file__).resolve().parents[1] / "data" / "boe" / "documents.jsonl"

_MEETING = re.compile(r"meeting (?:ending|held) on (\d{1,2} \w+ \d{4})")
_H1 = re.compile(r"<h1[^>]*>(.*?)</h1>", re.S)


def main() -> None:
    sitemap = fetch(SITEMAP)
    urls = sorted(set(re.findall(re.escape(PREFIX) + r"[^<\s]+", sitemap)))
    print(f"{len(urls)} summary pages in sitemap", file=sys.stderr)

    docs, seen_dates = [], set()
    for url in urls:
        try:
            page = fetch(url)
        except RuntimeError as e:
            print(f"[warn] {url}: {e}", file=sys.stderr)
            continue
        text = html_to_text(page)
        # Trim the header chrome (related-links block, PDF dialog, publish
        # date) — the body proper starts at the "Monetary Policy Summary"
        # heading; the announcement headline is kept in the title field.
        cut = text.find("Monetary Policy Summary,")
        if cut < 0:
            cut = text.find("The Bank of England’s Monetary Policy Committee")
        if cut > 0:
            text = text[cut:]
        m = _MEETING.search(text)
        if not m:
            print(f"[warn] no meeting date on {url} — skipped", file=sys.stderr)
            continue
        meeting_end = datetime.strptime(m.group(1), "%d %B %Y").date()
        iso = (meeting_end + timedelta(days=1)).isoformat()  # noon-next-day release
        if iso in seen_dates:
            continue
        seen_dates.add(iso)
        h1 = _H1.search(page)
        title = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", h1.group(1))).strip() \
            if h1 else "Monetary Policy Summary"
        docs.append(make_doc(f"boe_stmt_{iso}", iso, title, url, text))
    write_docs(OUT, docs)


if __name__ == "__main__":
    main()
