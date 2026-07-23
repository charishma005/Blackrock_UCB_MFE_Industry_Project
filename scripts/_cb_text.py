"""Shared plumbing for the central-bank statement fetchers (ECB/BoE/BoJ).

Stdlib-only (urllib + regex HTML stripping — the pages are simple enough that
a parser dependency isn't warranted). Each fetcher writes
``data/<bank>/documents.jsonl`` in the FOMC corpus schema subset consumed by
``FomcCorpus``: ``doc_type``/``release_date``/``text`` plus provenance fields.
"""
from __future__ import annotations

import html
import json
import re
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36")
SLEEP = 0.5  # polite delay between requests


def fetch(url: str, retries: int = 3) -> str:
    """GET a URL with a browser UA, small retry loop, and polite delay."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
            time.sleep(SLEEP)
            return raw.decode("utf-8", errors="replace")
        except Exception as e:  # noqa: BLE001 — retry any transport error
            last = e
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"failed after {retries} tries: {url}") from last


def fetch_bytes(url: str, retries: int = 3) -> bytes:
    """GET raw bytes (for PDF targets), same retry/delay policy as fetch()."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
            time.sleep(SLEEP)
            return raw
        except Exception as e:  # noqa: BLE001 — retry any transport error
            last = e
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"failed after {retries} tries: {url}") from last


def pdf_to_text(raw: bytes) -> str:
    """Text of a machine-generated PDF (BoJ statements). Needs pypdf —
    a scripts-only dependency: run with ``uv run --with pypdf``."""
    import io

    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(raw))
    pages = [(page.extract_text() or "") for page in reader.pages]
    text = "\n".join(pages)
    return re.sub(r"[ \t]+", " ", text).strip()


def html_to_text(page: str, main_only: bool = True) -> str:
    """Visible text of a page: prefer <main>, drop script/style, keep
    paragraph breaks so sentence splitting downstream stays sane."""
    body = page
    if main_only:
        m = re.search(r"<main[^>]*>(.*?)</main>", page, re.S | re.I)
        if m:
            body = m.group(1)
    body = re.sub(r"<(script|style|nav|header|footer)[^>]*>.*?</\1>", " ",
                  body, flags=re.S | re.I)
    body = re.sub(r"</(p|div|li|h[1-6]|tr)>|<br\s*/?>", "\n", body, flags=re.I)
    body = re.sub(r"<[^>]+>", " ", body)
    body = html.unescape(body)
    body = re.sub(r"[ \t]+", " ", body)
    body = re.sub(r"\s*\n\s*", "\n", body).strip()
    return body


def make_doc(doc_id: str, release_date: str, title: str, source_url: str,
             text: str) -> dict:
    return {
        "doc_id": doc_id,
        "doc_type": "statement",
        "release_date": release_date,
        "title": title,
        "source_url": source_url,
        "text": text,
        "n_words": len(text.split()),
        "fetch_ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# Unicode line/paragraph separators and stray control chars inside text would
# survive json.dumps(ensure_ascii=False) raw — and FomcCorpus reads the file
# with str.splitlines(), which splits on them, corrupting the JSONL. Scrub
# them to plain spaces before writing.
_LINE_BREAKERS = re.compile("[  \x85\x0b\x0c\r]")


def write_docs(out_path: Path, docs: list[dict]) -> None:
    """Sorted by release_date, one JSON object per line (splitlines-safe)."""
    docs = sorted(docs, key=lambda d: d["release_date"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for d in docs:
            d = {**d, "text": _LINE_BREAKERS.sub(" ", d["text"]),
                 "title": _LINE_BREAKERS.sub(" ", d["title"])}
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"{len(docs)} docs -> {out_path} "
          f"({docs[0]['release_date']} .. {docs[-1]['release_date']})")
