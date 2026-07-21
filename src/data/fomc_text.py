"""Point-in-time FOMC text source for the LLM input-modality experiment.

The experiment changes ONE thing about the analyst layer: what the Phase-2 LLM
reasons over — the numeric reading (``vector``), the FOMC document (``text``), or
both (``text+vector``). This module serves the ``text`` half: the latest FOMC
statement (or minutes) *available as of* a given date, i.e. with
``release_date <= asof`` — so it can never leak a document the market had not yet
seen. Statements release same-day; minutes ~3 weeks later; both are keyed off
their own ``release_date``, so the as-of filter is correct for either.

The processed corpus lives in the sibling ``watching-crowding-build`` repo
(``FOMC/data/processed/documents.jsonl``); override the path with the
``FOMC_DOCS_PATH`` env var if it sits elsewhere.
"""
from __future__ import annotations

import bisect
import json
import os
from pathlib import Path

import pandas as pd

# .../Blackrock_UCB_MFE_Industry_Project/src/data/fomc_text.py -> parents[3] == dev/
_DEFAULT_PATH = (Path(__file__).resolve().parents[3]
                 / "watching-crowding-build" / "FOMC" / "data" / "processed" / "documents.jsonl")


class FomcCorpus:
    """Latest FOMC ``doc_type`` document available as of a date (no lookahead)."""

    def __init__(self, doc_type: str = "statement", path: str | os.PathLike | None = None,
                 max_chars: int | None = None):
        self.doc_type = doc_type
        self.max_chars = max_chars
        p = Path(path or os.environ.get("FOMC_DOCS_PATH", _DEFAULT_PATH))
        if not p.exists():
            raise FileNotFoundError(
                f"FOMC corpus not found at {p}. Point FOMC_DOCS_PATH at the processed "
                f"documents.jsonl (statements+minutes) from the watching-crowding-build repo."
            )
        docs: list[tuple[pd.Timestamp, str]] = []
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("doc_type") != doc_type:
                continue
            docs.append((pd.Timestamp(d["release_date"]), d["text"]))
        docs.sort(key=lambda x: x[0])
        self._release_dates = [r for r, _ in docs]
        self._texts = [t for _, t in docs]

    @property
    def count(self) -> int:
        return len(self._texts)

    def as_of(self, asof) -> str | None:
        """The most recent document with ``release_date <= asof``, or None if the
        date precedes the first document. Truncated to ``max_chars`` if set."""
        i = bisect.bisect_right(self._release_dates, pd.Timestamp(asof)) - 1
        if i < 0:
            return None
        text = self._texts[i]
        return text[: self.max_chars] if self.max_chars else text

    def pair_as_of(self, asof) -> tuple[str | None, str | None]:
        """``(current, previous)`` documents available as of ``asof``.

        Consecutive statements are ~0.80 similar — the language is heavily
        templated, so the information sits in what *changed*, not in the document.
        Serving the predecessor alongside the current one is what lets a selector
        show an analyst the edit rather than 400 words of recurring boilerplate.
        """
        i = bisect.bisect_right(self._release_dates, pd.Timestamp(asof)) - 1
        if i < 0:
            return None, None
        cut = (lambda t: t[: self.max_chars]) if self.max_chars else (lambda t: t)
        return cut(self._texts[i]), (cut(self._texts[i - 1]) if i >= 1 else None)
