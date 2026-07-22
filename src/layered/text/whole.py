"""Whole-document text — the control arm.

Reproduces the un-partitioned behaviour: every analyst receives the same document.
It is kept precisely so the cost of *not* partitioning stays measurable — this arm
is what produced the 0.221 → 0.339 correlation collapse, and a fix is only worth
claiming against the thing it fixed.

Dates are still scrubbed here. The control varies the partition, not the leak
surface, so both arms are stripped identically and only one thing differs.
"""
from __future__ import annotations

import pandas as pd

from src.layered.text.cue import strip_chrome
from src.layered.text.selector import TextContext, TextSelector, scrub_dates, sentences


class WholeDocumentSelector(TextSelector):
    """Serves the entire point-in-time document, ignoring cues."""

    def select(self, asof: pd.Timestamp, cues: list[str], driver: str = "") -> TextContext:
        current, _ = self.corpus.pair_as_of(asof)
        if current is None:
            return TextContext(driver=driver, doc_type=self.doc_type, available=False)
        body = strip_chrome(scrub_dates(current))
        return TextContext(
            driver=driver,
            doc_type=self.doc_type,
            available=True,
            unchanged=sentences(body),
        )
