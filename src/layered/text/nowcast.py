"""Shared market-nowcast news — the cross-driver macro channel.

Every other selector in this package partitions text *by driver*: cues route each
sentence to the one analyst it belongs to. This one does the opposite on purpose.
The nowcast digest is a weekly cross-asset read (Fed policy, inflation, growth,
equities, bonds, credit, geopolitics) and the axis worth filtering on is time, not
topic — "for a given date, only that week's and the two before it" — so every
analyst that opts in sees the same three-week window, cues argument ignored.

Dates are still an absolute-date leak surface here, arguably more than in the FOMC
text: a run of entries in calendar order is itself a calendar. So entries are
labelled by recency ("this week" / "1 week ago" / "2 weeks ago"), never by their
key, and every narrative/summary string is passed through ``scrub_dates`` besides,
in case cleaning left a stray month or year inside the prose.
"""
from __future__ import annotations

import pandas as pd

from src.layered.text.selector import TextContext, TextSelector, scrub_dates


def _label(n_back: int) -> str:
    if n_back == 0:
        return "this week"
    if n_back == 1:
        return "1 week ago"
    return f"{n_back} weeks ago"


def _render_entry(entry: dict) -> list[str]:
    """One week's digest as a few plain-language lines: the summary, then each
    topic's narrative with its sentiment/magnitude reading. ``file_leak_risk`` is
    a data-quality field from the cleaning pipeline, not market content, and is
    dropped here."""
    lines: list[str] = []
    summary = entry.get("summary")
    if summary:
        lines.append(scrub_dates(str(summary)))
    for topic, v in entry.items():
        if topic in ("summary", "file_leak_risk") or not isinstance(v, dict):
            continue
        narrative = v.get("narrative", "")
        sentiment = v.get("sentiment")
        magnitude = v.get("magnitude")
        reading = f" (sentiment {sentiment:+.2f}, magnitude {magnitude})" \
            if sentiment is not None and magnitude is not None else ""
        label = topic.replace("_", " ")
        lines.append(f"  {label}: {scrub_dates(str(narrative))}{reading}")
    return lines


class NowcastNewsSelector(TextSelector):
    """Serves the shared 3-week nowcast window, unpartitioned by driver."""

    def select(self, asof: pd.Timestamp, cues: list[str], driver: str = "") -> TextContext:
        window = self.corpus.window_as_of(asof)
        if not window:
            return TextContext(driver=driver, doc_type=self.doc_type, available=False)

        n = len(window)
        blocks: list[str] = []
        for i, (_, entry) in enumerate(window):
            n_back = (n - 1) - i  # last entry (i == n-1) is "this week"
            block_lines = [f"[{_label(n_back)}]"] + _render_entry(entry)
            blocks.append("\n".join(block_lines))

        return TextContext(
            driver=driver,
            doc_type=self.doc_type,
            available=True,
            unchanged=blocks,
        )
