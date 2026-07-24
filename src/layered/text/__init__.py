"""Text layer — the context behind the measurements, partitioned by driver.

    selector.py  TextSelector interface, TextContext, date scrubbing
    cue.py       CueSelector — driver-specific passages + diff vs the previous doc
    whole.py     WholeDocumentSelector — the un-partitioned control arm
    nowcast.py   NowcastNewsSelector — shared cross-asset weekly news, not
                 partitioned by driver (every analyst opting in sees the same window)
"""
from src.layered.text.cue import CueSelector
from src.layered.text.nowcast import NowcastNewsSelector
from src.layered.text.selector import TextContext, TextSelector, scrub_dates
from src.layered.text.whole import WholeDocumentSelector

__all__ = [
    "CueSelector",
    "NowcastNewsSelector",
    "TextContext",
    "TextSelector",
    "WholeDocumentSelector",
    "scrub_dates",
]
