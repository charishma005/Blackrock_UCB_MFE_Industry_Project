"""Text layer — the context behind the measurements, partitioned by driver.

    selector.py  TextSelector interface, TextContext, date scrubbing
    cue.py       CueSelector — driver-specific passages + diff vs the previous doc
    whole.py     WholeDocumentSelector — the un-partitioned control arm
"""
from src.layered.text.cue import CueSelector
from src.layered.text.selector import TextContext, TextSelector, scrub_dates
from src.layered.text.whole import WholeDocumentSelector

__all__ = ["CueSelector", "TextContext", "TextSelector", "WholeDocumentSelector", "scrub_dates"]
