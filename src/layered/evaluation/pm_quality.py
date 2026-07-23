"""Grade the PM's prose, not just its numbers.

A deliberate sibling of ``report_quality`` rather than a reuse of it, because two of
that module's central checks **invert** at this layer:

  * ``names_trade`` is a mandate *violation* for an analyst ("you never name a trade —
    expressing a view as a position is someone else's job") and is the PM's actual
    job. Here it is recorded, not penalised.
  * ``cross_driver`` measures *drift* for an analyst, who is supposed to stay on its
    own driver. A PM speaking across drivers is doing the thing the layer exists for,
    so the same measurement is inverted into ``n_drivers_named`` — coverage, where
    more is better.

Importing ``evaluate_report`` and reinterpreting its output would leave those two
inversions implicit in whoever reads the table. ``_DRIVER_LEXICON`` *is* imported —
it is data about vocabulary, not a mandate, and duplicating it would let the two
layers drift apart.

The genuinely portable checks (length, emptiness, direction consistency) are
re-expressed here against the PM's shape, which has N directions rather than one.
"""
from __future__ import annotations

import json
from typing import Optional

import pandas as pd

from src.layered.evaluation.report_quality import (
    _ACCEL,
    _DECEL,
    _DRIVER_LEXICON,
    _TRADE_TERMS,
    _contains,
)


def evaluate_arbitration(rec: dict) -> dict:
    """One meeting's arbitration → a dict of flags."""
    av = rec.get("arbitrated") or {}
    notes = (av.get("notes") or "").strip()
    drivers: dict = av.get("drivers") or {}
    board: dict = rec.get("board") or {}
    present = {d for d, b in board.items() if (b or {}).get("present")}

    # Coverage of the panel in prose: how many of the drivers it heard from does the
    # PM actually engage with? Low coverage with high conviction is the shape of a PM
    # that read one report and extrapolated.
    named = {d for d, words in _DRIVER_LEXICON.items() if _contains(notes, words)}

    # Grounding: a driver scored or discussed that was not on the board this meeting.
    # The parse path already drops these from `drivers`, so a hit here means the prose
    # went somewhere the numbers could not.
    ungrounded = sorted(named - present) if present else []

    # Did the PM move the panel, and did it say why? An override the prose never
    # mentions is the least defensible thing this layer can produce.
    overrides, explained = [], []
    for d, v in drivers.items():
        b = board.get(d) or {}
        if not b.get("present"):
            continue
        analyst = _signed(b)
        if analyst is None:
            continue
        if (v > 0) != (analyst > 0) and v != 0 and analyst != 0:
            overrides.append(d)
            if d in named:
                explained.append(d)

    a, dn = len(_contains(notes, _ACCEL)), len(_contains(notes, _DECEL))

    return {
        "n_drivers_scored": len(drivers),
        "n_drivers_named": len(named),
        "coverage_prose": len(named & present) / len(present) if present else 0.0,
        "ungrounded_driver": bool(ungrounded),
        "ungrounded": ungrounded,
        "n_overrides": len(overrides),
        "override_explained": (len(explained) / len(overrides)) if overrides else 1.0,
        "names_trade": bool(_contains(notes, _TRADE_TERMS)),   # informational, not a fault
        "notes_words": len(notes.split()),
        "empty": not notes,
        "prose_lean": "up" if a > dn else "down" if dn > a else "flat",
        "degraded": bool(rec.get("degraded")),
    }


def _signed(board_entry: dict) -> Optional[float]:
    """The analyst's signed conviction, from the board snapshot in the record."""
    direction, conviction = board_entry.get("direction"), board_entry.get("conviction")
    if direction is None or conviction is None:
        return None
    return {"up": 1.0, "down": -1.0, "flat": 0.0}.get(direction, 0.0) * float(conviction)


def evaluate_pm_run(path: str) -> dict:
    """Aggregate over one PM run's JSONL (graded rows only)."""
    with open(path) as fh:
        recs = [json.loads(line) for line in fh if line.strip()]
    graded = [r for r in recs if not r.get("degraded")]
    if not graded:
        return {"run": path, "n": 0}

    rows = [evaluate_arbitration(r) for r in graded]
    df = pd.DataFrame(rows)
    out = {"run": path.rsplit("/", 1)[-1][:-6], "n": len(df),
           "degraded_rate": 1.0 - len(graded) / len(recs)}
    for col in ("n_drivers_scored", "n_drivers_named", "coverage_prose",
                "n_overrides", "override_explained", "notes_words"):
        out[col] = float(df[col].mean())
    for col in ("ungrounded_driver", "names_trade", "empty"):
        out[col + "_rate"] = float(df[col].mean())
    return out
