"""The PM-side perturbation: scramble which report sits under which driver (Han et al.).

This is the only perturbation module that imports the PM layer (``pm.board.Meeting``),
so it is reached only from the PM run script — the analyst layer never touches it, which
keeps the data→analyst→PM dependency one-way.

``ScrambleReports`` implements the prior-vs-evidence probe from the causal-agent paper:
each present driver's *view* is rotated into the next slot, so the driver header (the
label the model's prior attaches to) disagrees with the report body (the evidence). A PM
that reasons from the evidence answers for the rotated call; one that answers from the
label is reciting a prior. ``perturbation_bench`` scores which.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

from src.layered.perturb.base import Perturbation
# Shared, PM-free string perturbations — offered on the PM registry too so the same
# meaning-preserving battery can be run one layer up.
from src.layered.perturb.text import RewordScaffolding, WhitespaceVariant
from src.layered.pm.board import Meeting


class ScrambleReports(Perturbation):
    """Rotate each present driver's view into another driver's slot.

    Only present drivers are rotated (an absent driver has no view to move). The
    rotation is a derangement for two or more present drivers — no driver keeps its own
    report — and is deterministic (a fixed ``offset``, no RNG), so a scrambled run
    reproduces exactly. The driver *keys* are untouched, so grounding, coverage, and the
    submit enum still see the real driver set; only the evidence bound to each label
    moves. The stored ``disagreement`` is computed on the rotated views and is therefore
    not meaningful for a scrambled run — it is not graded.
    """

    def __init__(self, offset: int = 1):
        self.offset = int(offset)
        self.name = f"scramble_reports_{offset}"

    def apply_meeting(self, meeting: Meeting) -> Meeting:
        present = meeting.present
        if len(present) < 2:
            return meeting
        views = [meeting.entries[d].view for d in present]
        k = self.offset % len(present)
        if k == 0:                       # a full turn is the identity — force a real shift
            k = 1
        rotated = views[k:] + views[:k]
        entries = dict(meeting.entries)
        for driver, view in zip(present, rotated):
            # Replace the whole view, keeping the slot's driver key (the header/label)
            # and its age/staleness context. Header says `driver`, evidence is `view`.
            entries[driver] = dataclasses.replace(entries[driver], view=view)
        return dataclasses.replace(meeting, entries=entries)


_PM = {
    "scramble_reports": ScrambleReports,
    "whitespace": WhitespaceVariant,
    "reword_scaffold": RewordScaffolding,
}

#: Names accepted by ``--perturb`` on the PM run script.
PM_NAMES = sorted(_PM)


def pm_perturbation(name: Optional[str]) -> Optional[Perturbation]:
    """Resolve a ``--perturb`` name to a PM-side perturbation, or ``None`` for the
    unperturbed arm. Raises on an unknown name rather than silently running clean."""
    if name is None:
        return None
    try:
        return _PM[name]()
    except KeyError:
        raise ValueError(f"unknown PM perturbation {name!r}; have {PM_NAMES}")
