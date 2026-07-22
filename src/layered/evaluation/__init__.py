"""Evaluation — scoring any signal against what the driver actually did.

Deliberately separate from the analyst and from the legacy ``diagnostics.py``,
because it has to grade three different kinds of thing across this project's life —
raw features now, analyst views next, benchmark rules later — and none of them
should have to know about the others. Everything here consumes a signal indexed by
release date, so a feature, a rule, and an LLM's signed conviction are scored by
identical code.

    panel.py  replay a feature spec across history into a (date × feature) matrix
    ic.py     rank IC against the driver's move over the next N releases
    runs.py   load a saved analyst run into scoreable signed/level series
"""
from src.layered.evaluation.ic import ICEvaluator, ICResult, required_ic
from src.layered.evaluation.panel import FeaturePanel, release_dates
from src.layered.evaluation.runs import Run, discover_runs, load_run

__all__ = ["ICEvaluator", "ICResult", "FeaturePanel", "release_dates", "required_ic",
           "Run", "load_run", "discover_runs"]
