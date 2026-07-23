"""Prompt-perturbation arms for the leak and robustness tests.

The analyst-safe surface: the base class, the identity no-op, the concrete feature and
string perturbations, and ``analyst_perturbation`` to resolve a ``--perturb`` name. It
deliberately does NOT import ``brief`` (the meeting/scramble perturbation), because that
module imports the PM board and the analyst layer must not depend on the PM layer. The
PM run script reaches the scramble arm through ``src.layered.perturb.brief`` directly.
"""
from __future__ import annotations

from typing import Optional

from src.layered.perturb.base import IDENTITY, Perturbation
from src.layered.perturb.features import (CounterfactualPath, RescaleFeatures,
                                          ReorderFeatureLines, ShiftLevel,
                                          SignFlipMomentum)
from src.layered.perturb.text import RewordScaffolding, WhitespaceVariant

# name → factory. Parameterised perturbations use a sensible default under the CLI; the
# classes are importable for programmatic use (notebooks, tests) with explicit params.
_ANALYST = {
    "rescale": RescaleFeatures,
    "shift_level": ShiftLevel,
    "signflip_momentum": SignFlipMomentum,
    "counterfactual_path": CounterfactualPath,
    "reorder_features": ReorderFeatureLines,
    "whitespace": WhitespaceVariant,
    "reword_scaffold": RewordScaffolding,
}

#: Names accepted by ``--perturb`` on the analyst run scripts.
ANALYST_NAMES = sorted(_ANALYST)


def analyst_perturbation(name: Optional[str]) -> Optional[Perturbation]:
    """Resolve a ``--perturb`` name to an analyst-side perturbation, or ``None`` for the
    unperturbed arm. Raises on an unknown name rather than silently running clean."""
    if name is None:
        return None
    try:
        return _ANALYST[name]()
    except KeyError:
        raise ValueError(f"unknown analyst perturbation {name!r}; have {ANALYST_NAMES}")


__all__ = [
    "Perturbation", "IDENTITY",
    "RescaleFeatures", "ShiftLevel", "SignFlipMomentum", "CounterfactualPath",
    "ReorderFeatureLines", "WhitespaceVariant", "RewordScaffolding",
    "analyst_perturbation", "ANALYST_NAMES",
]
