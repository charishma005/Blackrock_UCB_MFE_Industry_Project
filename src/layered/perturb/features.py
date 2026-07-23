"""Perturbations of the analyst's measurement block (Canayaz A + Homo Silicus C).

Two families, both operating on a ``FeatureSet`` between ``FeatureEngine.compute`` and
``FeatureSet.render`` (the clean compute→render seam):

  * **Leak probes** (Canayaz's "unlearning" test) alter the *content* so a memorised
    answer becomes wrong. A model reasoning from the evidence follows the altered
    arithmetic; one reciting a remembered outcome does not. ``perturbation_bench``
    measures which happened.
  * **Meaning-preserving** (Homo Silicus's robustness battery) changes only the render
    order, holding the content, so a reasoning model is invariant and a
    prompt-order-sensitive one is not.

Every transform returns a ``model_copy`` — the original ``FeatureSet`` is never mutated,
so the run script can still grade the model's answer against a clean, unperturbed
outcome.
"""
from __future__ import annotations

from src.layered.perturb.base import Perturbation

# A change/momentum measurement, by name — a feature whose *sign* encodes a direction
# of movement (rising/falling), as opposed to a level, range, or position. The closed op
# vocabulary names these consistently, so a token match finds them without a per-persona
# list; the tokens below are validated against every shipped persona's feature namespace
# (``tests/test_perturb.py``), which is what keeps this safe as personas are added.
#
#   * ``change`` — every ``*_change_*`` (op ``diff``) and every ``pct_change`` feature.
#   * ``mom``    — the monthly-momentum family (``headline_mom``, ``outgoing_mom``,
#                  ``headline_mom_3m_avg``, ``mom_gap_vs_outgoing``).
#   * ``annualized`` — the annualized short-horizon rates (``headline_3m_annualized``),
#                  which are rate-of-change readings and were silently missed before.
#
# Deliberately NOT tokens: ``yoy`` — a year-over-year *rate* (``headline_cpi_yoy``) is a
# level, not a change, so negating it would corrupt the level. And ``gap`` — it looks
# momentum-ish but ``sahm_gap`` is a *level-space* spread (unemployment above its 12m
# low), so matching it would flip a level while claiming to hold levels fixed; its one
# genuine target, ``mom_gap_vs_outgoing``, is already caught by ``mom``.
_CHANGE_TOKENS = ("change", "mom", "annualized")


def _is_change(name: str) -> bool:
    n = name.lower()
    return any(tok in n for tok in _CHANGE_TOKENS)


class RescaleFeatures(Perturbation):
    """Multiply every measurement by ``k``. Magnitudes move; signs and ordering do not,
    so a reasoning analyst's *direction* should be invariant while its conviction may
    scale. Primarily a sizing/robustness probe rather than a direction one."""

    def __init__(self, k: float = 2.0):
        self.k = float(k)
        self.name = f"rescale_{k:g}x"

    def apply_features(self, features):
        series = [f.model_copy(update={"values": [v * self.k for v in f.values]})
                  for f in features.series]
        scalars = [f.model_copy(update={"value": f.value * self.k})
                   for f in features.scalars]
        return features.model_copy(update={"series": series, "scalars": scalars})


class ShiftLevel(Perturbation):
    """Add a constant to the level feature only — a counterfactual shift in where the
    driver sits, holding its dynamics. Tests whether a level/mean-reversion reading
    tracks the number in front of it or a remembered regime. A no-op if the persona
    declares no ``level_feature``."""

    def __init__(self, delta: float = 1.0):
        self.delta = float(delta)
        self.name = f"shift_level_{delta:+g}"

    def apply_features(self, features):
        lvl = features.level_feature
        if lvl is None:
            return features
        series = [(f.model_copy(update={"values": [v + self.delta for v in f.values]})
                   if f.name == lvl else f) for f in features.series]
        scalars = [(f.model_copy(update={"value": f.value + self.delta})
                    if f.name == lvl else f) for f in features.scalars]
        return features.model_copy(update={"series": series, "scalars": scalars})


class SignFlipMomentum(Perturbation):
    """Negate every change/momentum measurement, holding the levels. Recent dynamics
    reverse while the level path is untouched, so a model reading momentum should flip
    its call and one anchored on a remembered outcome should not. A no-op if the persona
    exposes no change-like feature."""

    name = "signflip_momentum"

    def apply_features(self, features):
        lvl = features.level_feature
        # Never flip the level itself, even if its name matched — the whole point is to
        # reverse the dynamics while holding the level path fixed.
        def flip(f):
            return f.name != lvl and _is_change(f.name)
        series = [(f.model_copy(update={"values": [-v for v in f.values]})
                   if flip(f) else f) for f in features.series]
        scalars = [(f.model_copy(update={"value": -f.value})
                    if flip(f) else f) for f in features.scalars]
        return features.model_copy(update={"series": series, "scalars": scalars})


class CounterfactualPath(Perturbation):
    """Reverse every trajectory (oldest ↔ newest). Same values, opposite recent
    direction — an "unlearning" probe: a memorised path cannot be read backwards, so a
    model that flips its call is reading the evidence, not reciting the era. Note this
    also moves the graded level (the last value becomes the old first one), so grade a
    counterfactual run against its *own* reversed outcome.

    Reversing the path negates the sign of every rate-of-change reading, so the derived
    momentum *scalars* are flipped with it — otherwise the block contradicts itself (a
    reversed series showing falling inflation beside a ``+0.5`` momentum scalar), which
    muddies whether a flip is reasoning or recall. Level/range/position scalars are left
    as-is: this post-compute seam cannot re-derive them from reversed inputs, so a
    counterfactual run is a directional probe over the trajectories and their momentum,
    not a fully recomputed history — a real limit, stated rather than hidden."""

    name = "counterfactual_path"

    def apply_features(self, features):
        series = [f.model_copy(update={"values": list(reversed(f.values))})
                  for f in features.series]
        scalars = [(f.model_copy(update={"value": -f.value}) if _is_change(f.name) else f)
                   for f in features.scalars]
        return features.model_copy(update={"series": series, "scalars": scalars})


class ReorderFeatureLines(Perturbation):
    """Meaning-preserving: reverse the order features are rendered in. The content is
    byte-identical; only its position in the prompt changes. A reasoning model is
    invariant (Homo Silicus battery); a template-sensitive one is not. ``FeatureSet.level``
    resolves by name, so reversing the lists does not move the graded level."""

    name = "reorder_features"

    def apply_features(self, features):
        return features.model_copy(update={
            "series": list(reversed(features.series)),
            "scalars": list(reversed(features.scalars)),
        })
