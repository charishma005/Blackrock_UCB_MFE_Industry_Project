"""Controlled prompt perturbations — the shared mechanism behind the leak and
robustness arms (Tier-1 modifications A/B/C).

A ``Perturbation`` transforms a render-time object — a ``FeatureSet``, a
``TextContext``, or a PM ``Meeting`` — or the assembled prompt string, in the gap
between ``build_inputs`` and the model call. It is applied only when an explicit
``--perturb`` arm selects it, recorded in the run's ``.meta.json`` config, and (for the
analyst) registered in ``board.IDENTITY_KEYS`` so a board can never silently mix
perturbed and clean legs. Nothing here reaches a shipped run: these are evaluation arms,
held to the honesty rule that a diagnostic must never inform a production prompt
(``docs/analyst-layer.md`` §6).

**Reproducibility.** Every transform is deterministic — no RNG, no temperature (the
client exposes neither). A real perturbation run still carries the model's own sampling
noise; the offline tests drive deterministic stub clients and are exact.

**Layering.** The analyst layer must not import the PM layer (the boundary
``pm/build.py`` guards). So the feature/text/string perturbations live in ``features.py``
and ``text.py`` and are resolved by ``analyst_perturbation`` (no PM dependency), while
the meeting perturbation lives in ``brief.py`` and is resolved by ``pm_perturbation``,
which imports the PM board and is therefore reached only from the PM run script. This
module — the base class — imports neither, so both sides can depend on it.
"""
from __future__ import annotations


class Perturbation:
    """A no-op by default; a concrete perturbation overrides only what it touches.

    Every hook returns its input unchanged, so a perturbation that rewrites features
    leaves text, meetings, and the assembled prompt exactly as they were — which is what
    keeps each arm a one-variable change against its baseline. The hooks are duck-typed
    on purpose: ``LLMAnalyst``/``LLMPM`` hold a perturbation without importing this
    package, so the object only has to answer to these method names.
    """

    #: A short, stable label. Recorded in the run config and used to key comparisons.
    name = "identity"

    def fingerprint(self) -> str:
        """A one-line identity for the audit trail. Defaults to ``name``; a
        parameterised perturbation folds its parameters into ``name`` at construction."""
        return self.name

    # ── analyst-side hooks ───────────────────────────────────────────────────
    def apply_features(self, features):
        """Transform the measurement block (a ``FeatureSet``). Return a copy — never
        mutate — so the caller's original is intact for scoring against a clean outcome."""
        return features

    def apply_text(self, text):
        """Transform the policy-language block (a ``TextContext``)."""
        return text

    # ── PM-side hook ─────────────────────────────────────────────────────────
    def apply_meeting(self, meeting):
        """Transform the panel (a PM ``Meeting``) before it is rendered into a brief."""
        return meeting

    # ── shared string hook ───────────────────────────────────────────────────
    def apply_prompt(self, prompt: str) -> str:
        """Transform the assembled user prompt after rendering. Used by the
        meaning-preserving battery (whitespace, scaffolding rewording), whose whole
        point is to change bytes without changing meaning."""
        return prompt


#: The no-op, shared so a caller can substitute it for ``None`` when it prefers not to
#: branch. Stateless, so a single instance is safe to reuse everywhere.
IDENTITY = Perturbation()
