"""Meaning-preserving prompt perturbations (Homo Silicus C).

These change the exact bytes of the assembled user prompt without changing what it
says, so a reasoning model's answer should be invariant and the spread of IC across
them measures fragility, not skill. They act on the final prompt string (``apply_prompt``)
rather than on a ``FeatureSet``/``TextContext``, because scaffolding and whitespace live
in the rendered template, not in the measurement objects. Being string-only and
PM-free, they are shared by both the analyst and PM registries.
"""
from __future__ import annotations

import re

from src.layered.perturb.base import Perturbation


class WhitespaceVariant(Perturbation):
    """Insert an extra blank line between blocks. Pure surface noise — a reasoning
    model ignores it; a model keyed to an exact template layout may not."""

    name = "whitespace"

    def apply_prompt(self, prompt: str) -> str:
        return prompt.replace("\n\n", "\n\n\n")


# Surface synonyms for fixed scaffolding the renderer emits. Each pair is
# meaning-preserving: it renames a label or connector the model was never asked to key
# on. Deliberately narrow — nothing here touches a measurement value or a policy
# sentence, only the boilerplate around them.
_SWAPS: tuple[tuple[str, str], ...] = (
    (r"last (\d+) observations, oldest → newest", r"most recent \1 readings, oldest to newest"),
    (r"\bDerived measurements\b", "Derived readings"),
    (r"^Driver: ", "Driver — "),
)


class RewordScaffolding(Perturbation):
    """Swap a few fixed scaffolding phrases for synonyms. Same instruction, different
    wording — the "prompt permutation" Homo Silicus §4.5 prescribes as an
    anti-prompt-hacking robustness check."""

    name = "reword_scaffold"

    def apply_prompt(self, prompt: str) -> str:
        out = prompt
        for pat, repl in _SWAPS:
            out = re.sub(pat, repl, out, flags=re.MULTILINE)
        return out
