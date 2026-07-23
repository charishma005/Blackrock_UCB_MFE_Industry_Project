"""The pod's mandate, composed into system-prompt text.

Kept out of ``llm_pm.py`` so that *what a PM is asked to do* and *how a PM is wired*
are edited independently — the mandate is the part a research question changes, the
wiring is the part it must not. It is also why ``llm_pm._system_prompt`` takes a
one-line call rather than growing a formatter.

The analyst layer solves the same problem with ``mandate:`` as a bullet list in the
persona and nothing else; a PM needs more structure because it is answering several
distinct questions (how to weigh a stale view, a split panel, an uncovered driver)
that are easier to author, diff, and compare across pods as named blocks than as one
prose blob.

Two properties matter for comparability, and both are mechanical here:

  * **Fixed section order.** ``weighing`` is a mapping, so YAML gives no guaranteed
    key order across files. Rendering in a declared order means two pods' prompts
    differ only where their *text* differs — otherwise an A/B between two mandates
    would be partly an A/B between two orderings.
  * **Nothing is silently dropped.** A ``weighing`` key this module has never heard of
    is rendered after the known ones rather than ignored, so a mandate author can add
    a dimension without editing code. Silently discarding it would let someone write a
    mandate, read it back in the YAML, and never learn the model was not shown it.
"""
from __future__ import annotations

from typing import Iterable

# The order weighing keys are rendered in. Known keys first, in the order a PM
# actually confronts them: what is the evidence worth, is the panel agreed, what is
# missing, and only then am I entitled to depart from it.
_WEIGHING_ORDER = ("staleness", "disagreement", "gaps", "override")

_WEIGHING_HEADER = "How to weigh the panel:"
_TRADE_HEADER = "The trade:"


def _bullets(items: Iterable) -> list[str]:
    return [f"- {str(x).strip()}" for x in (items or []) if str(x).strip()]


def _weighing_block(weighing: dict) -> str:
    """The named weighing dimensions, known keys in declared order, then the rest."""
    if not weighing:
        return ""
    rest = [k for k in weighing if k not in _WEIGHING_ORDER]
    keys = [k for k in _WEIGHING_ORDER if k in weighing] + sorted(rest)
    lines = [f"- {k.replace('_', ' ')}: {str(weighing[k]).strip()}"
             for k in keys if str(weighing.get(k, "")).strip()]
    return "\n".join([_WEIGHING_HEADER, *lines]) if lines else ""


def _trade_block(trade: dict) -> str:
    """The trade mandate and its universe.

    The universe is stated in prose *as well as* being compiled into the tool's enum.
    The enum is what makes it binding; saying it here is what stops the model spending
    a tool call discovering the constraint by being rejected.
    """
    if not trade:
        return ""
    lines = _bullets(trade.get("mandate"))
    universe = [str(s) for s in (trade.get("universe") or [])]
    if universe:
        lines.append(f"- Legs may only name these instruments: {', '.join(universe)}.")
    max_legs = trade.get("max_legs")
    if max_legs:
        n = int(max_legs)
        lines.append(f"- Use at most {n} leg{'' if n == 1 else 's'}.")
    return "\n".join([_TRADE_HEADER, *lines]) if lines else ""


def render_mandate(config: dict) -> str:
    """The pod's mandate as system-prompt text.

    Falls back to a raw ``system:`` string for pods written before the structured
    blocks existed. No shipped pod relies on this any more — the one that did
    (``macro_rates``, an acknowledged placeholder) has been retired. It is kept because
    the fallback is what let the blocks land without a flag day, and an external pod
    directory (``build_pm(pod_dir=...)``) may still be written the old way; the two
    dict-level tests in ``test_pm_prompt_guardrails`` hold it in place.
    """
    config = config or {}
    blocks = [
        "\n".join(_bullets(config.get("mandate"))),
        _weighing_block(config.get("weighing") or {}),
        _trade_block(config.get("trade") or {}),
    ]
    body = "\n\n".join(b for b in blocks if b)
    # The fallback is decided by the MANDATE blocks alone. `display_name` is prepended
    # afterwards precisely so it cannot make an empty mandate look populated and
    # suppress the `system:` text a legacy pod depends on.
    if not body:
        body = str(config.get("system", "")).strip()
    name = str(config.get("display_name", "") or "").strip()
    return "\n\n".join(b for b in (f"You are the {name}." if name else "", body) if b)
