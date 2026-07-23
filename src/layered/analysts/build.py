"""Shared wiring for the run scripts — build an analyst, preflight the model, audit.

The three runners (``run_analyst``, ``run_analyst_ic``, ``run_feature_ic``) all
stand up the same chain: an FOMC text selector, an ``LLMAnalyst`` from a persona,
and (for the scored runs) a live model checked once before a long loop. That wiring
lived copy-pasted in each script; it lives here once so a change to the build order,
the preflight message, or the audit format happens in a single place.

The ``anthropic`` import is deferred into ``preflight_llm`` so the no-model paths
(``run_feature_ic`` and ``--dry-run``) never require the package installed.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import yaml

from src.data.fomc_text import FomcCorpus
from src.layered.analysts.carry_forward import CarryForward
from src.layered.analysts.llm_analyst import PERSONA_DIR, LLMAnalyst
from src.layered.text import CueSelector, WholeDocumentSelector

# .../src/layered/analysts/build.py -> parents[3] == repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]


def build_selector(text_mode: str, text_doc: str = "statement",
                   text_max_chars: int | None = None, *,
                   corpus_path: str | os.PathLike | None = None,
                   verbose: bool = False):
    """The text channel: cue-partitioned, whole-document control, or none.

    ``corpus_path`` points at any documents.jsonl in the FOMC schema
    (``doc_type``/``release_date``/``text``); None keeps the FOMC default, so
    every pre-international persona behaves exactly as before.
    """
    if text_mode == "none":
        return None
    corpus = FomcCorpus(doc_type=text_doc, path=corpus_path,
                        max_chars=text_max_chars)
    cls = CueSelector if text_mode == "cue" else WholeDocumentSelector
    if verbose:
        src = Path(corpus_path).parent.name if corpus_path else "fomc"
        print(f"[info] {src} {text_doc}s loaded: {corpus.count} docs, "
              f"selector={text_mode} (point-in-time by release_date)",
              file=sys.stderr)
    return cls(corpus)


def persona_corpus_path(driver: str) -> Path | None:
    """The persona's declared ``text_corpus`` path (repo-root-relative), or
    None for the FOMC default. Read here rather than in ``from_persona`` so
    the analyst's own signature and the selector interface stay frozen."""
    spec = yaml.safe_load((PERSONA_DIR / f"{driver}.yaml").read_text()) or {}
    rel = spec.get("text_corpus")
    return (_REPO_ROOT / rel) if rel else None


def build_analyst(driver: str, llm, *, text_mode: str = "cue",
                  text_doc: str = "statement", text_max_chars: int | None = None,
                  describe_features: bool = False, verbose: bool = True) -> LLMAnalyst:
    """An ``LLMAnalyst`` wired from its persona + that persona's text channel."""
    selector = build_selector(text_mode, text_doc, text_max_chars,
                              corpus_path=persona_corpus_path(driver),
                              verbose=verbose)
    return LLMAnalyst.from_persona(driver, llm=llm, text_selector=selector,
                                   describe_features=describe_features)


def preflight_llm(model: str, *, max_tokens: int = 2000):
    """Construct and validate the client once, or exit with a clear message.

    A 120-250 word report plus its JSON scaffolding lands near 500-700 output
    tokens, so the default is 2000 — the client's own 1024 default truncates the
    tail often enough that the JSON fails to parse and the call is wastefully
    retried. ``anthropic`` is imported here, lazily, for the reason in the module
    docstring.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("[error] ANTHROPIC_API_KEY not set. Use --dry-run to inspect the prompt.")
        raise SystemExit(1)
    from src.llm.anthropic_client import AnthropicClient
    llm = AnthropicClient(model=model, max_tokens=max_tokens)
    try:
        llm.validate()
    except Exception as e:  # noqa: BLE001
        print(f"[error] LLM preflight failed — check ANTHROPIC_API_KEY and --model.\n"
              f"        {type(e).__name__}: {e}")
        raise SystemExit(1)
    return llm


def print_run_audit(llm, runner=None) -> None:
    """The per-run token/cost tally, plus carry-forward stats when it wrapped."""
    audit = {"llm": llm.usage_summary()}
    if isinstance(runner, CarryForward):
        audit["carry_forward"] = runner.stats
    print("\n## Run audit\n" + json.dumps(audit, indent=2))
