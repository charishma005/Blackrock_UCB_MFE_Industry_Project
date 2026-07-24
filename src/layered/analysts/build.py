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
from src.data.nowcast_news import NowcastNewsCorpus
from src.layered.analysts.carry_forward import CarryForward
from src.layered.analysts.llm_analyst import PERSONA_DIR, LLMAnalyst
from src.layered.text import CueSelector, NowcastNewsSelector, WholeDocumentSelector

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


def build_news_selector(use_news: bool, *, news_path: str | os.PathLike | None = None,
                        news_weeks: int = 3, verbose: bool = False):
    """The shared market-nowcast channel: one selector, reused by every analyst.

    ``use_news`` is the on/off switch this exists for — False (the default) returns
    None and every downstream call site treats that as "no news channel", so the
    unopted-in path is unchanged. True builds one corpus and one selector, shared
    across every ``build_analyst`` call in the same run rather than reloaded per
    driver, since the window it serves does not vary by driver.
    """
    if not use_news:
        return None
    corpus = NowcastNewsCorpus(path=news_path, weeks=news_weeks)
    if verbose:
        print(f"[info] nowcast news loaded: {corpus.count} weekly entries, "
              f"window={news_weeks}w (point-in-time by week)", file=sys.stderr)
    return NowcastNewsSelector(corpus)


def persona_corpus_path(driver: str) -> Path | None:
    """The persona's declared ``text_corpus`` path (repo-root-relative), or
    None for the FOMC default. Read here rather than in ``from_persona`` so
    the analyst's own signature and the selector interface stay frozen."""
    spec = yaml.safe_load((PERSONA_DIR / f"{driver}.yaml").read_text()) or {}
    rel = spec.get("text_corpus")
    return (_REPO_ROOT / rel) if rel else None


def build_analyst(driver: str, llm, *, text_mode: str = "cue",
                  text_doc: str = "statement", text_max_chars: int | None = None,
                  describe_features: bool = False, use_memory: bool = False,
                  use_news: bool = False, news_selector=None,
                  news_path: str | os.PathLike | None = None, news_weeks: int = 3,
                  perturbation=None, verbose: bool = True) -> LLMAnalyst:
    """An ``LLMAnalyst`` wired from its persona + that persona's text channel.

    ``perturbation`` is an evaluation-only leak/robustness arm (``src.layered.perturb``);
    ``None`` is the shipped path. The run script resolves the ``--perturb`` name to a
    ``Perturbation`` and passes it through here. The text corpus comes from the
    persona's ``text_corpus`` field (per-bank corpora for the international
    analysts), falling back to the FOMC default.

    ``use_news`` is the on/off switch for the shared market-nowcast channel
    (default False — unopted-in runs reproduce the prompt exactly as before). Pass
    an already-built ``news_selector`` when calling this once per driver in a loop,
    so the nowcast file is parsed once for the whole population rather than once
    per analyst; leave it None for a single ad-hoc build and it is constructed here.
    """
    selector = build_selector(text_mode, text_doc, text_max_chars,
                              corpus_path=persona_corpus_path(driver),
                              verbose=verbose)
    if use_news and news_selector is None:
        news_selector = build_news_selector(use_news, news_path=news_path,
                                            news_weeks=news_weeks, verbose=verbose)
    return LLMAnalyst.from_persona(driver, llm=llm, text_selector=selector,
                                   describe_features=describe_features,
                                   use_memory=use_memory,
                                   news_selector=news_selector, use_news=use_news,
                                   perturbation=perturbation)


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
