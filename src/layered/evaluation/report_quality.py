"""Grade the report, not just the number.

Every metric elsewhere (IC, calibration) scores the *header* — direction and
conviction. But the report prose is the actual contract to the PM, and until now it
was unevaluated. These are deterministic, offline checks over saved runs: they cost
nothing and catch the failures that matter for a report an LLM downstream will read.

Five checks, each a rate over the graded meetings:

  trade_naming        does the report name a trade? (a hard mandate violation —
                      "you never name a trade")
  cross_driver        does it reason about another driver's vocabulary? (soft,
                      lexical — flags drift, not proof of it)
  evidence_hallucin.  did it cite a feature that was not supplied? (grounding —
                      computed from the raw cited list before validation strips it)
  direction_consist.  does the prose lean the same way as the header direction?
                      (crude sentiment match)
  completeness        report non-empty and within length, falsifier present
  declared_gaps       did it say what it was never handed, and is the conviction
                      consistent with that? (a confident call resting on several
                      absent inputs is the pairing worth reviewing)

All checks are coarse by construction; they flag, they do not adjudicate. An
LLM-judge pass is the natural deeper layer, deliberately not done here (it adds a
judge and a cost, and can confound).

``conviction_response`` sits apart from these: it grades the *sequence* rather than
any one report, asking whether conviction moves after the analyst is proved wrong.
That question is unanswerable from a single meeting, which is why it lives at run
level rather than in ``evaluate_report``.
"""
from __future__ import annotations

import json

import pandas as pd

# On-topic vocabulary per driver — the cross-driver drift check reads a report as
# "off its own driver" when it leans on another driver's words. Coarse and lexical
# by design: it flags for review, it does not prove contamination.
_DRIVER_LEXICON: dict[str, list[str]] = {
    "inflation": ["inflation", "cpi", "price", "yoy", "disinflat", "deflation"],
    "labor_tightness": ["labor", "unemploy", "employ", "payroll", "jobs", "wage", "sahm"],
    "balance_sheet": ["balance sheet", "qt", "runoff", "reserve", "liquidity", "quantitative", "fed asset", "walcl"],
    "term_premium": ["term premium", "long end", "long-end", "10-year", "10y", "duration", "supply", "issuance"],
    "curve_slope": ["2s10s", "slope", "steepen", "flatten", "curve", "spread"],
    "inflation_expectations": ["breakeven", "break-even", "t10yie", "expectation", "expected inflation"],
    "financial_conditions": ["financial conditions", "nfci", "tighten", "eas", "credit", "conditions"],
}

# Report-specific trade lexicon. The legacy ``_TRADE_TERMS`` includes "position" and
# "curve", which false-positive badly on a report: "position" catches the feature
# `yoy_range_position` and ordinary English ("well-positioned"), "curve" catches the
# Phillips curve. Only unambiguous trade language counts as a mandate violation.
_TRADE_TERMS = ("flattener", "steepener", "2s10s", "overweight", "underweight",
                "go long", "go short", "long position", "short position",
                "spread trade", "curve trade", "buy the", "sell the", "long the", "short the")

_ACCEL = ("acceler", "rising", "rise", "higher", "pick up", "firm", "hot", "elevat", "increas", "climb")
_DECEL = ("deceler", "easing", "ease", "cooling", "cool", "falling", "fall", "lower", "soften", "slow", "declin", "moderat")


def _contains(text: str, terms) -> list[str]:
    low = text.lower()
    return [t for t in terms if t in low]


def evaluate_report(rec: dict, driver: str, feature_names: set[str]) -> dict:
    """One meeting's report → a dict of boolean/soft flags."""
    view = rec["view"]
    report = (view.get("report") or "").strip()
    direction = view.get("direction")

    # trade naming — hard violation
    trade_hits = _contains(report, _TRADE_TERMS)

    # cross-driver vocabulary — soft drift signal
    own = set(_DRIVER_LEXICON.get(driver, [driver]))
    foreign = [t for d, ws in _DRIVER_LEXICON.items() if d != driver for t in ws if t not in own]
    foreign_hits = _contains(report, foreign)

    # evidence hallucination — cited a *feature* that was never supplied.
    # Read the ORIGINAL cited list from the raw tool output, before validation strips it.
    try:
        raw_ke = json.loads(rec.get("raw_response") or "{}").get("key_evidence") or []
    except Exception:  # noqa: BLE001
        raw_ke = view.get("key_evidence") or []
    # Some models fill the array field with a single comma-joined string; iterating
    # that yields characters. Coerce to a real list before anything else.
    if isinstance(raw_ke, str):
        raw_ke = [s.strip() for s in raw_ke.split(",") if s.strip()]
    cited_raw = [str(c) for c in raw_ke]
    # Feature names are single tokens (no spaces); a multi-word citation is the model
    # referencing the TEXT channel in prose ("policy language change"), which is
    # legitimate evidence it was given — not a hallucinated feature. Only feature-like
    # tokens that don't exist count as hallucinations.
    feature_like = [c for c in cited_raw if " " not in c]
    text_cites = [c for c in cited_raw if " " in c]
    hallucinated = [c for c in feature_like if c not in feature_names]

    # direction consistency — does the prose lean with the header?
    a, d = len(_contains(report, _ACCEL)), len(_contains(report, _DECEL))
    prose_dir = "up" if a > d else "down" if d > a else "flat"
    consistent = (direction == "flat") or (prose_dir == "flat") or (prose_dir == direction)

    # declared gaps — what the analyst says it was never handed. Deliberately read from
    # the structured field and never from the prose: naming another driver here is the
    # sanctioned way to flag a dependency, so it must not feed the cross-driver check
    # above (which scans `report` only, and so already excludes this by construction).
    gaps = view.get("missing_inputs") or []
    n_missing = len(gaps)
    try:
        conv = float(view.get("conviction", 0.0))
    except (TypeError, ValueError):
        conv = 0.0
    # A call resting on several absent inputs should not be a confident call. Coarse,
    # like everything else here — it flags the pairing for review, it does not adjudicate.
    overconfident = conv >= 0.6 and n_missing >= 2

    return {
        "names_trade": bool(trade_hits),
        "trade_hits": trade_hits,
        "cross_driver": bool(foreign_hits),
        "n_foreign": len(foreign_hits),
        "hallucinated_evidence": bool(hallucinated),
        "cites_text": bool(text_cites),
        "n_cited": len(cited_raw),
        "dir_consistent": bool(consistent),
        "report_words": len(report.split()),
        "has_falsifier": bool((view.get("falsifier") or "").strip()),
        "declares_gaps": bool(n_missing),
        "n_missing": n_missing,
        "overconfident_given_gaps": bool(overconfident),
        "empty": not report,
        "degraded": bool(view.get("degraded")),
    }


def evaluate_run(path: str, driver: str = "inflation") -> dict:
    """Aggregate report-quality rates over one run's JSONL (graded rows only)."""
    recs = [json.loads(l) for l in open(path)]
    feature_names: set[str] = set()
    for r in recs:
        f = r.get("features", {})
        feature_names |= set(f.get("series", {})) | set(f.get("scalars", {}))
    rows = [evaluate_report(r, driver, feature_names) for r in recs if not r["view"].get("degraded")]
    n = len(rows)
    if not n:
        return {"run": path.split("/")[-1], "n": 0}
    frac = lambda k: round(sum(r[k] for r in rows) / n, 3)  # noqa: E731
    return {
        "run": path.split("/")[-1].replace(".jsonl", ""),
        "n": n,
        "names_trade": frac("names_trade"),
        "cross_driver": frac("cross_driver"),
        "hallucinated": frac("hallucinated_evidence"),
        "cites_text": frac("cites_text"),
        "dir_consistent": frac("dir_consistent"),
        "has_falsifier": frac("has_falsifier"),
        "declares_gaps": frac("declares_gaps"),
        "overconfident_given_gaps": frac("overconfident_given_gaps"),
        "med_words": int(pd.Series([r["report_words"] for r in rows]).median()),
    }


def compare_runs(paths: list[str], driver: str = "inflation") -> pd.DataFrame:
    return pd.DataFrame([evaluate_run(p, driver) for p in paths]).set_index("run")


def conviction_response(path: str) -> dict:
    """Does the analyst update after it is wrong?

    ``has_falsifier`` only asks whether a falsifier was written. This asks the
    question that actually matters and that no single report can answer: having been
    wrong at the previous release, does the next call come back softer or flipped —
    or at the same conviction, as though nothing happened?

    It needs no judge and no new data. Both halves are already in the run file: the
    previous view's direction, and the realized move, which is the change in ``level``
    between consecutive records — the same quantity ``ICEvaluator`` grades against
    (``level.shift(-1) - level``).

    Degraded and carried rows are dropped first, matching how the run is scored;
    a carried view is the previous view re-emitted, so its conviction change is zero
    by construction and would dilute the statistic rather than inform it. Pairs are
    then adjacent in the surviving sequence, exactly as in ``ICEvaluator``.

    Flat calls are excluded: "will be essentially unchanged" has no sign to grade
    against a continuous move without an arbitrary threshold for what counts as flat.
    """
    with open(path) as fh:
        recs = [json.loads(line) for line in fh if line.strip()]
    views = [r["view"] for r in recs
             if not r["view"].get("degraded") and not r["view"].get("carried")]

    rows = []
    for prev, cur in zip(views, views[1:]):
        if prev.get("level") is None or cur.get("level") is None:
            continue
        if prev.get("direction") not in ("up", "down"):
            continue
        move = float(cur["level"]) - float(prev["level"])
        if move == 0.0:
            continue
        rows.append({
            "right": (move > 0) == (prev["direction"] == "up"),
            "d_conv": float(cur.get("conviction", 0.0)) - float(prev.get("conviction", 0.0)),
            "flipped": cur.get("direction") != prev.get("direction"),
        })

    if not rows:
        return {"run": path.split("/")[-1].replace(".jsonl", ""), "n_transitions": 0}

    right = [r for r in rows if r["right"]]
    wrong = [r for r in rows if not r["right"]]
    mean = lambda xs, k: round(sum(x[k] for x in xs) / len(xs), 3) if xs else None  # noqa: E731
    return {
        "run": path.split("/")[-1].replace(".jsonl", ""),
        "n_transitions": len(rows),
        "n_right": len(right),
        "n_wrong": len(wrong),
        # The discriminating pair. A calibrated analyst softens after a miss and holds
        # after a hit, so d_conv_after_wrong should sit clearly below d_conv_after_right.
        "d_conv_after_right": mean(right, "d_conv"),
        "d_conv_after_wrong": mean(wrong, "d_conv"),
        "flip_rate_after_right": mean(right, "flipped"),
        "flip_rate_after_wrong": mean(wrong, "flipped"),
    }
