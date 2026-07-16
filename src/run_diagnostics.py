"""Phase-1 diagnostics report — deterministic vs LLM analysts, side by side.

Grades the analyst layer on the four things that matter before trusting it:
faithfulness, correctness, lookahead, and cross-agent correlation. Each is shown
for the deterministic Phase-1 agents and (if ANTHROPIC_API_KEY is set) the
LLM-refined agents, so you can see what the LLM adds.

Run:
  python3 -m src.run_diagnostics                       # synthetic, deterministic only
  python3 -m src.run_diagnostics --regime dovish
  ANTHROPIC_API_KEY=... python3 -m src.run_diagnostics # adds the LLM column
  ANTHROPIC_API_KEY=... FRED_API_KEY=... \
      python3 -m src.run_diagnostics --source fred --start 2019-01-01 --end 2024-12-31
  python3 -m src.run_diagnostics --out reports/phase1.md   # also save markdown

The LLM training-cutoff prescience test is only meaningful on --source fred:
synthetic series have no real future for a model to have memorized.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from functools import partial

import pandas as pd

from src.layered.analysts.macro_rates import MACRO_RATES_SERIES, macro_rates_analysts
from src.layered.diagnostics import DiagnosticsReport, run_diagnostics
from src.layered.pm.macro_rates import RATES_UNIVERSE


def _load_data(args):
    if args.source == "fred":
        from src.data.markets import fetch_macro_bundle, fetch_prices
        syms = [i.symbol for i in RATES_UNIVERSE]
        prices = fetch_prices(syms, args.start, args.end)
        macro_start = (pd.Timestamp(args.start) - pd.Timedelta(days=1095)).strftime("%Y-%m-%d")
        macro = fetch_macro_bundle(list(MACRO_RATES_SERIES), macro_start, args.end)
        return macro, prices, None
    from src.layered.synthetic import generate
    macro, prices = generate(args.start, args.end, regime=args.regime)
    return macro, prices, args.regime


def _pair(title: str, det: pd.DataFrame, llm, note: str = "") -> list[str]:
    out = [f"### {title}", ""]
    if note:
        out += [f"_{note}_", ""]
    out += ["**Deterministic Phase-1**", "", "```", det.to_string(), "```", ""]
    if llm is not None:
        out += ["**LLM Phase-2**", "", "```", llm.to_string(), "```", ""]
    else:
        out += ["_LLM column: n/a (no ANTHROPIC_API_KEY set)_", ""]
    return out


def render(r: DiagnosticsReport) -> str:
    L: list[str] = []
    L.append("# Phase-1 analyst diagnostics")
    L.append("")
    src = r.source + (f" ({r.regime})" if r.regime else "")
    L.append(f"- source: **{src}**, horizon: **{r.horizon_days}d**, meetings: **{len(r.dates)}** "
             f"({r.dates[0].date()} → {r.dates[-1].date()})")
    L.append(f"- LLM agents: **{'yes' if r.has_llm else 'no (deterministic only)'}**")
    L.append("")

    L.append("## 1. Faithfulness")
    L.append("")
    L.append("_Input isolation is structural — the base class hands each analyst an AsOf gate, "
             "so it can only read its declared series. `isolation_ok` and `no_lookahead` should be True for all._")
    L.append("")
    L += ["```", r.input_isolation.to_string(), "```", ""]
    L += _pair(
        "Responsiveness vs contamination (view vs each driver's honest measurement)",
        r.faithfulness_det, r.faithfulness_llm,
        note="own_corr high + cross_corr low ⇒ faithful. Deterministic own_corr = 1 by construction; "
             "the LLM column is where drift shows up.",
    )
    L += _pair("Reasoning stays on-topic (lexicon proxy)", r.reasoning_det, r.reasoning_llm,
               note="on_topic_rate high, contamination_rate low ⇒ the text stays about the driver. "
                    "CAVEAT: high contamination on balance_sheet / term_premium is largely a lexicon "
                    "artifact — their mandates describe the transmission (e.g. 'runoff → term premium'), "
                    "so the text legitimately name-drops the other end. Cross-check against §4 (correlation): "
                    "if view correlation did NOT rise, it is not genuine contamination.")

    L.append("## 2. Correctness")
    L += _pair(
        "Directional accuracy over the horizon, vs a persistence baseline",
        r.correctness_det, r.correctness_llm,
        note="edge_vs_persistence > 0 means the analyst beats 'the last move continues'; "
             "edge_vs_random > 0 means it beats a coin flip.",
    )

    L.append("## 3. Lookahead")
    L.append("")
    L.append("_Data-slice lookahead is covered by input isolation above (no_lookahead column) and the AsOf unit test. "
             "Below is the LLM training-cutoff test._")
    L.append("")
    if r.prescience.empty:
        L += ["_Prescience: n/a — needs the LLM column._", ""]
    else:
        L += ["```", r.prescience.to_string(), "```", ""]
        L += ["_`verdict` is computed per driver (not a fixed template): a leak only shows as the LLM "
              "beating the no-future-info baseline, so `information_gain ≤ 0` ⇒ no leak signal. The test "
              "only bites when the window spans the model's training cutoff — a pre-cutoff-only window "
              "(e.g. 2019–2024 for a 2025-cutoff model) has no post-cutoff control and is inconclusive by "
              "construction. `override_hit` is noise when `override_n` is small (< 10)._", ""]

    L.append("## 4. Correlation between agents")
    L.append("")
    L.append(f"Average |off-diagonal| — deterministic: **{r.avg_offdiag_det}**"
             + (f", LLM: **{r.avg_offdiag_llm}**" if r.corr_llm is not None else "")
             + ".  Lower = more independent (the isolation the thesis buys).")
    L.append("")
    L += ["**Deterministic**", "", "```", r.corr_det.to_string(), "```", ""]
    if r.corr_llm is not None:
        L += ["**LLM**", "", "```", r.corr_llm.to_string(), "```", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "fred"])
    ap.add_argument("--regime", default="hawkish", choices=["hawkish", "dovish"])
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--freq", default="W-FRI")
    ap.add_argument("--horizon-days", type=int, default=63)
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--out", default=None, help="optional path to save the report as markdown")
    # The experiment's only knob. Everything else — analysts, dates, scoring — is
    # held fixed across arms, so the diagnosis stays a controlled A/B on the input.
    ap.add_argument("--input-mode", default="vector", choices=["vector", "text", "text+vector"],
                    help="what the Phase-2 LLM reasons over (vector reproduces the original run)")
    ap.add_argument("--text-doc", default="statement", choices=["statement", "minutes"],
                    help="which FOMC document type feeds the text arms")
    ap.add_argument("--text-max-chars", type=int, default=None,
                    help="optional cap on FOMC text length (useful for long minutes)")
    args = ap.parse_args()

    llm_client = None
    if os.environ.get("ANTHROPIC_API_KEY"):
        from src.llm.anthropic_client import AnthropicClient
        llm_client = AnthropicClient(model=args.model)
        # Preflight: one cheap call so a bad key/model fails NOW, not after
        # grinding through the whole run.
        try:
            llm_client.validate()
        except Exception as e:  # noqa: BLE001
            print(f"[error] LLM preflight failed — check ANTHROPIC_API_KEY and --model.\n"
                  f"        {type(e).__name__}: {e}")
            raise SystemExit(1)
    else:
        print("[warn] ANTHROPIC_API_KEY not set — running the deterministic column only.\n")

    t0 = time.time()

    def _progress(label, i, total):
        # Only the LLM pass is slow enough to warrant a live counter.
        if label != "llm":
            return
        elapsed = time.time() - t0
        eta = (elapsed / i) * (total - i) if i else 0.0
        print(f"\r[llm] meeting {i}/{total} ({100 * i / total:4.1f}%) · "
              f"elapsed {elapsed / 60:4.1f}m · eta {eta / 60:4.1f}m",
              end="" if i < total else "\n", file=sys.stderr, flush=True)

    # Build the FOMC text source only for the text arms; ``vector`` stays offline-clean.
    text_source = None
    if args.input_mode in ("text", "text+vector"):
        from src.data.fomc_text import FomcCorpus
        text_source = FomcCorpus(doc_type=args.text_doc, max_chars=args.text_max_chars)
        print(f"[info] input-mode={args.input_mode} · FOMC {args.text_doc}s loaded: "
              f"{text_source.count} docs (point-in-time by release_date).")

    # Only the input changes: same analysts, same dates, same diagnosis. With
    # input_mode="vector" this is identical to the original macro_rates_analysts.
    make_analysts = partial(macro_rates_analysts, input_mode=args.input_mode, text_source=text_source)

    macro, prices, regime = _load_data(args)
    report = run_diagnostics(
        make_analysts=make_analysts,
        macro=macro, prices=prices,
        start=args.start, end=args.end, freq=args.freq,
        horizon_days=args.horizon_days, llm_client=llm_client,
        source=args.source, regime=regime,
        progress=_progress,
    )
    text = render(report)
    print(text)

    # Audit trail — every launch reports what it ran, how long it took, and (when
    # the LLM column is on) exactly how many calls/tokens/dollars it cost.
    audit = {
        "model": args.model,
        "source": args.source,
        "regime": regime,
        "input_mode": args.input_mode,
        "text_doc": args.text_doc if text_source is not None else None,
        "window": [args.start, args.end],
        "meetings": len(report.dates),
        "wall_clock_s": round(time.time() - t0, 1),
    }
    if llm_client is not None:
        audit["llm"] = llm_client.usage_summary()
    print("\n## Run audit\n\n```json\n" + json.dumps(audit, indent=2) + "\n```")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text + "\n")
        audit_path = os.path.splitext(args.out)[0] + ".audit.json"
        with open(audit_path, "w") as f:
            json.dump(audit, f, indent=2)
        print(f"\n[saved] {args.out}")
        print(f"[saved] {audit_path}")


if __name__ == "__main__":
    main()
