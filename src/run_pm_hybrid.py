"""Run the hybrid PM (v1) — mechanical relevance baseline + a bounded LLM multiplier.

The mechanical ``RelevancePM`` (v0) sets each analyst's walk-forward relevance weight; the
LLM may nudge each by a report-justified multiplier in [0.5, 2.0]. This writes the same
JSONL + ``.meta.json`` schema the other PM runs use (plus an ``adjustments`` field), so
``trade_pnl`` scores it head-to-head with v0 and the baselines. The decisive read is
v1-vs-v0: do the report-driven adjustments beat the pure mechanical baseline?

    python3 -m src.run_pm_hybrid --pod duration --model claude-sonnet-5 \
        --out reports/pm/duration_hybrid.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import pandas as pd

from src.data.fred_local import load_bundle
from src.layered.evaluation.pm_runs import load_pm_run
from src.layered.evaluation.trade_pnl import (load_trades, score_trades, trade_validity,
                                              yield_pnl)
from src.layered.evaluation.trade_pnl import summarize as summarize_trades
from src.layered.pm.board import ViewBoard
from src.layered.pm.build import preflight_llm
from src.layered.pm.disagreement import override, panel_disagreement
from src.layered.pm.hybrid_pm import HybridPM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod", default="duration")
    ap.add_argument("--board", default="reports/ab")
    ap.add_argument("--board-suffix", default="_on")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--weighting", default="ic")
    ap.add_argument("--min-obs", type=int, default=12)
    ap.add_argument("--shrink-k", type=float, default=8.0)
    ap.add_argument("--no-identity-check", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the prompt, make no call")
    ap.add_argument("--out", default="reports/pm/pm_hybrid.jsonl")
    args = ap.parse_args()

    llm = None if args.dry_run else preflight_llm(args.model, max_tokens=args.max_tokens)
    pm = HybridPM.from_pod(args.pod, llm=llm, weighting=args.weighting,
                           min_obs=args.min_obs, shrink_k=args.shrink_k)
    board = ViewBoard.from_dir(args.board, args.board_suffix, drivers=pm.reads,
                               check_identity=not args.no_identity_check, **pm.board_kwargs)
    dates = board.meeting_dates(freq=pm.clock_freq, start=args.start, end=args.end)
    if args.limit:
        dates = dates[: args.limit]
    pm.fit(board, dates, load_bundle(list(pm.trade_config.get("universe") or [])))

    if args.dry_run:
        m = pm.build_inputs(board, dates[0])
        print("SYSTEM\n" + "=" * 70 + f"\n{pm._system_prompt()}\n")
        base = pm._baseline(m)
        tot = sum(abs(v) for v in base.values()) or 1.0
        print("USER (brief + baseline weights)\n" + "=" * 70)
        from src.layered.pm.brief import render_brief
        print(render_brief(m, drivers=pm.reads)[:1200] + "\n  ...")
        print("\nbaseline weights:\n" + "\n".join(
            f"  {d}: {base[d] / tot:+.2f}" for d in sorted(base, key=lambda d: -abs(base[d]))))
        return

    print(f"[info] HYBRID pod={args.pod} weighting={args.weighting} meetings={len(dates)} "
          f"model={args.model}", file=sys.stderr)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    meta_path = os.path.splitext(args.out)[0] + ".meta.json"
    with open(meta_path, "w") as fh:
        json.dump({"config": vars(args), "pod": args.pod, "kind": f"hybrid:{args.weighting}",
                   "listens_to": pm.listens_to, "polarity": pm.polarity,
                   "clock_freq": pm.clock_freq, "answer_space": pm.answer_space,
                   "memory": False, "board_thresholds": pm.board_kwargs,
                   "system_prompt": pm._system_prompt(), "n_meetings": len(dates),
                   "window": [str(dates[0].date()), str(dates[-1].date())] if len(dates) else [],
                   "board_sources": board.sources}, fh, indent=2, default=str)

    with open(args.out, "w") as fh:
        for i, asof in enumerate(dates, 1):
            m = pm.build_inputs(board, asof)
            brief = pm._user_prompt(m)
            av = pm.arbitrate(m)
            fh.write(json.dumps({
                "asof": asof, "degraded": not av.drivers,
                "brief_sha256": hashlib.sha256(brief.encode("utf-8")).hexdigest(),
                "user_prompt": brief,
                "board": {d: {"present": e.present, "carried": e.carried,
                              "direction": e.view.direction if e.present else None,
                              "conviction": e.view.conviction if e.present else None}
                          for d, e in m.entries.items()},
                "raw_response": pm.last_raw,
                "arbitrated": av.model_dump(mode="json"),
                "adjustments": pm.last_adjust,
                "why": pm.why(),
                "override": override(av.drivers, m),
                "coverage": m.coverage,
                "panel_disagreement": panel_disagreement(m, pm.polarity),
            }, default=str) + "\n")
            if i % 12 == 0:
                print(f"  [{i}/{len(dates)}] {asof.date()}", file=sys.stderr)
    print(f"[saved] {args.out}\n[saved] {meta_path}")

    if pm.trade_config:
        instruments = list(pm.trade_config.get("universe") or [])
        try:
            trades = load_trades(args.out, pm.trade_config)
            macro = load_bundle(instruments)
            pnl = yield_pnl(trades, macro, instruments, freq=pm.clock_freq)
            score = score_trades(pnl, trades.reindex(pnl.index)["conviction"])
            print("\n## The hybrid trade — yield-space P&L (compare to v0 relevance + mechanical)\n")
            print(summarize_trades(score, trade_validity(trades)))
        except Exception as e:  # noqa: BLE001
            print(f"\n[warn] trade scoring failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
