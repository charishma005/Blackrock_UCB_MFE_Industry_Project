"""Offline re-scoring: what does the structural trade layer add, holding views fixed?

The experiment the weight-vs-IC scatter calls for, at $0. Take a saved LLM PM run, keep
every meeting's *arbitrated driver block* exactly as the model produced it, and replace
only the ``trade`` with the one ``pm.structural.structural_trade`` derives from those same
convictions. The result is a normal run file — identical schema — so the existing
``trade_pnl`` graders (``load_trades`` / ``yield_pnl`` / ``trade_validity``) score it
unchanged, and "LLM freehand trade" vs "structural trade on the LLM's own views" becomes a
head-to-head on one board, one clock, one outcome.

No model calls: this reads committed driver blocks and re-derives legs by arithmetic. A
degraded meeting stays degraded (a failed arbitration has no driver block to structure).
"""
from __future__ import annotations

import json
from typing import Mapping, Optional

from src.layered.pm.structural import structural_trade


def restructure_records(records: list[dict], polarity: Mapping[str, float],
                        trade_config: Optional[dict], pod: str) -> list[dict]:
    """Return copies of ``records`` with each trade rebuilt structurally from its own
    driver block. Pure — the input records are not mutated.

    A degraded record passes through untouched (nothing to structure). A record whose
    structural trade is an abstention (``None``) has its ``trade`` set to ``None`` and its
    ``raw_response`` cleared, so ``trade_validity`` reads it as a genuine no-trade rather
    than an emitted-then-rejected one.
    """
    out: list[dict] = []
    for r in records:
        rec = dict(r)
        if rec.get("degraded"):
            out.append(rec)
            continue
        av = dict(rec.get("arbitrated") or {})
        drivers = {str(d): float(v) for d, v in (av.get("drivers") or {}).items()}
        trade = structural_trade(drivers, polarity, trade_config,
                                 pod=pod, asof=rec["asof"])
        if trade is None:
            av["trade"] = None
            rec["raw_response"] = None
        else:
            legs = list(trade.legs.items())
            av["trade"] = {
                "strategy": trade.strategy,
                "asof": str(trade.asof.date()),
                "legs": dict(legs),
                "conviction": trade.conviction,
                "rationale": trade.rationale,
                "risk": trade.risk,
            }
            # A synthesized tool-shaped reply so trade_validity reports the trade as fully
            # emitted with nothing dropped — true, since structural legs are in-universe by
            # construction. Mirrors MechanicalPM.arbitrate's last_raw.
            rec["raw_response"] = json.dumps({
                "notes": av.get("notes", ""),
                "drivers": [{"driver": d, "conviction": v, "why": "structural"}
                            for d, v in drivers.items()],
                "trade": {"flat": not legs,
                          "legs": [{"instrument": k, "weight": w} for k, w in legs],
                          "conviction": trade.conviction,
                          "rationale": trade.rationale},
            })
        rec["arbitrated"] = av
        out.append(rec)
    return out


def restructure_run(in_path: str, out_path: str, polarity: Mapping[str, float],
                    trade_config: Optional[dict], pod: str) -> int:
    """Read an LLM PM run JSONL, write its structural-trade counterpart. Returns the
    record count. The ``.meta.json`` is the caller's to copy/annotate."""
    with open(in_path) as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    rebuilt = restructure_records(records, polarity, trade_config, pod)
    with open(out_path, "w") as fh:
        for rec in rebuilt:
            fh.write(json.dumps(rec, default=str) + "\n")
    return len(rebuilt)
