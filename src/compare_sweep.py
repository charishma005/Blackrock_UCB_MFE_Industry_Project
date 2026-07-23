"""Compare analyst IC across model runs — the model-sweep report.

Reads several ``run_analyst_ic`` JSONL outputs and tabulates, per model, the
numbers that decide whether a difference is real and whether the reasoning is
worth anything:

  * IC (full sample) with its t-stat — skill against the null of zero
  * IC pre-COVID (≤2019) and COVID-on (≥2020) — is the skill regime-robust?
  * calibration split — signed-conviction IC vs direction-only IC. If they match,
    the model's confidence carries no information beyond its call.

Same clock, same outcome, same code for every model, so the only thing varying is
the reasoner. With ~12 independent observations a year, small IC gaps are not
distinguishable — read the t-stats, not the third decimal.

    python3 -m src.compare_sweep reports/sweep_haiku.jsonl reports/sweep_sonnet.jsonl \
        reports/sweep_opus.jsonl
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.layered.evaluation import ICEvaluator, load_run


def load(path: str) -> tuple[str, pd.Series, pd.Series]:
    run = load_run(path)   # shared loader: signed conviction + level, degraded dropped
    name = path.split("/")[-1].replace("sweep_", "").replace(".jsonl", "")
    return name, run.signed, run.level


def row(name: str, signed: pd.Series, level: pd.Series) -> dict:
    lv = level.dropna()
    full = ICEvaluator(lv, steps=1).evaluate(signed, name)
    pre = _sub(signed, lv, None, "2020-01-01")
    cov = _sub(signed, lv, "2020-01-01", None)
    ev = ICEvaluator(lv, steps=1)
    dir_ic = ev.evaluate(np.sign(signed), "d").ic
    return {
        "model": name, "n": full.n,
        "IC": round(full.ic, 3), "t": round(full.t_stat, 2),
        "IC_preCOVID": pre, "IC_COVID+": cov,
        "IC_dir_only": round(dir_ic, 3),
        "conv_adds": round(full.ic - dir_ic, 3),
        "hit": round(full.hit_rate, 3),
    }


def _sub(signed, level, a, b) -> float:
    idx = signed.index
    m = np.ones(len(idx), dtype=bool)
    if a: m &= idx >= pd.Timestamp(a)
    if b: m &= idx < pd.Timestamp(b)
    r = ICEvaluator(level[m].dropna(), steps=1).evaluate(signed[m], "s")
    return round(r.ic, 3)


def main():
    paths = sys.argv[1:]
    if not paths:
        print("usage: python -m src.compare_sweep <run1.jsonl> <run2.jsonl> ...")
        raise SystemExit(1)
    rows = [row(*load(p)) for p in paths]
    df = pd.DataFrame(rows).set_index("model")
    print(df.to_string())
    print("\nbar: IR 1.0 needs IC 0.29 (breadth ~11.8/yr). "
          "conv_adds = IC(signed) - IC(direction); >0 means conviction carries information. "
          "Read t, not small IC gaps.")


if __name__ == "__main__":
    main()
