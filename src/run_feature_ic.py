"""Feature IC on the release clock — the free check, before any LLM spend.

Answers the question that has to come first: **is this driver predictable at all
from measurements available at the time?** If nothing in the feature block carries
an information coefficient against the next print, then a weak result from the
analyst is a fact about the problem, not about the analyst.

Costs nothing — local FRED CSVs, no API key, no model call.

    python3 -m src.run_feature_ic                                  # inflation, next release
    python3 -m src.run_feature_ic --start 2005-01-01 --end 2019-12-31
    python3 -m src.run_feature_ic --steps 1,2,3                    # evaluation only

Note the steps flag grades features further ahead; it does **not** change what the
analyst predicts, which is fixed at the next release.
"""
from __future__ import annotations

import argparse

import pandas as pd
import yaml

from src.data.equity_local import load_any_bundle as load_bundle
from src.layered.analysts.llm_analyst import PERSONA_DIR
from src.layered.evaluation import FeaturePanel, ICEvaluator, release_dates, required_ic
from src.layered.features import FeatureEngine, from_persona


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", default="inflation")
    ap.add_argument("--start", default="2005-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--steps", default="1", help="releases ahead to grade (comma-separated)")
    ap.add_argument("--clock", default=None, help="clock series (default: first declared input)")
    ap.add_argument("--out", default=None, help="write the IC table as CSV")
    args = ap.parse_args()

    persona = yaml.safe_load((PERSONA_DIR / f"{args.driver}.yaml").read_text()) or {}
    horizon = persona.get("horizon") or {}
    engine = FeatureEngine(from_persona(args.driver, persona))
    panel_builder = FeaturePanel(engine)

    macro = load_bundle(list(engine.inputs))
    clock = args.clock or horizon.get("clock") or panel_builder.clock_series
    freq = horizon.get("clock_freq")   # e.g. "ME" for a daily market driver
    dates = release_dates(macro, clock, args.start, args.end, freq=freq)

    print(f"driver     {args.driver}")
    print(f"inputs     {', '.join(engine.inputs)}")
    print(f"clock      {clock}{f' (resampled {freq})' if freq else ''} — {len(dates)} releases, "
          f"{dates[0].date()} → {dates[-1].date()}")

    panel = panel_builder.build(macro, dates)
    level = panel_builder.level(panel)
    print(f"panel      {panel.shape[0]} dates × {panel.shape[1]} features\n")

    frames = []
    for step in [int(s) for s in args.steps.split(",") if s.strip()]:
        ev = ICEvaluator(level, steps=step)
        table = ev.evaluate_frame(panel)
        breadth = ev.breadth
        print("=" * 72)
        print(f"IC vs the move over the next {step} release(s) — "
              f"breadth {breadth:.1f} bets/yr")
        print(f"bar: IR 0.5 needs IC {required_ic(0.5, breadth):.2f} · "
              f"IR 1.0 needs IC {required_ic(1.0, breadth):.2f}")
        print("=" * 72)
        print(table.to_string())
        print()
        frames.append(table.assign(steps=step))

    if args.out:
        pd.concat(frames).to_csv(args.out)
        print(f"[saved] {args.out}")

    print("Caveats: observations are non-overlapping, so the t-stat needs no "
          "autocorrelation correction; p_approx is a normal approximation. "
          "hit_rate is meaningless for features without a natural zero.\n"
          "These ICs diagnose the PROBLEM. They must not be used to select "
          "features for the prompt — that would turn a measurement into a fitted signal.")


if __name__ == "__main__":
    main()
