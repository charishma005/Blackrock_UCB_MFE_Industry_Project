"""Offline test for the Phase-1 diagnostics harness — no API keys, no network.

The LLM path is exercised with a STUB client (returns JSON like the real one),
so the deterministic-vs-LLM side-by-side machinery is proven end to end without
touching Anthropic. Checks:
  1. deterministic-only run produces every diagnostic
  2. input isolation + no-lookahead hold for all four analysts
  3. deterministic faithfulness own_corr == 1 (graded against itself)
  4. with a stub LLM: LLM columns populate; prescience table has override stats
  5. correlation matrix is 4x4 with a finite average off-diagonal
"""
import json
import re
import sys

import numpy as np

from src.layered.analysts.macro_rates import macro_rates_analysts
from src.layered.diagnostics import run_diagnostics
from src.layered.synthetic import generate

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if (not condition and detail != "") else ""))


class StubLLM:
    """Deterministic fake: echoes the Phase-1 reading, but always calls
    term_premium 'up' to manufacture overrides vs the deterministic direction —
    enough to exercise the prescience/override code path offline."""

    def complete(self, system: str, user: str) -> str:
        driver = (re.search(r"Driver:\s*(\S+)", user) or [None, "x"])[1]
        det_dir = (re.search(r'"direction":\s*"(\w+)"', user) or [None, "flat"])[1]
        conv = float((re.search(r'"conviction":\s*([0-9.]+)', user) or [None, "0.4"])[1])
        direction = "up" if driver == "term_premium" else det_dir
        return json.dumps({"direction": direction, "conviction": max(conv, 0.3),
                           "reasoning": f"assessment of {driver}: {direction}"})


macro, prices = generate("2022-01-01", "2024-12-31", regime="hawkish")

# ── 1: deterministic-only ────────────────────────────────────────────────────
r = run_diagnostics(macro_rates_analysts, macro, prices, "2022-01-01", "2024-12-31",
                    llm_client=None, source="synthetic", regime="hawkish")
check("Deterministic run has 4 analysts in input isolation", len(r.input_isolation) == 4,
      r.input_isolation.index.tolist())
check("All analysts pass input isolation", bool(r.input_isolation["isolation_ok"].all()))
check("No analyst reads past asof", bool(r.input_isolation["no_lookahead"].all()))
check("Deterministic faithfulness own_corr == 1", bool((r.faithfulness_det["own_corr"] == 1.0).all()),
      r.faithfulness_det["own_corr"].tolist())
check("Correctness table has a row per driver", len(r.correctness_det) == 4)
check("Correlation matrix is 4x4", r.corr_det.shape == (4, 4), r.corr_det.shape)
check("Average off-diagonal is finite", np.isfinite(r.avg_offdiag_det), r.avg_offdiag_det)
check("Prescience empty without LLM", r.prescience.empty)
check("LLM columns are None without a client", r.faithfulness_llm is None and r.correctness_llm is None)

# ── 2: with a stub LLM — side-by-side machinery ──────────────────────────────
r2 = run_diagnostics(macro_rates_analysts, macro, prices, "2022-01-01", "2024-12-31",
                     llm_client=StubLLM(), source="synthetic", regime="hawkish")
check("LLM run flags has_llm", r2.has_llm)
check("LLM faithfulness table populated", r2.faithfulness_llm is not None and len(r2.faithfulness_llm) == 4)
check("LLM correctness table populated", r2.correctness_llm is not None and len(r2.correctness_llm) == 4)
check("LLM reasoning on-topic populated", r2.reasoning_llm is not None and len(r2.reasoning_llm) == 4)
check("LLM correlation matrix computed", r2.corr_llm is not None and r2.corr_llm.shape == (4, 4))
check("Prescience table populated with LLM", not r2.prescience.empty and "information_gain" in r2.prescience.columns)
check("Prescience captured term_premium overrides (stub forces 'up')",
      int(r2.prescience.loc["term_premium", "override_n"]) > 0,
      r2.prescience.loc["term_premium", "override_n"])
check("Prescience verdict marks synthetic as non-testable",
      "synthetic" in r2.prescience.loc["inflation", "verdict"])

# ── summary ──────────────────────────────────────────────────────────────
print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
