"""The structural trade layer: driver block → legs, deterministically.

Pins the mechanism the weight-vs-IC scatter motivates — hold the LLM's driver views
fixed, construct the trade by declared structure — on synthetic inputs, and checks the
offline re-scoring harness produces a file the existing ``trade_pnl`` grader consumes.
"""
from __future__ import annotations

import json

import pandas as pd

from src.layered.evaluation.structural_bench import restructure_records
from src.layered.evaluation.trade_pnl import load_trades
from src.layered.pm.structural import rate_axis_projection, structural_trade

# A duration-shaped pod: level view, both legs one sign.
DURATION_CFG = {"universe": ["DGS2", "DGS10"], "max_legs": 2, "sign_convention": "same"}
DURATION_POL = {"inflation": 1.0, "balance_sheet": -1.0, "term_premium": 1.0}
# A curve-shaped pod: slope view, legs equal and opposite.
CURVE_CFG = {"universe": ["DGS2", "DGS10"], "max_legs": 2, "sign_convention": "opposed"}
CURVE_POL = {"term_premium": 1.0, "curve_slope": -1.0}


# ── projection: polarity orients, undeclared drivers are skipped ─────────────
def test_projection_orients_by_declared_polarity():
    """A −1-polarity driver whose own measurement FALLS (conviction −0.8) is UPWARD rate
    pressure: polarity × conviction = (−1)(−0.8) = +0.8. Hotter inflation (+0.6, +1) adds
    to it. Mean of {+0.8, +0.6} = +0.70."""
    proj = rate_axis_projection({"balance_sheet": -0.8, "inflation": 0.6}, DURATION_POL)
    assert proj == 0.70


def test_projection_skips_undeclared_drivers():
    """A driver with no declared polarity cannot enter the projection — the same rule
    disagreement.oriented holds, so a pod reading more than it owns stays clean."""
    proj = rate_axis_projection({"inflation": 0.5, "some_other_driver": 0.9}, DURATION_POL)
    assert proj == 0.5   # only inflation counts


# ── same-convention (duration): both legs the projection's sign, unit gross ──
def test_same_pod_takes_both_legs_one_sign():
    t = structural_trade({"inflation": 0.6, "balance_sheet": -0.4}, DURATION_POL,
                         DURATION_CFG, pod="duration", asof="2023-06-30")
    assert t is not None
    assert set(t.legs) == {"DGS2", "DGS10"}
    assert all(w > 0 for w in t.legs.values())          # projection +0.5 → up
    assert abs(sum(abs(w) for w in t.legs.values()) - 1.0) < 1e-9   # unit gross


def test_same_pod_flips_sign_with_the_projection():
    t = structural_trade({"inflation": -0.6, "balance_sheet": 0.4}, DURATION_POL,
                         DURATION_CFG, pod="duration", asof="2023-06-30")
    assert t is not None and all(w < 0 for w in t.legs.values())


# ── a zero projection is a decided flat, not an abstention ───────────────────
def test_zero_projection_is_a_real_flat_not_none():
    t = structural_trade({"inflation": 0.5, "balance_sheet": 0.5}, DURATION_POL,
                         DURATION_CFG, pod="duration", asof="2023-06-30")
    # +0.5 (inflation) and −0.5 (balance_sheet, polarity −1 × +0.5) → mean 0.0
    assert t is not None and t.legs == {} and t.conviction == 0.0


# ── opposed (curve): abstains without leg_roles, decomposes with them ────────
def test_opposed_pod_abstains_without_declared_leg_roles():
    t = structural_trade({"curve_slope": 0.6}, CURVE_POL, CURVE_CFG,
                         pod="curve", asof="2023-06-30")
    assert t is None            # no front/long split declared → honest abstention


def test_opposed_pod_builds_a_steepener_with_leg_roles():
    cfg = {**CURVE_CFG, "leg_roles": {"front": "DGS2", "long": "DGS10"}}
    # curve_slope −0.6, polarity −1 → projection +0.6 (upward rate pressure = steepen).
    t = structural_trade({"curve_slope": -0.6}, CURVE_POL, cfg,
                         pod="curve", asof="2023-06-30")
    assert t is not None
    assert t.legs["DGS10"] > 0 and t.legs["DGS2"] < 0        # long the long end, short front
    assert abs(t.legs["DGS10"] + t.legs["DGS2"]) < 1e-9      # equal and opposite


def test_no_trade_config_is_none():
    assert structural_trade({"inflation": 0.6}, DURATION_POL, None,
                            pod="x", asof="2023-06-30") is None


def test_shipped_curve_pod_declares_leg_roles_and_decomposes():
    """Not a synthetic config — the real ``curve.yaml`` on disk. The mechanism above only
    fires if the pod actually declares ``leg_roles``; this pins that the shipped curve seat
    does, so the structural layer trades the slope pod instead of abstaining. Guards the
    config line itself against a rename or a dropped key."""
    import yaml
    from src.layered.pm.mechanical_pm import POD_DIR

    cfg = yaml.safe_load((POD_DIR / "curve.yaml").read_text())
    trade_cfg = cfg["trade"]
    pol = {d: float((c or {}).get("polarity", 1.0))
           for d, c in cfg["listens_to"].items()}
    assert trade_cfg.get("sign_convention") == "opposed"
    roles = trade_cfg.get("leg_roles") or {}
    assert roles.get("front") in trade_cfg["universe"]
    assert roles.get("long") in trade_cfg["universe"]
    # term_premium +1 calling 'up' is long-end pressure → a steepener, equal and opposite.
    t = structural_trade({"term_premium": 0.8}, pol, trade_cfg,
                         pod="curve", asof="2023-06-30")
    assert t is not None
    assert t.legs[roles["long"]] > 0 and t.legs[roles["front"]] < 0
    assert abs(t.legs[roles["long"]] + t.legs[roles["front"]]) < 1e-9


# ── the harness: driver block preserved, only the trade rebuilt ──────────────
def _llm_record(asof, drivers, llm_legs, degraded=False):
    return {"asof": asof, "degraded": degraded,
            "arbitrated": {"asof": asof, "drivers": drivers, "disagreement": 0.1,
                           "notes": "n",
                           "trade": None if degraded else
                           {"strategy": "duration", "asof": asof, "legs": llm_legs,
                            "conviction": 0.5, "rationale": "llm", "risk": {"tags": []}}}}


def test_restructure_keeps_the_driver_block_and_swaps_only_the_trade():
    recs = [_llm_record("2023-06-30", {"inflation": 0.6, "balance_sheet": -0.4},
                        {"DGS2": 0.2, "DGS10": 0.9})]   # LLM's lopsided freehand legs
    out = restructure_records(recs, DURATION_POL, DURATION_CFG, "duration")
    assert out[0]["arbitrated"]["drivers"] == recs[0]["arbitrated"]["drivers"]  # views untouched
    new_legs = out[0]["arbitrated"]["trade"]["legs"]
    assert new_legs != {"DGS2": 0.2, "DGS10": 0.9}                              # trade replaced
    assert abs(new_legs["DGS2"] - new_legs["DGS10"]) < 1e-9                     # structural: equal legs


def test_restructure_passes_degraded_through_untouched():
    recs = [_llm_record("2023-06-30", {}, {}, degraded=True)]
    out = restructure_records(recs, DURATION_POL, DURATION_CFG, "duration")
    assert out[0]["arbitrated"]["trade"] is None and out[0]["degraded"]


def test_restructured_run_is_scoreable_by_trade_pnl(tmp_path):
    recs = [_llm_record("2023-06-30", {"inflation": 0.6, "balance_sheet": -0.4},
                        {"DGS2": 0.5, "DGS10": 0.5}),
            _llm_record("2023-07-31", {"inflation": -0.5}, {"DGS2": -0.5, "DGS10": -0.5})]
    out = restructure_records(recs, DURATION_POL, DURATION_CFG, "duration")
    path = tmp_path / "duration_structural.jsonl"
    path.write_text("\n".join(json.dumps(r, default=str) for r in out) + "\n")

    trades = load_trades(str(path), DURATION_CFG)
    assert len(trades) == 2
    assert trades["has_trade"].all()
    assert int(trades["sign_violation"].sum()) == 0     # same-sign legs, no violation
