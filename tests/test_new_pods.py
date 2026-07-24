"""The two cross-roster pods (global_rv, equities) compose and control correctly.

The glob suites already cover prompt composition for every pod; these tests pin
the two new pods by name, guard their listens_to against persona typos, and pin
global_rv's declared structural pair (mirroring the shipped-curve guard in
test_structural.py).
"""
from __future__ import annotations

import yaml

from src.layered.analysts.llm_analyst import PERSONA_DIR
from src.layered.pm.build import build_pm
from src.layered.pm.mechanical_pm import POD_DIR, MechanicalPM
from src.layered.pm.structural import structural_trade

NEW_PODS = ["global_rv", "equities"]


def _spec(pod):
    return yaml.safe_load((POD_DIR / f"{pod}.yaml").read_text())


def test_new_pods_compose_a_system_prompt():
    for pod in NEW_PODS:
        pm = build_pm(pod, None)
        assert pm._system_prompt().strip(), f"{pod}: empty system prompt"


def test_every_listened_driver_has_a_persona():
    for pod in NEW_PODS:
        for driver in _spec(pod)["listens_to"]:
            assert (PERSONA_DIR / f"{driver}.yaml").exists(), \
                f"{pod} listens to {driver!r} but no persona exists"


def test_mechanical_pm_constructs_with_full_polarity():
    for pod in NEW_PODS:
        m = MechanicalPM.from_pod(pod)
        assert set(m.polarity) == set(m.listens_to)
        assert all(p in (+1.0, -1.0) for p in m.polarity.values())


def test_global_rv_declares_the_transatlantic_structural_pair():
    """A US-side hawkish call (spread-widening on this pod's axis) must decompose
    into an equal-and-opposite +DGS10 / -Bund pair, not abstain."""
    cfg = _spec("global_rv")
    trade_cfg = cfg["trade"]
    pol = {d: float((c or {}).get("polarity", 1.0))
           for d, c in cfg["listens_to"].items()}
    assert trade_cfg.get("sign_convention") == "opposed"
    roles = trade_cfg.get("leg_roles") or {}
    assert roles.get("long") == "DGS10" and roles.get("front") == "INTL_DE10Y"
    assert set(roles.values()) <= set(trade_cfg["universe"])
    # term_premium +1 calling 'up' = US long-end pressure = spread widens.
    t = structural_trade({"term_premium": 0.8}, pol, trade_cfg,
                         pod="global_rv", asof="2023-06-30")
    assert t is not None
    assert t.legs[roles["long"]] > 0 and t.legs[roles["front"]] < 0
    assert abs(t.legs[roles["long"]] + t.legs[roles["front"]]) < 1e-9
    # A foreign-yields-up call (ea_rates 'up', polarity -1) must NARROW the spread.
    t2 = structural_trade({"ea_rates": 0.8}, pol, trade_cfg,
                          pod="global_rv", asof="2023-06-30")
    assert t2 is not None and t2.legs[roles["long"]] < 0 < t2.legs[roles["front"]]


def test_equities_pod_has_no_trade_property():
    pm = build_pm("equities", None)
    assert not pm.trade_config
