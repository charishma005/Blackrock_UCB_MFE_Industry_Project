"""Invariants over the committed run artifacts themselves.

The rest of the suite proves the *code* holds an invariant on synthetic fixtures;
these tests hold the *shipped run files* to the same invariants. They are the
difference between "the parser would reject a mandate-violating trade" and "no
mandate-violating trade is in the run the results tables were computed from".

Every test here skips when the artifacts are absent (the house pattern for
real-corpus tests), so the suite passes on a fresh clone and starts biting the
moment run logs are committed. NOTE: ``reports/`` is currently in .gitignore —
these tests are written ahead of the logs landing; committing a run requires
``git add -f reports/...`` or a .gitignore carve-out.

What is deliberately NOT here: anything already pinned on the real corpus by
``test_pm_board`` (board loading, identity check), ``test_pm_no_lookahead``
(snap causality), or ``test_pm_prompt_guardrails`` (prompt hygiene).
"""
from __future__ import annotations

import glob
import json
import os
import re

import pytest
import yaml

from src.layered.evaluation.trade_pnl import load_trades

POD_DIR = "src/layered/pm/pods"
AB_GLOB = "reports/ab/*_on.jsonl"
PM_GLOB = "reports/pm/*.jsonl"


# ── discovery helpers ────────────────────────────────────────────────────────
def _pod_names() -> list[str]:
    """Pod names, longest first so ``front_end_mem_on`` matches ``front_end``,
    not a hypothetical pod called ``front``."""
    names = [os.path.splitext(os.path.basename(p))[0]
             for p in glob.glob(os.path.join(POD_DIR, "*.yaml"))
             if not os.path.basename(p).startswith("_")]
    return sorted(names, key=len, reverse=True)


def _pod_of(run_path: str) -> str | None:
    base = os.path.splitext(os.path.basename(run_path))[0]
    for name in _pod_names():
        if base == name or base.startswith(name + "_"):
            return name
    return None


def _pod_config(pod: str) -> dict:
    with open(os.path.join(POD_DIR, f"{pod}.yaml")) as fh:
        return yaml.safe_load(fh) or {}


def _records(path: str) -> list[dict]:
    with open(path) as fh:
        recs = [json.loads(line) for line in fh if line.strip()]
    return sorted(recs, key=lambda r: r["asof"])


def _meta(path: str) -> dict:
    meta_path = os.path.splitext(path)[0] + ".meta.json"
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path) as fh:
        return json.load(fh)


def _pm_runs() -> list[str]:
    return sorted(glob.glob(PM_GLOB))


def _ab_runs() -> list[str]:
    return sorted(glob.glob(AB_GLOB))


# ── 1. the mandate held in the artifact, not just in the parser ─────────────
def test_committed_trades_respect_the_pod_sign_convention():
    """Zero sign violations in every committed PM run of a convention pod.

    ``_parse_trade`` rejects a violating trade at emission and ``trade_pnl``
    counts violations after the fact; this asserts the two agree on the runs
    the results tables actually cite. A nonzero count here means a run predates
    the enforcement fix and its P&L blends mandate-violating trades — the exact
    failure ``trade_validity`` warns is invisible in the headline numbers.
    """
    runs = [p for p in _pm_runs()
            if _pod_of(p) and _pod_config(_pod_of(p)).get("trade", {}).get("sign_convention")]
    if not runs:
        pytest.skip(f"no committed PM runs for convention pods under {PM_GLOB}")
    for path in runs:
        trades = load_trades(path, _pod_config(_pod_of(path))["trade"])
        n_bad = int(trades["sign_violation"].sum())
        assert n_bad == 0, (
            f"{path}: {n_bad} trade(s) violate the pod's declared sign_convention "
            f"— this run predates parse-time enforcement and its P&L is suspect")


# ── 2. memory replays the previous meeting, verbatim and backwards-only ─────
_CALLS_LINE = re.compile(r"You called: (.+)")
_CALL_TOKEN = re.compile(r"(\w+) ([+-]\d+\.\d{2})")


def test_memory_block_replays_the_previous_meeting_verbatim():
    """In a memory-on run, the recorded prompt's memory block must restate the
    PREVIOUS non-degraded record's arbitration — same drivers, same convictions
    (to the 2dp the block renders), and the correct one of the three position
    states (carrying legs / chose flat / no position view).

    This is the replay audit for the one prompt surface that reaches backwards:
    if it ever disagrees with the prior record, either the memory is not what
    was recorded (audit-trail bug) or it reached somewhere other than the
    previous meeting (causality bug). Degraded meetings must be skipped over,
    never replayed — a failed call does not become the PM's position.
    """
    runs = [p for p in _pm_runs()
            if (_meta(p).get("config") or {}).get("memory")
            and not (_meta(p).get("config") or {}).get("perturb")]
    if not runs:
        pytest.skip(f"no memory-on, unperturbed PM runs under {PM_GLOB}")

    for path in runs:
        recs = _records(path)
        last_good = None
        for i, rec in enumerate(recs):
            prompt = rec.get("user_prompt") or ""
            if last_good is None:
                assert "Your previous meeting" not in prompt, (
                    f"{path} record {i}: memory block before any arbitration exists")
            else:
                assert "Your previous meeting" in prompt, (
                    f"{path} record {i}: memory-on run missing its memory block")
                prev = last_good["arbitrated"]

                m = _CALLS_LINE.search(prompt)
                prev_drivers = prev.get("drivers") or {}
                if prev_drivers:
                    assert m, f"{path} record {i}: no 'You called:' line"
                    replayed = {d: float(v) for d, v in _CALL_TOKEN.findall(m.group(1))}
                    assert set(replayed) == set(prev_drivers), (
                        f"{path} record {i}: memory names {sorted(replayed)} but the "
                        f"previous meeting called {sorted(prev_drivers)}")
                    for d, v in prev_drivers.items():
                        assert abs(replayed[d] - float(v)) <= 0.005 + 1e-9, (
                            f"{path} record {i}: {d} replayed as {replayed[d]:+.2f}, "
                            f"previous record says {float(v):+.2f}")
                else:
                    assert "You took no driver view." in prompt

                trade = prev.get("trade")
                if trade and trade.get("legs"):
                    assert "The position you are carrying:" in prompt, (
                        f"{path} record {i}: carrying legs, block says otherwise")
                elif trade is not None:
                    assert "you chose to be flat" in prompt, (
                        f"{path} record {i}: a chosen flat replayed as something else")
                else:
                    assert "took no position view" in prompt, (
                        f"{path} record {i}: no-trade meeting replayed as a position")
            if not rec.get("degraded"):
                last_good = rec


# ── 3. a carried view is a replay, not a revision ────────────────────────────
_CARRY_INVARIANT = ("direction", "conviction", "report", "falsifier", "key_evidence")


def test_carried_views_are_replays_not_revisions():
    """A ``carried=True`` record must be byte-identical to the previous record's
    view on every judgment field — only ``asof`` and the flag itself may differ
    (``CarryForward`` re-emits via ``model_copy(update={'asof', 'carried'})``).
    A carried view that differs anywhere is a phantom revision: an opinion
    change on a meeting where the analyst was never called.
    """
    runs = _ab_runs()
    if not runs:
        pytest.skip(f"no committed analyst runs under {AB_GLOB}")
    seen_carried = 0
    for path in runs:
        recs = _records(path)
        for i in range(1, len(recs)):
            view, prev = recs[i].get("view") or {}, recs[i - 1].get("view") or {}
            if not view.get("carried"):
                continue
            seen_carried += 1
            for field in _CARRY_INVARIANT:
                assert view.get(field) == prev.get(field), (
                    f"{path} record {i}: carried view revises {field!r} — "
                    f"{prev.get(field)!r} -> {view.get(field)!r}")
    if seen_carried == 0:
        pytest.skip("no carried records in the committed runs (release-clock "
                    "runs carry ~never; this bites on weekly-clock runs)")


# ── 4. a failed call carries nothing scoreable ───────────────────────────────
def test_degraded_pm_meetings_carry_no_scoreable_content():
    """A degraded meeting must have an empty driver block and no trade in the
    artifact. If either survives, the run file scores a meeting the model never
    completed — the substitution ``_degraded`` exists to refuse.
    """
    runs = _pm_runs()
    if not runs:
        pytest.skip(f"no committed PM runs under {PM_GLOB}")
    for path in runs:
        for i, rec in enumerate(_records(path)):
            if not rec.get("degraded"):
                continue
            av = rec.get("arbitrated") or {}
            assert not av.get("drivers"), (
                f"{path} record {i}: degraded meeting has driver convictions")
            assert av.get("trade") is None, (
                f"{path} record {i}: degraded meeting has a trade")


# ── 5. every board leg carries its provenance ────────────────────────────────
def test_every_board_leg_has_its_meta():
    """Each committed board leg needs its ``.meta.json`` with a non-empty
    ``config`` — the identity check (``board.IDENTITY_KEYS``) reads it, and a
    leg without one silently passes identity as all-None, which is how a
    mixed-arm board slips through looking uniform.
    """
    runs = _ab_runs()
    if not runs:
        pytest.skip(f"no committed analyst runs under {AB_GLOB}")
    for path in runs:
        meta = _meta(path)
        assert meta.get("config"), (
            f"{path}: no .meta.json config — the board identity check has "
            f"nothing to verify and will wave this leg through")
