"""Score a PM's trade — the weighted sum of yield changes it was promised.

The layer below this one is graded in *driver space*: did the analyst call inflation,
did the PM call it better (``pm_bench``). This module grades the other output the PM
produces, the one that crosses the PM→fund seam — a ``StrategyTrade`` in *instrument
space*. Kept a separate module rather than folded into ``pm_bench`` because the outcome
is a different quantity entirely: ``pm_bench`` compares a signed conviction against the
change in a macro level, this compares signed instrument weights against the change in
a yield.

**The sign convention is not a choice made here.** It was fixed the moment the model was
asked for the trade. The leg schema in ``llm_pm.submit_arbitration_tool`` tells the model:

    "Signed weight on that instrument's YIELD. The trade is scored as the weighted sum
     of yield changes, so a steepener is negative on the short leg and positive on the
     long one."

So the P&L is ``Σ_leg w · Δy`` and a positive weight bets the yield *rises*. This is
deliberately **not** a bond return — a long-duration position earns when yields *fall*,
so the sign here is the opposite of a price-space P&L — and it is not duration-weighted,
so a unit on DGS2 and a unit on DGS10 count equally. The function is named ``yield_pnl``
rather than ``returns`` for exactly that reason: scoring these weights under a price
convention would invert every number while still looking plausible.

The units are percentage points of yield, because that is what the FRED DGS series carry.

``trade_validity`` is the other half and matters as much as the P&L. Every rejection path
in ``llm_pm._parse_trade`` is a silent ``continue`` — an out-of-universe instrument, a
zero weight, a leg count over the pod's limit — and the pods' sign conventions ("both legs
the SAME sign", "EQUAL AND OPPOSITE") are prompt text with no enforcement anywhere. A pod
that emits a trade every month while violating its own mandate half the time is a
different failure from one that abstains, and no P&L number distinguishes them. So the
validity block reads the stored ``raw_response`` alongside the grounded trade and reports
what grounding silently changed.

Nothing here imports ``src.layered.pm``. The pod's ``trade:`` block is passed in as a
plain dict, so this module sits cleanly below the PM layer and can be re-exported from
the package without the circular import that keeps ``pm_bench`` out of it.
"""
from __future__ import annotations

import json
import math
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from src.layered.evaluation.ic import _two_sided_p

# The pod's declared sign convention, read from the ``trade:`` block as
# ``sign_convention: same | opposed``. Absent means the pod does not constrain the
# relationship between legs (``real`` trades a single instrument; ``front_end`` is
# constrained by its universe instead), and no violation can be reported.
_SAME = "same"
_OPPOSED = "opposed"


# ── the raw side: what the model actually said ──────────────────────────────────
def _raw_legs(raw: Optional[str]) -> Optional[list[dict]]:
    """The trade legs as the model emitted them, before grounding.

    Returns ``None`` when the response carried no trade at all — which is a legitimate
    answer, since ``trade`` is not a required field of the tool schema. That is a
    different fact from "emitted a trade that grounding then rejected", and the whole
    point of reading the raw response is to keep the two apart.

    Tolerates the same shapes ``llm_pm._coerce_entries`` does: a list of objects, a
    ``{instrument: weight}`` map, or either delivered as a JSON string.
    """
    if not raw:
        return None
    try:
        parsed = json.loads(raw, strict=False)
    except Exception:  # noqa: BLE001 — a truncated reply is a finding, not a crash
        return None
    block = (parsed or {}).get("trade")
    if not isinstance(block, dict):
        return None
    legs = block.get("legs")
    if isinstance(legs, str):
        try:
            legs = json.loads(legs, strict=False)
        except Exception:  # noqa: BLE001
            return None
    if isinstance(legs, Mapping):
        return [{"instrument": str(k), "weight": v} for k, v in legs.items()]
    if isinstance(legs, list):
        return [x for x in legs if isinstance(x, Mapping)]
    return []


def _sign_violation(legs: Mapping[str, float], convention: str) -> bool:
    """Does this trade contradict the pod's declared leg relationship?

    Only meaningful with two or more legs — a single-leg trade cannot oppose or agree
    with itself, and reporting it as a violation would flag every ``real`` pod trade.
    """
    signs = {int(np.sign(w)) for w in legs.values() if w}
    if len(legs) < 2 or not signs:
        return False
    if convention == _SAME:
        return len(signs) > 1
    if convention == _OPPOSED:
        return len(signs) == 1
    return False


def load_trades(path: str, trade_config: Optional[dict] = None) -> pd.DataFrame:
    """Every meeting in a PM run as one row, trade and diagnosis together.

    One row per meeting including meetings with no trade, so an abstention stays
    visible: dropping them here would silently turn "the PM declined to trade" into
    "the PM was never asked", and the emission rate is one of the numbers we most need.

    Degraded meetings are excluded, on the same principle ``load_pm_run`` drops them —
    a failed call is not an abstention.
    """
    trade_config = trade_config or {}
    universe = {str(s) for s in (trade_config.get("universe") or [])}
    convention = str(trade_config.get("sign_convention", "") or "").strip().lower()

    with open(path) as fh:
        recs = [json.loads(line) for line in fh if line.strip()]
    if not recs:
        raise ValueError(f"{path}: no records")

    rows = []
    for r in recs:
        if r.get("degraded"):
            continue
        av = r.get("arbitrated") or {}
        trade = av.get("trade")
        legs: dict[str, float] = {str(k): float(v) for k, v in
                                  ((trade or {}).get("legs") or {}).items()}
        raw = _raw_legs(r.get("raw_response"))

        # What grounding removed. An instrument outside the universe and a weight of
        # exactly 0.0 are both dropped by `_parse_trade` without a trace; recovering
        # the count from the raw response is the only record that it happened.
        dropped_universe = dropped_zero = 0
        if raw is not None:
            for item in raw:
                name = str(item.get("instrument", "")).strip()
                w = item.get("weight")
                if universe and name not in universe:
                    dropped_universe += 1
                elif isinstance(w, (int, float)) and float(w) == 0.0:
                    dropped_zero += 1

        row: dict[str, Any] = {
            "asof": pd.Timestamp(r["asof"]),
            "emitted": raw is not None,
            "has_trade": trade is not None,
            # A position of nothing, chosen on purpose. It scores as a real zero rather
            # than being dropped: the PM decided to sit out and that decision has an
            # outcome, which is not true of a meeting where it simply never answered.
            "flat": trade is not None and not legs,
            "n_legs": len(legs),
            "gross": float(sum(abs(w) for w in legs.values())),
            "net": float(sum(legs.values())),
            "conviction": float(trade["conviction"]) if trade else np.nan,
            "rationale_words": len(str((trade or {}).get("rationale", "")).split()),
            "legs_dropped_universe": dropped_universe,
            "legs_dropped_zero": dropped_zero,
            "sign_violation": _sign_violation(legs, convention) if trade else False,
            "n_tags": len(((trade or {}).get("risk") or {}).get("tags") or []),
        }
        for inst in sorted(universe) or sorted(legs):
            row[f"w_{inst}"] = legs.get(inst, 0.0) if trade else np.nan
        rows.append(row)

    return pd.DataFrame(rows).set_index("asof").sort_index()


# ── the outcome: what the yields did next ───────────────────────────────────────
def forward_yield_change(macro: Mapping[str, pd.Series], instruments: list[str],
                         dates: pd.DatetimeIndex, steps: int = 1,
                         freq: str = "ME") -> pd.DataFrame:
    """Change in each instrument's yield over the next ``steps`` clock periods.

    The series are resampled to the pod's clock with ``.last()`` before differencing,
    so the level read at a meeting is the last one known by that date — the same "as
    of" reading ``AsOf`` gives, and what lets a daily series be graded on a monthly
    clock without overlapping observations.

    **The forward level comes from the full data grid, not from ``dates``.** Shifting
    within the sample instead would make the last meeting of any run unscoreable, and
    worse, would make a truncated run (``--limit``) silently drop months that the
    vendored CSVs can in fact settle — the P&L of a 6-meeting pilot would rest on 5
    observations for no reason visible in the output.
    """
    out = {}
    for inst in instruments:
        s = macro.get(inst)
        if s is None or s.dropna().empty:
            raise ValueError(f"instrument {inst!r} is not in the macro bundle")
        grid = s.dropna().sort_index().resample(freq).last().dropna()
        if grid.empty:
            raise ValueError(f"instrument {inst!r} has no observations on a {freq!r} clock")

        # The level as of each meeting: last grid point at or before it.
        now = grid.reindex(grid.index.union(dates)).ffill().reindex(dates)

        # The level ``steps`` grid periods later. searchsorted("right") gives the first
        # grid position strictly after the meeting, so +(steps-1) lands on the steps-th.
        pos = grid.index.searchsorted(dates, side="right") + (steps - 1)
        vals = np.full(len(dates), np.nan)
        ok = pos < len(grid)
        vals[ok] = grid.to_numpy()[pos[ok]]
        out[inst] = pd.Series(vals, index=dates) - now
    return pd.DataFrame(out, index=dates)


def yield_pnl(trades: pd.DataFrame, macro: Mapping[str, pd.Series],
              instruments: list[str], steps: int = 1, freq: str = "ME") -> pd.Series:
    """``Σ_leg w · Δy`` per meeting, in percentage points of yield.

    Positive means the trade made money *in yield space* — the weighted yields moved
    the way the weights bet. See the module docstring: this is not a bond return.

    Meetings without a trade produce no observation rather than a zero. A zero would
    say "the PM took a flat position and it paid nothing", which is a claim about a
    decision that was never made, and it would dilute every downstream mean and
    t-statistic with rows carrying no information.
    """
    held = trades[trades["has_trade"]]
    if held.empty:
        return pd.Series(dtype=float)
    dates = pd.DatetimeIndex(held.index)
    fwd = forward_yield_change(macro, instruments, dates, steps=steps, freq=freq)
    w = pd.DataFrame({inst: held.get(f"w_{inst}", 0.0) for inst in instruments},
                     index=dates).fillna(0.0)
    # A meeting whose forward observation does not exist yet (the tail of the window)
    # is dropped, not treated as a zero move.
    return (w * fwd).sum(axis=1, skipna=False).dropna()


def score_trades(pnl: pd.Series, conviction: Optional[pd.Series] = None,
                 periods_per_year: Optional[float] = None) -> dict:
    """Headline statistics for a yield-space P&L series.

    ``t_stat`` is the ordinary one-sample t on the mean, which is legitimate here only
    because the meetings are non-overlapping by construction (month-end clock, one-step
    horizon). Overlap it and this number would need a Newey-West correction.

    ``ic_conviction`` asks whether the PM sized well: the rank correlation between the
    conviction it declared and the P&L it got. Reported separately from the P&L because
    a strategy can be profitable while sizing perversely, and the two failures need
    different fixes.
    """
    pnl = pnl.dropna()
    n = len(pnl)
    out: dict[str, Any] = {"n": n}
    if n < 3:
        return {**out, "mean": float("nan"), "t_stat": float("nan"),
                "p_approx": float("nan"), "hit_rate": float("nan"),
                "sharpe_ann": float("nan"), "ic_conviction": float("nan")}

    mean, sd = float(pnl.mean()), float(pnl.std(ddof=1))
    t = mean / (sd / math.sqrt(n)) if sd > 0 else float("nan")

    if periods_per_year is None:
        gap = float(pd.Series(pnl.index).diff().dt.days.median())
        periods_per_year = 365.25 / gap if gap and gap > 0 else float("nan")

    ic = float("nan")
    if conviction is not None:
        pair = pd.concat([conviction.rename("c"), pnl.rename("p")], axis=1).dropna()
        if len(pair) >= 3 and pair["c"].nunique() > 1 and pair["p"].nunique() > 1:
            ic = float(pair["c"].rank().corr(pair["p"].rank()))

    return {
        **out,
        "mean": mean,
        "std": sd,
        "t_stat": t,
        "p_approx": _two_sided_p(t),
        "hit_rate": float((pnl > 0).mean()),
        "sharpe_ann": (mean / sd * math.sqrt(periods_per_year)
                       if sd > 0 and periods_per_year == periods_per_year else float("nan")),
        "periods_per_year": periods_per_year,
        "ic_conviction": ic,
    }


def trade_validity(trades: pd.DataFrame) -> dict:
    """How often the trade path did what the pod's mandate says it should.

    Rates rather than counts, so runs of different lengths compare directly. Read
    ``sign_violation_rate`` first: it is the only mandate constraint with no structural
    defence anywhere — the tool enum pins the instrument universe and ``_parse_trade``
    pins the leg count, but nothing in the codebase inspects how the leg signs relate,
    so a curve-shaped trade stored under a duration pod passes every check.
    """
    n = len(trades)
    if not n:
        return {"n": 0}
    held = trades[trades["has_trade"]]
    return {
        "n": n,
        "emitted_rate": float(trades["emitted"].mean()),
        "grounded_rate": float(trades["has_trade"].mean()),
        # Emitted a trade that grounding then threw away — the failure mode that looks
        # identical to an abstention in the saved contract.
        "rejected_rate": float((trades["emitted"] & ~trades["has_trade"]).mean()),
        # A deliberate flat, distinct from never answering. Watch this next to
        # `grounded_rate`: a PM carrying memory can rationally choose to sit out, and
        # before ``flat`` existed that choice was indistinguishable from silence.
        "flat_rate": float(held["flat"].mean()) if len(held) else float("nan"),
        "sign_violation_rate": float(held["sign_violation"].mean()) if len(held) else float("nan"),
        "legs_dropped_universe": int(trades["legs_dropped_universe"].sum()),
        "legs_dropped_zero": int(trades["legs_dropped_zero"].sum()),
        "mean_n_legs": float(held["n_legs"].mean()) if len(held) else float("nan"),
        "mean_gross": float(held["gross"].mean()) if len(held) else float("nan"),
        "mean_conviction": float(held["conviction"].mean()) if len(held) else float("nan"),
        "mean_rationale_words": float(held["rationale_words"].mean()) if len(held) else float("nan"),
        "no_rationale_rate": float((held["rationale_words"] == 0).mean()) if len(held) else float("nan"),
    }


def summarize(score: dict, validity: dict, units: str = "pp of yield") -> str:
    """The two blocks as one printable paragraph, for the end of a run."""
    lines = [
        f"trades: {validity.get('grounded_rate', float('nan')):.0%} of "
        f"{validity.get('n', 0)} meetings produced a trade "
        f"(emitted {validity.get('emitted_rate', float('nan')):.0%}, "
        f"rejected {validity.get('rejected_rate', float('nan')):.0%})",
        f"mandate: sign violations {validity.get('sign_violation_rate', float('nan')):.0%}, "
        f"legs dropped {validity.get('legs_dropped_universe', 0)} out-of-universe / "
        f"{validity.get('legs_dropped_zero', 0)} zero-weight",
        f"yield P&L: n={score.get('n', 0)} mean={score.get('mean', float('nan')):+.4f} {units} "
        f"t={score.get('t_stat', float('nan')):+.2f} "
        f"hit={score.get('hit_rate', float('nan')):.3f} "
        f"sharpe={score.get('sharpe_ann', float('nan')):+.2f}",
        f"sizing: IC(conviction, P&L) = {score.get('ic_conviction', float('nan')):+.3f}",
    ]
    return "\n".join(lines)
