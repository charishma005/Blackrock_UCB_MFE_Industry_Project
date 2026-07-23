"""Information coefficient on the driver's own release clock.

The analyst's prediction is fixed at **the next release** — for inflation, the next
CPI print. That choice does more statistical work than it appears to.

Grading a fixed calendar horizon (say 63 days) on a weekly schedule makes
consecutive observations share almost their entire outcome window: a quarterly
horizon sampled weekly overlaps by roughly twelve weeks in thirteen. Errors that
autocorrelated make a naive t-statistic badly overstated, and correcting it needs
Newey-West, a block bootstrap, or discarding most of the sample. Release-to-release
changes have none of that — each outcome window ends exactly where the next begins,
so observations are **non-overlapping by construction** and the t-statistic is
honest untouched. The sample is then simply what it always was, about twelve a
year; the weekly schedule only ever made it look larger.

Rank correlation rather than linear, because conviction calibration is itself
untested: a signal that orders outcomes correctly should not be penalised for being
badly scaled. Whether the scaling is any good is a separate question, answered by
``calibration_split``.

What "good" means here is set by breadth. Under the fundamental law
``IR ≈ IC · √breadth``, and breadth is fixed by the design at one bet per release —
about twelve a year. So an IR of 1.0 needs an IC near 0.29, where a cross-sectional
equity book making hundreds of bets is content with 0.05. A real but small IC on a
single driver is not worth much on its own; breadth has to come from having many
weakly-skilled *independent* analysts, which is what makes the correlation
diagnostic load-bearing rather than decorative.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


def _two_sided_p(t: float) -> float:
    """Normal approximation to the two-sided tail — scipy is not a dependency.

    At n ≈ 70 the gap to the exact t-distribution is small, but it *is* an
    approximation and the column is named accordingly. Read the t-statistic.
    """
    if not np.isfinite(t):
        return float("nan")
    return math.erfc(abs(t) / math.sqrt(2.0))


def required_ic(target_ir: float, breadth: float) -> float:
    """IC needed for a given IR at a given breadth — the bar, stated in advance."""
    return target_ir / math.sqrt(breadth) if breadth > 0 else float("nan")


@dataclass(frozen=True)
class ICResult:
    name: str
    n: int                 # non-overlapping observations actually scored
    ic: float              # Spearman rank correlation with the next-release move
    t_stat: float
    p_approx: float
    hit_rate: float        # sign agreement; see the caveat in ``evaluate``

    def as_row(self) -> dict:
        r = lambda x, d: (round(x, d) if x == x else float("nan"))  # noqa: E731
        return {"signal": self.name, "n": self.n, "ic": r(self.ic, 3),
                "t_stat": r(self.t_stat, 2), "p_approx": r(self.p_approx, 4),
                "hit_rate": r(self.hit_rate, 3)}


class ICEvaluator:
    """Scores any signal against the driver's move over the next ``steps`` releases.

    ``level`` is the driver's measured level on its release clock — for inflation,
    headline CPI year-over-year, one observation per print. Anything producing a
    number per release can be scored: a raw feature, a rule, or an analyst's signed
    conviction. Keeping the evaluator ignorant of what produced the signal is what
    makes those comparisons apples-to-apples rather than a matter of discipline.
    """

    def __init__(self, level: pd.Series, steps: int = 1):
        if steps < 1:
            raise ValueError("steps must be >= 1")
        self.level = level.dropna().sort_index()
        self.steps = steps

    # ── the target ──────────────────────────────────────────────────────────
    @property
    def outcome(self) -> pd.Series:
        """Change in the driver's level over the next ``steps`` releases.

        Positive = the driver rose (inflation accelerated). This is what a
        direction call is graded against.
        """
        return (self.level.shift(-self.steps) - self.level).dropna()

    @property
    def releases_per_year(self) -> float:
        """Inferred from the clock's own spacing — never assumed."""
        if len(self.level) < 3:
            return float("nan")
        gap = float(pd.Series(self.level.index).diff().dt.days.median())
        return 365.25 / gap if gap > 0 else float("nan")

    @property
    def breadth(self) -> float:
        """Independent bets per year — one per release, by design."""
        return self.releases_per_year / self.steps

    # ── primary metric ──────────────────────────────────────────────────────
    def evaluate(self, signal: pd.Series, name: str = "signal") -> ICResult:
        aligned = pd.concat([signal.rename("s"), self.outcome.rename("y")], axis=1).dropna()
        n = len(aligned)
        if n < 3 or aligned["s"].nunique() < 2 or aligned["y"].nunique() < 2:
            nan = float("nan")
            return ICResult(name, n, nan, nan, nan, nan)

        # Spearman is Pearson on ranks. Computing it that way avoids pandas'
        # method="spearman", which imports scipy — not a dependency of this repo.
        # pandas' default average-rank tie handling matches scipy's.
        ic = float(aligned["s"].rank().corr(aligned["y"].rank()))
        if np.isfinite(ic) and abs(ic) < 1.0:
            t = ic * math.sqrt((n - 2) / (1.0 - ic * ic))
        else:
            t = float("nan")

        # Sign agreement. Meaningful only for a signal with a natural zero (a change,
        # a spread, a signed conviction). For a level such as `headline_yoy_12m_low`
        # the sign never varies and this number says nothing — read `ic` there.
        nz = aligned[(aligned["s"] != 0) & (aligned["y"] != 0)]
        hit = float((np.sign(nz["s"]) == np.sign(nz["y"])).mean()) if len(nz) else float("nan")

        return ICResult(name, n, ic, t, _two_sided_p(t), hit)

    def evaluate_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Score every column, strongest |IC| first."""
        rows = [self.evaluate(frame[c], c).as_row() for c in frame.columns]
        out = pd.DataFrame(rows).set_index("signal")
        return out.reindex(out["ic"].abs().sort_values(ascending=False).index)

    # ── is the conviction doing any work? ───────────────────────────────────
    def calibration_split(self, signed: pd.Series) -> pd.DataFrame:
        """IC of the direction alone against IC of the signed conviction.

        A cleaner read on calibration than a Sharpe ratio, and it needs no
        annualization of a quantity that is not a return. If the two are close, the
        conviction is carrying no ordering information and the calibration ladder in
        the prompt is not working. If the signed version scores materially higher,
        the sizing is real.
        """
        rows = [
            self.evaluate(np.sign(signed), "direction_only").as_row(),
            self.evaluate(signed, "signed_conviction").as_row(),
        ]
        out = pd.DataFrame(rows).set_index("signal")
        out["ic_from_conviction"] = round(
            float(out.loc["signed_conviction", "ic"] - out.loc["direction_only", "ic"]), 3
        )
        return out

    # ── secondary: sizing-sensitive, and explicitly not a P&L ───────────────
    def signal_sharpe(self, signed: pd.Series, periods_per_year: float | None = None) -> dict:
        """Sharpe of ``signed_conviction × next-release move``.

        Secondary to IC, and NOT tradable: the analyst predicts inflation while the
        fund earns from rates instruments, and nothing here becomes P&L until a PM
        performs the transmission. It is reported because it is sensitive to sizing
        in a way rank IC is not.

        Annualization is taken from the clock's real spacing rather than assumed.
        The legacy ``diagnostics.signal_sharpe`` hardcodes 52, which was right for a
        weekly meeting schedule and overstates a release-clock number by roughly
        ``sqrt(52/12) ≈ 2.08``.
        """
        ppy = periods_per_year if periods_per_year is not None else self.releases_per_year
        pnl = (signed * self.outcome).dropna()
        sd = float(pnl.std())
        mean = float(pnl.mean()) if len(pnl) else float("nan")
        sharpe = (math.sqrt(ppy) * mean / sd) if (len(pnl) >= 2 and sd > 0 and ppy == ppy) else float("nan")
        return {"n": int(len(pnl)), "periods_per_year": round(ppy, 2) if ppy == ppy else float("nan"),
                "mean": mean, "vol": sd, "sharpe": sharpe}
