"""Think only when the evidence moves.

The fund's meeting is weekly. The drivers are not: CPI and core PCE release monthly,
FOMC statements eight times a year. So on most meeting dates an analyst's evidence
is character-for-character what it saw the week before.

This wrapper hashes the rendered prompt and, when it matches the previous meeting,
re-emits the previous view marked ``carried`` instead of calling the model.

That is a correctness fix rather than an optimization. Three things go wrong without
it:

  * **Phantom revisions.** At any temperature above zero the same prompt returns a
    different answer, so the analyst appears to change its mind when nothing
    happened. That churn inflates the volatility term in signal Sharpe, shows up as
    turnover, and contaminates cross-agent correlation — noise entering the stream
    dressed as signal.
  * **An inflated sample.** Counting carried views as fresh ones makes a monthly
    driver look like it produced 52 independent opinions a year. It produced twelve.
  * **Wasted spend.** Roughly 24 of 52 weekly calls are redundant for a monthly
    driver.

It implements the same interface as the analyst it wraps — ``driver``, ``inputs``,
``form_view(world)`` — so it is a drop-in ``ViewSource`` and anything that grades
analysts grades this identically.

Degraded views are deliberately never cached: a failed call should be retried at the
next meeting, not frozen and repeated.
"""
from __future__ import annotations

import hashlib

from src.layered.contracts import DriverView


class CarryForward:
    """Wraps an analyst so it forms a view only when its evidence changes."""

    def __init__(self, analyst):
        self.analyst = analyst
        self._last_key: str | None = None
        self._last_view: DriverView | None = None
        self.calls_made = 0
        self.calls_carried = 0

    # ── delegated identity ──────────────────────────────────────────────────
    @property
    def driver(self) -> str:
        return self.analyst.driver

    @property
    def inputs(self) -> tuple[str, ...]:
        return self.analyst.inputs

    @property
    def cues(self) -> list[str]:
        return self.analyst.cues

    def build_inputs(self, world):
        return self.analyst.build_inputs(world)

    # ── the evidence fingerprint ────────────────────────────────────────────
    def _evidence_key(self, features, text) -> str:
        """Hash of exactly what the model would be sent.

        Both prompts are hashed, not just the evidence: a persona edit changes the
        system prompt, and that must invalidate the cache too.
        """
        payload = self.analyst._system_prompt() + "\x00" + self.analyst._user_prompt(features, text)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # ── entry point ─────────────────────────────────────────────────────────
    def form_view(self, world) -> DriverView:
        features, text = self.analyst.build_inputs(world)
        key = self._evidence_key(features, text)

        if key == self._last_key and self._last_view is not None:
            self.calls_carried += 1
            # Stamped with the current meeting — this is the view held today — but
            # flagged, so it is never mistaken for an independent observation. The
            # date it was actually formed is the last preceding view with carried=False.
            return self._last_view.model_copy(update={"asof": world.asof, "carried": True})

        view = self.analyst.form_view_from(features, text)
        self.calls_made += 1
        if not view.degraded:
            self._last_key, self._last_view = key, view
        return view

    @property
    def stats(self) -> dict:
        total = self.calls_made + self.calls_carried
        return {
            "meetings": total,
            "calls_made": self.calls_made,
            "carried": self.calls_carried,
            "carried_share": round(self.calls_carried / total, 3) if total else 0.0,
        }
