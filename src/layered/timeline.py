"""Time integrity — no agent ever sees data it could not have had.

The thesis raises this as a first-class question: "How is the integrity of time
protected end to end, so that no agent — analyst or PM — is ever allowed to act
on information it could not have had at the moment it is supposed to be acting?"

The answer here is a single choke point. Every observable an analyst reads goes
through ``AsOf.series`` / ``AsOf.frame``, which slice strictly to ``<= asof``.
Analysts are handed an ``AsOf`` rather than raw data, so there is one place to
audit for look-ahead, and an analyst physically cannot reach past its own clock.

(Publication-lag correction — that CPI stamped 2024-03-01 wasn't released until
mid-April — is handled upstream in ``src/data/markets.py`` by shifting each FRED
series to its release date. This gate then protects the *slice*; the two
together are the end-to-end guarantee.)
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class AsOf:
    """A frozen view of the world as of one instant.

    ``macro`` is a bundle of FRED-style series; ``prices`` is a (date x symbol)
    frame. Both are sliced to ``<= asof`` on every access — callers never touch
    the raw, full-history objects, so nothing downstream can leak the future.
    """

    asof: pd.Timestamp
    macro: dict[str, pd.Series]
    prices: pd.DataFrame

    def series(self, series_id: str) -> pd.Series:
        """One macro series, truncated to what was known at ``asof``."""
        s = self.macro.get(series_id)
        if s is None:
            return pd.Series(dtype=float)
        return s.loc[: self.asof]

    def price(self, symbol: str) -> pd.Series:
        """One instrument's price history, truncated to ``asof``."""
        if self.prices is None or symbol not in self.prices.columns:
            return pd.Series(dtype=float)
        return self.prices[symbol].loc[: self.asof].dropna()

    def frame(self, symbols: list[str] | None = None) -> pd.DataFrame:
        """The price frame (optionally column-subset), truncated to ``asof``."""
        if self.prices is None:
            return pd.DataFrame()
        cols = [c for c in (symbols or self.prices.columns) if c in self.prices.columns]
        return self.prices[cols].loc[: self.asof]

    @classmethod
    def build(
        cls,
        asof: str | pd.Timestamp,
        macro: dict[str, pd.Series] | None = None,
        prices: pd.DataFrame | None = None,
    ) -> "AsOf":
        return cls(
            asof=pd.Timestamp(asof),
            macro=macro or {},
            prices=prices if prices is not None else pd.DataFrame(),
        )
