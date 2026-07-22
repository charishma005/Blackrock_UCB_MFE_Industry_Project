"""PM pods — the LLM synthesis layer between analysts and the mechanical book.

    analysts (DriverView) → [ PM pods ] → agent_signals dict → ensemble → risk

Design (from the architecture slide + the "PMs to model" note):

  Each pod is a portfolio manager that *chooses which analysts it listens to*,
  reads their driver views, and expresses instrument-space calls. From the
  ensemble's point of view a pod IS an "agent": the ensemble weights, scores,
  and fires PODS exactly the way it used to weight investor agents. So a pod's
  output is the same shape the whole downstream tail already consumes:

      {symbol: {"signal": "bullish|bearish|neutral", "confidence": 0-100,
                "reasoning": str}}

  and the engine keys those by pod name into

      dict[pod_name -> {symbol -> {...}}]

The three pods per the design note:

  * relative_value  — long/short across bonds, equities, commodity (e.g. long US
    rates, short gold). The cross-asset RV book.
  * equities_topdown — equities only, top-down by sector/index.
  * trend_follower  — macro trend continuation across bonds, equities, commodity.

──────────────────────────────────────────────────────────────────────────────
STATUS: DUMMY PLACEHOLDERS. Real pods will be LLMs that read the analyst reports
and reason to a trade. For now each pod returns NEUTRAL / 0% confidence so the
full pipeline (ensemble → risk → backtest) runs end-to-end and can be validated
as plumbing without any API key. Replace ``PMPod.form_signals`` with the real
synthesis when ready; nothing downstream needs to change, because the OUTPUT
CONTRACT below is already what the ensemble expects.
──────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.instruments import AssetClass, Instrument

# The instrument-space signal vocabulary the ensemble's encode_signal understands.
Signal = str  # "bullish" | "bearish" | "neutral"


@dataclass
class PMPod:
    """One portfolio-manager pod.

    ``name``            the key it appears under in the agent_signals dict.
    ``asset_classes``   which tradeable buckets this pod is allowed to hold.
    ``listens_to``      which analyst drivers it reads (the "chooses its
                        analysts" edge in the diagram). Empty = listens to all.
    """

    name: str
    asset_classes: tuple[AssetClass, ...]
    listens_to: tuple[str, ...] = ()

    def tradeable_symbols(self, universe: list[Instrument]) -> list[str]:
        return [
            i.symbol for i in universe
            if i.tradeable and i.asset_class in self.asset_classes
        ]

    def form_signals(
        self,
        analyst_views: dict[str, dict],
        universe: list[Instrument],
    ) -> dict[str, dict]:
        """Return {symbol: {signal, confidence, reasoning}} for this pod.

        DUMMY: emits neutral / 0% for every instrument in the pod's asset
        classes. ``analyst_views`` (driver -> DriverView-like dict) is accepted
        now so the wiring is real; the real pod will reason over it.
        """
        _ = analyst_views  # placeholder — real pod reasons over the driver views
        return {
            sym: {
                "signal": "neutral",
                "confidence": 0.0,
                "reasoning": f"[dummy {self.name} pod] no view yet — placeholder.",
            }
            for sym in self.tradeable_symbols(universe)
        }


@dataclass
class PMPods:
    """The three pods, and the routing that turns analyst views into the
    agent_signals dict keyed by pod name."""

    pods: list[PMPod] = field(default_factory=lambda: [
        PMPod(
            "relative_value",
            asset_classes=(AssetClass.RATES, AssetClass.EQUITY, AssetClass.COMMODITY),
            listens_to=("inflation", "curve_slope", "term_premium", "financial_conditions"),
        ),
        PMPod(
            "equities_topdown",
            asset_classes=(AssetClass.EQUITY,),
            listens_to=("sector_breadth", "vol_regime", "risk_appetite"),
        ),
        PMPod(
            "trend_follower",
            asset_classes=(AssetClass.RATES, AssetClass.EQUITY, AssetClass.COMMODITY),
            listens_to=("positioning", "vol_regime", "inflation"),
        ),
    ])

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.pods]

    def run(
        self,
        analyst_views: dict[str, dict],
        universe: list[Instrument],
    ) -> dict[str, dict[str, dict]]:
        """analyst views -> {pod_name: {symbol: {signal, confidence, reasoning}}}.

        This is the exact shape the ensemble/attribution/risk/backtest layers
        consume; each pod is treated as one "agent" by the ensemble.
        """
        return {
            pod.name: pod.form_signals(
                {d: v for d, v in analyst_views.items()
                 if not pod.listens_to or d in pod.listens_to},
                universe,
            )
            for pod in self.pods
        }
