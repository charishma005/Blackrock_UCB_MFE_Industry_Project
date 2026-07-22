"""Turn a feature spec into measurements, through the AsOf gate.

One choke point. Every observable an analyst ever sees is computed here, from the
spec's declared sources, read through ``AsOf`` (which slices to ``<= asof``). That
makes input isolation a property of a single class rather than something each
analyst subclass has to be trusted to get right — and it is the reason the seven
hand-written analyst classes can collapse into seven config files.

The engine performs no judgment. It resolves sources, applies operations from the
closed vocabulary, and records which raw series it touched so the isolation audit
has something to check.
"""
from __future__ import annotations

import pandas as pd

from src.layered.contracts import FeatureSet, ScalarFeature, SeriesFeature
from src.layered.features import ops
from src.layered.features.spec import DERIVED_PREFIX, FeatureDef, FeatureSpec


class FeatureEngine:
    """Computes one driver's ``FeatureSet`` from its spec."""

    def __init__(self, spec: FeatureSpec):
        self.spec = spec

    @property
    def inputs(self) -> tuple[str, ...]:
        """Raw series this engine may read — the analyst's isolation contract."""
        return self.spec.declared_inputs

    def compute(self, world) -> FeatureSet:
        """Evaluate every definition. ``world`` is an ``AsOf`` (or any duck-type
        exposing ``series(id)``), so nothing dated after ``asof`` can enter."""
        cache: dict[str, pd.Series] = {}
        read: list[str] = []

        def evaluate(d: FeatureDef) -> pd.Series:
            inputs: list[pd.Series] = []
            for src in d.sources:
                if src.startswith(DERIVED_PREFIX):
                    key = src[len(DERIVED_PREFIX):]
                    if key not in cache:
                        raise ValueError(
                            f"{self.spec.driver}/{d.name}: references {src!r}, which is not "
                            f"defined earlier in the spec (available: {sorted(cache)})"
                        )
                    inputs.append(cache[key])
                else:
                    if src not in read:
                        read.append(src)
                    inputs.append(world.series(src))
            try:
                out = ops.apply(d.op, inputs, dict(d.params))
            except Exception as e:  # noqa: BLE001 — name the offending feature
                raise ValueError(f"{self.spec.driver}/{d.name}: {e}") from e
            cache[d.name] = out
            return out

        series_out: list[SeriesFeature] = []
        for d in self.spec.series:
            clean = evaluate(d).dropna()
            if clean.empty:
                continue          # not enough history yet at this asof — omit rather than invent
            series_out.append(SeriesFeature(
                name=d.name,
                values=[float(v) for v in clean.tail(d.history)],
                unit=d.unit,
                description=d.description,
            ))

        scalar_out: list[ScalarFeature] = []
        for d in self.spec.scalars:
            clean = evaluate(d).dropna()
            if clean.empty:
                continue
            scalar_out.append(ScalarFeature(
                name=d.name, value=float(clean.iloc[-1]), unit=d.unit,
                description=d.description,
            ))

        return FeatureSet(
            driver=self.spec.driver,
            asof=world.asof,
            series=series_out,
            scalars=scalar_out,
            level_feature=self.spec.level_feature,
            sources_read=read,
        )
