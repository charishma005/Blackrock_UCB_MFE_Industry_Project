"""The feature spec — what an analyst is permitted to notice, as config.

The spec fixes which measurements reach an analyst, which makes it a first-class
design object rather than a helper: an analyst cannot reason about something it
was never shown. Keeping it in the persona YAML alongside the mandate follows the
precedent already set by the PM's transmission map — the parts of the system that
encode a judgment call live as config, where they are legible and reviewable
without reading code.

A definition names an operation from the closed vocabulary in ``ops.py`` and its
inputs. An input is either a raw series id (``CPIAUCSL``) or a reference to an
earlier definition (``@headline_cpi_yoy``), so derived measurements compose
without any of them becoming a fitted quantity.
"""
from __future__ import annotations

from dataclasses import dataclass, field

DERIVED_PREFIX = "@"


@dataclass(frozen=True)
class FeatureDef:
    name: str
    op: str
    sources: tuple[str, ...]           # raw series ids, or "@earlier_feature"
    params: dict = field(default_factory=dict)
    unit: str = ""
    history: int = 1                   # >1 ⇒ rendered as a trajectory
    description: str = ""              # what the feature IS — construction only, never its meaning

    @property
    def raw_sources(self) -> tuple[str, ...]:
        return tuple(s for s in self.sources if not s.startswith(DERIVED_PREFIX))


@dataclass(frozen=True)
class FeatureSpec:
    driver: str
    series: tuple[FeatureDef, ...] = ()
    scalars: tuple[FeatureDef, ...] = ()
    level_feature: str | None = None

    @property
    def definitions(self) -> tuple[FeatureDef, ...]:
        """Series first — scalars may reference them by ``@name``."""
        return self.series + self.scalars

    @property
    def declared_inputs(self) -> tuple[str, ...]:
        """Every raw series this spec is allowed to read. The isolation contract."""
        seen: list[str] = []
        for d in self.definitions:
            for s in d.raw_sources:
                if s not in seen:
                    seen.append(s)
        return tuple(seen)


_RESERVED = {"name", "op", "source", "sources", "unit", "history", "description"}


def _parse_def(raw: dict, default_history: int) -> FeatureDef:
    if "name" not in raw or "op" not in raw:
        raise ValueError(f"feature definition needs 'name' and 'op': {raw!r}")
    if "sources" in raw:
        sources = tuple(raw["sources"])
    elif "source" in raw:
        sources = (raw["source"],)
    else:
        raise ValueError(f"feature {raw['name']!r} needs 'source' or 'sources'")
    params = {k: v for k, v in raw.items() if k not in _RESERVED}
    return FeatureDef(
        name=raw["name"],
        op=raw["op"],
        sources=sources,
        params=params,
        unit=raw.get("unit", ""),
        history=int(raw.get("history", default_history)),
        description=raw.get("description", ""),
    )


def from_persona(driver: str, persona: dict) -> FeatureSpec:
    """Build a spec from a persona YAML's ``features:`` block."""
    block = (persona or {}).get("features") or {}
    series = tuple(_parse_def(d, default_history=13) for d in block.get("series", []))
    scalars = tuple(_parse_def(d, default_history=1) for d in block.get("scalars", []))
    spec = FeatureSpec(
        driver=driver,
        series=series,
        scalars=scalars,
        level_feature=block.get("level_feature"),
    )
    names = [d.name for d in spec.definitions]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise ValueError(f"{driver}: duplicate feature names {sorted(dupes)}")
    if spec.level_feature and spec.level_feature not in names:
        raise ValueError(
            f"{driver}: level_feature {spec.level_feature!r} is not a defined feature ({names})"
        )
    return spec
