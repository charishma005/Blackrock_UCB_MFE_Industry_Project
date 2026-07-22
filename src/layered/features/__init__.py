"""Measurement layer — engineered features, strictly no signals.

    spec.py    what an analyst is permitted to notice, loaded from persona YAML
    ops.py     the closed vocabulary of operations a spec may use
    engine.py  evaluates a spec through the AsOf gate into a FeatureSet
"""
from src.layered.features.engine import FeatureEngine
from src.layered.features.spec import FeatureSpec, from_persona

__all__ = ["FeatureEngine", "FeatureSpec", "from_persona"]
