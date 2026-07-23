"""The perturbation arms — offline, no model call.

Two guarantees matter most and are asserted throughout: a perturbation is *pure* (the
caller's original object is never mutated, so scoring against a clean outcome stays
possible), and the *off* arm reproduces the shipped prompt byte-for-byte. The behavioural
question — does the model follow the perturbed evidence — needs a real model and lives in
the run scripts; here a stub stands in wherever a call is unavoidable.
"""
from __future__ import annotations

import json
import re

import pandas as pd

from src.data.fred_local import load_bundle
from src.layered.analysts.llm_analyst import LLMAnalyst
from src.layered.contracts import (DriverView, FeatureSet, ScalarFeature,
                                    SeriesFeature)
from src.layered.perturb import (IDENTITY, CounterfactualPath, ReorderFeatureLines,
                                 RescaleFeatures, RewordScaffolding, ShiftLevel,
                                 SignFlipMomentum, WhitespaceVariant,
                                 analyst_perturbation)
from src.layered.perturb.brief import ScrambleReports, pm_perturbation
from src.layered.pm.board import ViewBoard
from src.layered.pm.brief import render_brief
from src.layered.timeline import AsOf

# The report-prose date guardrails, copied from test_pm_prompt_guardrails so a
# perturbation cannot smuggle a date past the scrubber.
_DATE = re.compile(r"(?<![\d+\-.,])(?:19|20)\d{2}(?![\d.,])")
_MONTH = re.compile(r"\b(?:January|February|March|April|June|July|August|September|"
                    r"October|November|December)\b", re.IGNORECASE)


def _features() -> FeatureSet:
    return FeatureSet(
        driver="inflation", asof=pd.Timestamp("2023-06-01"),
        series=[SeriesFeature(name="headline_cpi_yoy", values=[2.0, 2.4, 3.0], unit="%")],
        scalars=[ScalarFeature(name="headline_yoy_change_3m", value=0.5, unit="pp"),
                 ScalarFeature(name="headline_mom", value=0.2, unit="pp")],
        level_feature="headline_cpi_yoy",
    )


# ── purity: never mutate the caller's object ────────────────────────────────────
def test_feature_perturbations_do_not_mutate_the_original():
    fs = _features()
    before = fs.model_dump()
    for name in ("rescale", "shift_level", "signflip_momentum", "counterfactual_path",
                 "reorder_features"):
        analyst_perturbation(name).apply_features(fs)
    assert fs.model_dump() == before


# ── leak probes (arm A) ─────────────────────────────────────────────────────────
def test_signflip_negates_changes_but_never_the_level():
    out = SignFlipMomentum().apply_features(_features())
    assert out.level == 3.0                                   # level held
    by = {f.name: f.value for f in out.scalars}
    assert by["headline_yoy_change_3m"] == -0.5 and by["headline_mom"] == -0.2


def test_counterfactual_path_reverses_the_trajectory():
    out = CounterfactualPath().apply_features(_features())
    assert out.series[0].values == [3.0, 2.4, 2.0]
    assert out.level == 2.0                                   # last value is now the old first
    # the momentum scalars flip with the reversed path, so the block does not contradict
    # itself (a falling series beside a +momentum reading)
    by = {f.name: f.value for f in out.scalars}
    assert by["headline_yoy_change_3m"] == -0.5 and by["headline_mom"] == -0.2


def _labor_features() -> FeatureSet:
    """A labor-like block: a level, a genuine change, and ``sahm_gap`` — a *level-space*
    spread (unemployment above its 12m low) whose name only looks momentum-ish."""
    return FeatureSet(
        driver="labor_tightness", asof=pd.Timestamp("2023-06-01"),
        series=[SeriesFeature(name="unemployment_rate", values=[3.5, 3.7, 3.9], unit="%")],
        scalars=[ScalarFeature(name="unrate_change_3m", value=0.4, unit="pp"),
                 ScalarFeature(name="sahm_gap", value=0.3, unit="pp")],
        level_feature="unemployment_rate",
    )


def test_signflip_leaves_a_level_space_gap_alone():
    """Regression: ``sahm_gap`` is a level-space spread, not momentum. It must not be
    negated (that would corrupt a level while claiming to hold levels), while the genuine
    change beside it is flipped and the level is untouched."""
    out = SignFlipMomentum().apply_features(_labor_features())
    by = {f.name: f.value for f in out.scalars}
    assert by["unrate_change_3m"] == -0.4                     # genuine change: flipped
    assert by["sahm_gap"] == 0.3                              # level-space spread: held
    assert out.level == 3.9                                   # level held


def test_signflip_catches_annualized_rates():
    """An annualized short-horizon rate is a rate-of-change reading and must flip — these
    were silently missed by the pre-fix token set."""
    fs = FeatureSet(
        driver="inflation", asof=pd.Timestamp("2023-06-01"),
        series=[SeriesFeature(name="headline_cpi_yoy", values=[2.0, 2.4, 3.0], unit="%")],
        scalars=[ScalarFeature(name="headline_3m_annualized", value=0.6, unit="%")],
        level_feature="headline_cpi_yoy",
    )
    out = SignFlipMomentum().apply_features(fs)
    assert {f.name: f.value for f in out.scalars}["headline_3m_annualized"] == -0.6


# op families that are unambiguously momentum vs level *regardless of derivation*.
# Excluded on purpose: spread/ratio/moving_average/lag — a spread of two levels is a
# level (``sahm_gap``) but a spread of two momenta is momentum (``mom_gap_vs_outgoing``),
# so their nature depends on their inputs and cannot be asserted from the op alone.
_MOMENTUM_OPS = {"diff", "pct_change", "pct_change_annualized"}
_LEVEL_OPS = {"level", "yoy", "rolling_min", "rolling_max", "distance_from_reference"}


def test_change_token_set_matches_the_persona_namespace():
    """The safety net for the token approach: over every shipped persona, a feature whose
    op is unambiguously a rate-of-change must be flipped, and one that is unambiguously a
    level/range must not. This is what makes the free-text tokens safe as personas grow —
    it would have caught the annualized miss and the ``sahm_gap`` over-match at their op."""
    import glob
    from pathlib import Path

    import yaml as _yaml

    from src.layered.features import from_persona
    from src.layered.perturb.features import _is_change

    persona_dir = Path("src/layered/analysts/personas")
    paths = [p for p in glob.glob(str(persona_dir / "*.yaml")) if "_TEMPLATE" not in p]
    assert paths, "no personas found"
    for path in paths:
        driver = Path(path).stem
        spec = from_persona(driver, _yaml.safe_load(Path(path).read_text()) or {})
        for d in spec.definitions:
            if d.op in _MOMENTUM_OPS:
                assert _is_change(d.name), f"{driver}:{d.name} ({d.op}) should be flipped"
            elif d.op in _LEVEL_OPS and d.name != spec.level_feature:
                # Vendored precomputed changes (EQ_*_CHG*, upstream-computed
                # diffs read with op: level) ARE changes — the name-based flip
                # is semantically right for them, so they are exempt from the
                # level-must-not-flip rule.
                vendored_change = any("_CHG" in s for s in (d.sources or ()))
                assert vendored_change or not _is_change(d.name), \
                    f"{driver}:{d.name} ({d.op}) must not flip"


def test_rescale_scales_every_measurement():
    out = RescaleFeatures(2.0).apply_features(_features())
    assert out.level == 6.0 and out.scalars[0].value == 1.0


def test_shift_level_moves_only_the_level_feature():
    out = ShiftLevel(1.0).apply_features(_features())
    assert out.level == 4.0                                   # 3.0 + 1
    assert {f.value for f in out.scalars} == {0.5, 0.2}       # non-level untouched


# ── meaning-preserving (arm C) ──────────────────────────────────────────────────
def test_reorder_features_holds_content_and_level():
    out = ReorderFeatureLines().apply_features(_features())
    assert out.level == 3.0                                   # level resolves by name
    assert {f.name for f in out.scalars} == {"headline_yoy_change_3m", "headline_mom"}


def test_string_perturbations_change_bytes_not_a_date():
    prompt = "Driver: inflation\n\nheadline_cpi_yoy (%) — last 3 observations, oldest → newest"
    for p in (WhitespaceVariant(), RewordScaffolding()):
        out = p.apply_prompt(prompt)
        assert out != prompt
        assert not _DATE.search(out) and not _MONTH.search(out)


# ── the no-op guarantee ─────────────────────────────────────────────────────────
def test_identity_is_a_true_noop():
    fs = _features()
    assert IDENTITY.apply_features(fs) is fs
    assert IDENTITY.apply_prompt("x\n\ny") == "x\n\ny"


def test_unknown_name_raises():
    for resolver, name in ((analyst_perturbation, "nope"), (pm_perturbation, "nope")):
        try:
            resolver(name)
            assert False, "expected ValueError"
        except ValueError:
            pass
    assert analyst_perturbation(None) is None and pm_perturbation(None) is None


# ── wired through the analyst, no call ──────────────────────────────────────────
def _world():
    macro = load_bundle(["CPIAUCSL", "PCEPILFE"])
    return AsOf(asof=pd.Timestamp("2018-06-30"), macro=macro, prices=pd.DataFrame())


def test_off_arm_reproduces_the_prompt_byte_for_byte():
    world = _world()
    clean = LLMAnalyst.from_persona("inflation", text_selector=None)
    noop = LLMAnalyst.from_persona("inflation", text_selector=None, perturbation=IDENTITY)
    fc, tc = clean.build_inputs(world)
    fn, tn = noop.build_inputs(world)
    assert clean._user_prompt(fc, tc) == noop._user_prompt(fn, tn)


def test_perturbation_changes_the_rendered_prompt():
    world = _world()
    clean = LLMAnalyst.from_persona("inflation", text_selector=None)
    flipped = LLMAnalyst.from_persona("inflation", text_selector=None,
                                      perturbation=SignFlipMomentum())
    fc, tc = clean.build_inputs(world)
    ff, tf = flipped.build_inputs(world)
    assert clean._user_prompt(fc, tc) != flipped._user_prompt(ff, tf)
    # and the perturbed prompt still leaks no date
    assert not _DATE.search(flipped._user_prompt(ff, tf))


# ── the PM scramble (arm B) ─────────────────────────────────────────────────────
def _view(driver, report, direction="up"):
    return DriverView(driver=driver, asof=pd.Timestamp("2023-06-30"), direction=direction,
                      conviction=0.5, horizon_days=31, level=1.0, report=report)


def _board():
    return ViewBoard({
        "inflation": [_view("inflation", "prices accelerating")],
        "labor_tightness": [_view("labor_tightness", "unemployment rising", "down")],
        "term_premium": [_view("term_premium", "long end cheapening")],
    })


def test_scramble_is_a_derangement_and_pure():
    m = _board().at("2023-06-30")
    original = {d: e.view.report for d, e in m.entries.items()}
    scrambled = ScrambleReports().apply_meeting(m)
    # keys preserved, original untouched
    assert list(scrambled.entries) == list(m.entries)
    assert {d: e.view.report for d, e in m.entries.items()} == original
    # no slot keeps its own report
    for d, e in scrambled.entries.items():
        assert e.view.report != original[d]


def test_scramble_noop_below_two_present_drivers():
    b = ViewBoard({"inflation": [_view("inflation", "solo")]})
    m = b.at("2023-06-30")
    assert ScrambleReports().apply_meeting(m).entries["inflation"].view.report == "solo"


def test_scramble_changes_the_brief_but_not_the_driver_set():
    m = _board().at("2023-06-30")
    scrambled = ScrambleReports().apply_meeting(m)
    assert render_brief(scrambled) != render_brief(m)
    assert set(scrambled.present) == set(m.present)          # grounding vocabulary intact
