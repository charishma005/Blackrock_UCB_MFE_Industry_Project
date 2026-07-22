"""The analyst — an LLM specialist that reads evidence and writes a report.

This replaces the "deterministic reading, then optional LLM refinement" pattern.
That design put the finished ``DriverView`` — direction and conviction included —
inside the prompt and asked the model to refine it, so the model was shown the
conclusion and invited to agree; measured agreement with the formula ran 0.965 and
"the LLM added nothing" was the prompt's own doing. Here the model receives
evidence and nothing else, and every act of judgment is its own.

Two channels in, one report out:

    features   what moved — measurements from the closed vocabulary (features/)
    text       why it moved — driver-specific policy language (text/)
    ↓
    DriverView with a report, the evidence it leaned on, and what would falsify it

Note what is *not* here. There is no ``read()``, because measurement now belongs
to the feature engine and judgment belongs to the model — those were always two
different jobs sharing one method. And there is no subclass per driver: an analyst
is fully described by its persona file, so adding one is configuration.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.layered.contracts import DriverView, FeatureSet
from src.layered.features import FeatureEngine, from_persona
from src.layered.text import TextContext, TextSelector

PERSONA_DIR = Path(__file__).parent / "personas"

_CALIBRATION = """Use the full conviction range — most readings are not extreme:
  0.0-0.2  the evidence is mixed, or the driver is going nowhere
  0.3-0.5  a lean
  0.6-0.8  a clear signal
  0.9-1.0  unambiguous; rare"""

_OUTPUT_CONTRACT = """Submit your view with the submit_view tool. Fill "report"
first — the analysis in prose — then let "direction" and "conviction" follow from
it, so the call is a conclusion of the reasoning rather than a label you defend
after the fact. Cite measurements in "key_evidence" by the exact names given to you."""

# Tool schema: forcing this tool is the model-agnostic way to guarantee a parseable
# object. Field descriptions carry the same guidance the prose contract used to.
SUBMIT_VIEW_TOOL = {
    "name": "submit_view",
    "description": "Submit your analysis and view on the driver.",
    "input_schema": {
        "type": "object",
        "properties": {
            "report": {"type": "string",
                       "description": "Your analysis in prose, 120-250 words. Write this first."},
            "key_evidence": {"type": "array", "items": {"type": "string"},
                             "description": "Names of the measurements you relied on."},
            "falsifier": {"type": "string",
                          "description": "What you would need to see to change this view (<=30 words)."},
            "direction": {"type": "string", "enum": ["up", "down", "flat"]},
            "conviction": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["report", "key_evidence", "falsifier", "direction", "conviction"],
    },
}


class LLMAnalyst:
    """One driver, one specialist, one report."""

    def __init__(self, driver: str, persona: dict, engine: FeatureEngine,
                 llm=None, text_selector: TextSelector | None = None,
                 horizon_days: int = 63, horizon_label: str = "the next observation",
                 horizon_clock: str | None = None, horizon_freq: str | None = None,
                 describe_features: bool = False):
        self.driver = driver
        self.persona = persona
        self.engine = engine
        self.llm = llm
        self.text_selector = text_selector
        # When True, each feature is shown with its construction note (what it IS).
        # The A/B knob for step 2 — off reproduces the un-described prompt exactly.
        self.describe_features = describe_features
        # Two representations of one horizon. ``horizon_label`` is what the analyst is
        # told and what it is scored on; ``horizon_days`` only satisfies the
        # DriverView contract, which predates the release clock and wants a day count.
        self.horizon_days = horizon_days
        self.horizon_label = horizon_label
        # The evaluation clock: which series' releases are graded, and (for a daily
        # market series) how to resample it — ``clock_freq: ME`` gives a monthly,
        # non-overlapping clock. Both default so a monthly-release persona needs neither.
        self._horizon_clock = horizon_clock
        self.horizon_freq = horizon_freq
        self.last_raw: str | None = None     # last raw model response, for auditing

    @property
    def clock(self) -> str:
        """The graded series — persona's ``horizon.clock``, else the first input."""
        return self._horizon_clock or self.engine.inputs[0]

    @classmethod
    def from_persona(cls, driver: str, llm=None, text_selector: TextSelector | None = None,
                     persona_dir: Path | None = None, describe_features: bool = False) -> "LLMAnalyst":
        path = (persona_dir or PERSONA_DIR) / f"{driver}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"no persona spec for driver {driver!r} at {path}")
        persona = yaml.safe_load(path.read_text()) or {}
        horizon = persona.get("horizon") or {}
        return cls(
            driver=driver,
            persona=persona,
            engine=FeatureEngine(from_persona(driver, persona)),
            llm=llm,
            text_selector=text_selector,
            # Fall back to a plain `horizon_days:` for personas not yet migrated.
            horizon_days=int(horizon.get("approx_days", persona.get("horizon_days", 63))),
            horizon_label=horizon.get("label", "the next observation of your driver"),
            horizon_clock=horizon.get("clock"),
            horizon_freq=horizon.get("clock_freq"),
            describe_features=describe_features,
        )

    # ── isolation contract ──────────────────────────────────────────────────
    @property
    def inputs(self) -> tuple[str, ...]:
        return self.engine.inputs

    @property
    def cues(self) -> list[str]:
        return list(self.persona.get("text_cues") or [])

    # ── the two channels ────────────────────────────────────────────────────
    def build_inputs(self, world) -> tuple[FeatureSet, TextContext]:
        """Everything the analyst is allowed to see. Exposed so a prompt can be
        inspected without spending a call."""
        features = self.engine.compute(world)
        if self.text_selector is None:
            text = TextContext(driver=self.driver, available=False)
        else:
            text = self.text_selector.select(world.asof, self.cues, self.driver)
        return features, text

    # ── prompts ─────────────────────────────────────────────────────────────
    def _system_prompt(self) -> str:
        p = self.persona
        parts = [
            f"You are a specialist analyst covering exactly one driver: "
            f"{p.get('display_name', self.driver)}. You have no view on anything else, "
            f"and you never name a trade — expressing a view as a position is someone "
            f"else's job. You report on your driver only."
        ]
        if p.get("mandate"):
            parts.append("Mandate:\n" + "\n".join(f"- {m}" for m in p["mandate"]))
        parts.append(
            "You are shown two kinds of evidence about your driver: measurements "
            "(what moved) and policy language (why it moved). Both are as of the "
            "moment you are writing; there is no later information available to you, "
            "and no direction has been computed for you. Dates have been removed "
            "deliberately — do not try to identify the calendar period, and do not "
            "reason from anything you believe you know about it. Reason only from "
            "the evidence in front of you."
        )
        # The horizon must be stated. "Inflation is rising" over one month and over
        # six months are different claims, and naming the graded measurement removes
        # the last ambiguity about what the call actually means.
        target = self.engine.spec.level_feature
        measured = f" ({target})" if target else ""
        parts.append(
            f"Your view covers exactly one horizon: {self.horizon_label}. Report "
            f"whether your driver's headline measurement{measured} will be HIGHER "
            f"(up), LOWER (down), or essentially unchanged (flat) at that point, "
            f"compared with the most recent value you have been shown. Do not hedge "
            f"across other time frames — one horizon is being asked for, and it is "
            f"the one you will be scored against."
        )
        parts.append(_CALIBRATION)
        parts.append(
            "Reporting 'flat' with low conviction is a legitimate finding when the "
            "evidence does not support a call. A confident wrong call is worse than "
            "an honest abstention."
        )
        parts.append(_OUTPUT_CONTRACT)
        return "\n\n".join(parts)

    def _user_prompt(self, features: FeatureSet, text: TextContext) -> str:
        return "\n\n".join([
            f"Driver: {self.driver}",
            features.render(describe=self.describe_features),
            text.render(),
        ])

    # ── entry point ─────────────────────────────────────────────────────────
    def form_view(self, world) -> DriverView:
        features, text = self.build_inputs(world)
        return self.form_view_from(features, text)

    def form_view_from(self, features: FeatureSet, text: TextContext) -> DriverView:
        """One call, on evidence already built.

        Split out from ``form_view`` so a caller that has already assembled the
        evidence — ``CarryForward``, deciding whether anything moved — does not pay
        to assemble it twice.
        """
        if self.llm is None:
            raise RuntimeError(
                f"{self.driver}: LLMAnalyst needs an llm client — judgment is the "
                f"model's job here. Use build_inputs() to inspect the evidence without one."
            )
        try:
            raw = self.llm.complete(system=self._system_prompt(),
                                    user=self._user_prompt(features, text),
                                    tool=SUBMIT_VIEW_TOOL)   # forced tool = portable structured output
            self.last_raw = raw          # kept for the audit log
            # strict=False: reports are prose and legitimately contain newlines.
            parsed = json.loads(raw, strict=False)
        except Exception as e:  # noqa: BLE001 — one bad call must not end the meeting
            return self._degraded(features, f"{type(e).__name__}: {e}")

        direction = parsed.get("direction")
        if direction not in ("up", "down", "flat"):
            return self._degraded(features, f"invalid direction {direction!r}")
        try:
            conviction = min(1.0, max(0.0, float(parsed.get("conviction", 0.0))))
        except (TypeError, ValueError):
            return self._degraded(features, "non-numeric conviction")

        # Every cited measurement must be one we actually handed over. Because the
        # feature names are known exactly, this is a mechanical grounding check
        # rather than a lexicon guess. Some models fill the array field with one
        # comma-joined string; coerce so we don't shred it into characters.
        raw_ke = parsed.get("key_evidence") or []
        if isinstance(raw_ke, str):
            raw_ke = [s.strip() for s in raw_ke.split(",") if s.strip()]
        cited = [str(c) for c in raw_ke]
        valid = [c for c in cited if c in features.names]
        report = str(parsed.get("report", "")).strip()

        return DriverView(
            driver=self.driver,
            asof=features.asof,
            direction=direction,
            conviction=conviction,
            horizon_days=self.horizon_days,
            level=features.level,
            reasoning=report,
            report=report,
            key_evidence=valid,
            falsifier=str(parsed.get("falsifier", "")).strip(),
            source=f"llm:{self.driver}",
        )

    def _degraded(self, features: FeatureSet, why: str) -> DriverView:
        """An explicit abstention. Never a benchmark's answer — substituting one
        would mix the comparison into the thing being compared."""
        return DriverView(
            driver=self.driver, asof=features.asof, direction="flat", conviction=0.0,
            horizon_days=self.horizon_days, level=features.level,
            reasoning=f"no view formed ({why})", source=f"llm:{self.driver}", degraded=True,
        )
