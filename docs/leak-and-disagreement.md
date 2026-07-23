# Leak & disagreement arms — design record

Status: **built and tested, not yet run against the model.** Added 2026-07-22. Four
evaluation-side additions derived from four papers (the "Tier-1" set), each attacking a
live, unresolved problem in the two-layer LLM fund (analysts → PM). No production prompt
is touched; everything here is a diagnostic, held to the honesty rule of
`analyst-layer.md` §6 — *these numbers diagnose the problem and must never inform the
prompt.* Validated offline with the fake-LLM stubs; the scored arms cost API calls and
are run separately.

---

## 1. Why this exists

Two standing problems the papers speak to:

- **Training-cutoff leak, now across two stacked LLM layers.** IC rises monotonically
  with model capability over a window inside every model's training data — the signature
  of memorization, which `analyst-layer.md` §10 concedes it cannot separate from skill.
  Date-scrubbing removes calendar tokens but leaves era identity ("pandemic", "war"). The
  PM brief adds a second prose surface with the same exposure.
- **The PM computes `panel_disagreement` and does nothing with it.** The machinery to
  preserve analyst conflict as a number exists (`pm/disagreement.py`) and is stored per
  meeting, but it is only a size-down flag; whether it is a *signal* was untested.

| # | Paper | What it contributes here |
|---|---|---|
| A | Canayaz, *AI Agency* | Input-perturbation "unlearning" test: alter the evidence so a memorised answer becomes wrong. A reasoning model follows the altered arithmetic; a recalling one does not — and this runs on the full sample, not the n=15 post-cutoff slice. |
| B | Han et al., *Causal Agent based on LLM* | Prior-vs-evidence probe: scramble which report sits under which driver label, so world-knowledge priors and the evidence disagree. |
| C | Horton et al., *Homo Silicus* | Prompt-permutation battery: report IC across meaning-preserving variants. "Prompt-hacking is p-hacking"; a signal that survives only one exact phrasing is fragile. |
| D | Bali-Kelly-Mörke-Rahman, *Machine Forecast Disagreement* | The dispersion across heterogeneous forecasters is itself informative — the PM already computes it; test whether it predicts anything. |

A/B/C share one mechanism — transform the render-time object between `build_inputs` and
the call, re-run, compare to the clean baseline — so they share one harness. D is pure
offline scoring over saved run files.

---

## 2. Decisions taken, and why

| # | Decision | Rationale |
|---|---|---|
| 1 | **Evaluation-side only; never feeds a shipped run** | An arm is selected by an explicit `--perturb` flag, recorded in `.meta.json`, and excluded from board runs. This is the same honesty rule the feature ICs live under. |
| 2 | **Perturb at the compute→render seam, not the raw data** | Every layer already splits `build_inputs` (compute) from `render`. Applying the transform on the returned `FeatureSet`/`TextContext`/`Meeting` is side-effect-free and leaves the raw series and the scoring outcome clean. |
| 3 | **Perturbations are pure — `model_copy`, never mutate** | The caller's original object must survive so the model's answer can be graded against an unperturbed outcome. Enforced by a test. |
| 4 | **The off arm reproduces byte-for-byte** | Guarded on `self.perturbation is not None`, so the shipped path runs identical code. A no-op `IDENTITY` is offered for callers that prefer not to branch. |
| 5 | **Deterministic transforms; no seed/temperature added** | The client exposes neither (out of scope this round). Offline tests use deterministic stubs and are exact; a real run carries the model's own sampling noise, stated in every docstring. |
| 6 | **Two perturbation registries, split on the layering boundary** | The analyst layer must not import the PM layer. Feature/text/string perturbations resolve via `analyst_perturbation` (PM-free); the scramble resolves via `pm_perturbation` in `perturb/brief.py`, reached only from the PM run script. Asserted: the analyst import path loads zero `src.layered.pm` modules. |
| 7 | **`perturb` added to `board.IDENTITY_KEYS`** | So a board can never silently mix a perturbed analyst leg with a clean one. Existing run files lack the key → all read `None` → no spurious mismatch. |
| 8 | **`signflip_momentum` never flips the level** | A year-over-year *rate* (`headline_cpi_yoy`) is a level, not a change; the token filter excludes `yoy` and the level feature is protected explicitly. The `_change_3m` suffix still catches the genuine changes. |
| 9 | **`disagreement_signal` reuses `pm_bench`'s clock rebuild → import-by-path** | It pulls in the PM layer, so like `pm_bench` it is not re-exported from the evaluation `__init__` (circular). `perturbation_bench` is PM-free and *is* re-exported. |
| 10 | **Scramble is a deterministic rotation (a derangement for ≥2 present drivers)** | No RNG, so a scrambled run reproduces. Driver *keys* are untouched, so grounding, coverage, and the submit enum still see the real driver set; only the evidence bound to each label moves. |

---

## 3. Module map

```
src/layered/perturb/            the shared harness (A/B/C)
  base.py       Perturbation ABC (no-op hooks) + IDENTITY
  features.py   Rescale · ShiftLevel · SignFlipMomentum · CounterfactualPath ·
                ReorderFeatureLines  — operate on a FeatureSet (arm A + C)
  text.py       WhitespaceVariant · RewordScaffolding — meaning-preserving, on the
                assembled prompt string (arm C); shared by both layers
  brief.py      ScrambleReports — rotates a report under the wrong driver label
                (arm B); the only perturb module that imports the PM layer
  __init__.py   analyst_perturbation(name) + ANALYST_NAMES  (PM-free surface)

src/layered/evaluation/
  disagreement_signal.py  the MFD read (D) — reuses pm_bench.driver_levels; NOT
                          re-exported (imports layered.pm)
  perturbation_bench.py   score a perturbed run vs baseline (A/B/C); PM-free, re-exported:
                          direction_response · ic_stability/ic_dispersion · scramble_response

tests/    test_perturb.py · test_perturbation_bench.py · test_disagreement_signal.py
notebooks/ perturbation_evaluation.ipynb · disagreement_signal.ipynb
```

The seam is one guarded line in each layer, mirroring the existing `blind`/`memory` arms:

- `LLMAnalyst.build_inputs` applies `apply_features`/`apply_text`; `_user_prompt` applies
  `apply_prompt`. Threaded through `build_analyst`, exposed as `--perturb` on
  `run_analyst_ic` (and `run_analyst --dry-run` for inspection).
- `LLMPM.build_inputs` applies `apply_meeting`; `_user_prompt` applies `apply_prompt`.
  Threaded through `build_pm`, exposed as `--perturb` / `--scramble-reports` on `run_pm_ic`.

---

## 4. The comparison metrics, and which direction is "good"

Two arms report a *rate of change*, and the sign of "good" is opposite between them.

| arm | function | reads | good direction |
|---|---|---|---|
| A leak | `direction_response(base, perturbed)` | `flip_rate` on non-flat calls | **high** = reading the evidence; low = the recall fingerprint |
| B scramble | `scramble_response(base_pm, scrambled_pm)` | per-driver sign-change rate | **high** = read the mislabeled evidence; low = recited the label's prior |
| C battery | `ic_stability({label: path})` + `ic_dispersion` | IC per variant, spread | **small spread** = robust; wide = fragile |
| D signal | `disagreement_signal.evaluate_run(pm, board_dir)` | see §5 | — |

---

## 5. First offline result — disagreement conditioning (D)

Run on the committed `reports/pm/duration_on.jsonl` (120 meetings, Sonnet) with the
`reports/ab` board, no model calls:

| meetings | disagreement → next \|move\| | PM hit low vs high disagreement | PM IC low vs high |
|---|---|---|---|
| 120 (mean disagree 0.59) | IC +0.084, t 0.91 | **0.57 vs 0.51** | **+0.117 vs +0.014** |

The magnitude and graph-density ICs are within noise (|t| < 1), but the **conditioning
split is the finding**: the PM is materially more accurate when the panel agrees than
when it is split. That makes disagreement a usable trust-discount signal — the calibrated
use the pod mandate already gestures at — and is the evidence that informs whether
disagreement should *drive the trade*, a decision the PM does not make for itself.

---

## 6. How to run

**No spend — offline (this is what the tests and notebooks exercise):**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/          # 160 pass
python3 -m src.run_analyst --driver inflation --dry-run --asof 2023-08-01 --perturb signflip_momentum
python3 -m src.run_pm_ic --pod duration --dry-run --scramble-reports
python3 -c "from src.layered.evaluation.disagreement_signal import evaluate_run, summarize; \
            print(summarize(evaluate_run('reports/pm/duration_on.jsonl', board_dir='reports/ab')))"
```

**Scored — real API spend, run separately, then compare:**

```bash
python3 -m src.run_analyst_ic --driver inflation --perturb signflip_momentum \
        --model claude-haiku-4-5-20251001 --limit 24 --out reports/perturb/inflation_signflip.jsonl
python3 -m src.run_pm_ic --pod duration --scramble-reports --out reports/pm/duration_scramble.jsonl
# then: perturbation_bench.direction_response / scramble_response / ic_stability
```

---

## 7. Open questions and known gaps

1. **The scored arms are unrun.** Every metric here is validated on stubs and identity
   checks; the behavioural question — does the model follow the perturbed evidence — needs
   real calls. This is the deliberate build-only boundary of this round.
2. **No seed/temperature.** A real perturbation run carries sampling noise the offline
   tests do not. Adding a temperature knob (Homo Silicus's point-mass finding) was scoped
   out; it would also make the arms reproducible.
3. **Counterfactual grading.** `CounterfactualPath` reverses the trajectory, which moves
   the graded level — a counterfactual run must be graded against its *own* reversed
   outcome, not the clean one. `perturbation_bench.direction_response` compares signals,
   not outcomes, so it is unaffected; a future outcome-based scorer must respect this.
4. **A failed perturbation is ambiguous.** As Homo Silicus found under translation, a
   perturbation can destroy semantic content alongside surface form; a null result is only
   interpretable if you know which was broken. The meaning-preserving battery (C) is the
   control that keeps that legible.
