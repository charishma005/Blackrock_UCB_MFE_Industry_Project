# `data/equity_replay/` — validated r7 vector-arm signals

The four macro-llm r7 single-driver equity analysts' **vector-arm** outputs, one
CSV per driver, columns `date,pos,conviction,reasoning`, 752 Fridays 2012-01-13 →
2026-06-05. Consumed by `src/portfolio/replay_analyst.py` (`CsvReplayAnalyst`),
which replays them as `DriverView`s point-in-time — no LLM calls, $0.

## Provenance

Copied verbatim by `macrollm.scripts.export_equity_series` from macro-llm
`results/equity_agents/<driver>_vector.csv`. These are the **validated** signal
(round-7 verdict, EXPERIMENTS.md): the date-blind vector arm, shuffle/leak-tested.
Unlike the live persona path, the r7 verdict was "no honest arm adds value" — the
replay feed supplies *real* driver signals to the pods, not proven alpha.

## Direction-semantics caveat (read before wiring into scoring)

`pos ∈ [−1, +1]` is the r7 analyst's desired **S&P 500 position**, i.e. a claim
about the *market*. `CsvReplayAnalyst` maps it to `DriverView.direction` as
`pos > +0.15 → up`, `pos < −0.15 → down`, else `flat` (the ±0.15 threshold is from
the r7 prereg). So a replay view's `direction` carries **market-call** semantics
under a driver-named key — whereas a live `LLMAnalyst` persona emits
**driver-direction** semantics (will the driver's own level rise?). Pods must
disambiguate the two via the `source` prefix (`replay:` vs `llm:`). This split is
flagged for team sign-off.
