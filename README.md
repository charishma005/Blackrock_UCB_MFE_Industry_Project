# Multi-Asset AI Fund — persona-conditioned agents with performance-based weighting

A multi-asset extension of the agent-ensemble trading framework, adding macro /
fixed-income / commodity investor agents, per-agent performance attribution,
dynamic agent weighting ("firing"), and persona conditioning from primary sources.

> Built on the open-source [ai-hedge-fund](https://github.com/virattt/ai-hedge-fund)
> framework by virattt (MIT license). The equity agents (Warren Buffett, Aswath
> Damodaran) and the Phase-1/Phase-2 agent pattern originate there; everything in
> `src/ensemble/`, `src/instruments.py`, the multi-asset data layer, the macro
> agents, and the persona-conditioning pipeline is new work in this repo.

## The four contributions

1. **Multi-asset agent universe** — `Instrument`/`AssetClass` abstraction routes
   fixed income (TLT/IEF/LQD/HYG), commodities (GC/CL/HG), and FRED macro series
   to agents that cover them. Macro agents emit a *regime* consumed as context
   by the whole book, not just per-ticker signals. (`src/instruments.py`,
   `src/data/markets.py`, `src/agents/ray_dalio.py`)
2. **Full portfolio metrics** — Sharpe, Sortino, Calmar, max drawdown, hit rate,
   turnover, Spearman IC. (`src/backtest/metrics.py`)
3. **Attribution + firing** — every agent gets a paper portfolio; rolling Sharpe
   / IC feed a softmax weight with regime-aware floors and strike-based hard
   exclusion. (`src/ensemble/attribution.py`, `src/ensemble/weights.py`)
4. **Persona conditioning** — agents load distilled YAML persona specs (beliefs,
   decision rules, vocabulary) synthesized offline from shareholder letters and
   interview transcripts. Specs, not raw transcripts, are committed.
   (`src/agents/personas/`, `src/agents/base.py`)

## Next step: the layered agent fund (`src/layered/`)

The flat ensemble above blends whole-investor *opinions*. The next step, set out
in the project's **"A Layered Agent Fund"** thesis, reorganizes the fund around
the weekly **analyst–PM meeting** and separates three things a monolithic system
blurs together: **belief formation**, **action selection**, and **survival**.

```
  analyst layer            PM layer                      unifying layer
  (isolated single-        (arbitrate the views +        (net exposures across
   driver experts)          express ONE relative-         strategies, allocate
        │                   value trade per strategy)     capital, feed back down)
   DriverView  ───────►   StrategyTrade   ───────►    FundAllocation ──┐
        ▲                                                              │
        └──────────────  capital + constraints flow back down  ◄───────┘
```

The three layers talk only through **stable contracts** (`src/layered/contracts.py`)
— `DriverView`, `StrategyTrade`, `FundAllocation` — so any layer's method can be
swapped without touching the others. The thesis commits to no forecasting
technique; these are the interfaces, plus one **worked end-to-end instance**: a
macro-rates PM fed by four single-driver analysts (inflation, labor, Fed balance
sheet, term premium) that expresses their joint view as a **DV01-neutral 2s10s
flattener** — the exact example from the thesis.

Because research quality and arbitrage quality are attributed *separately*
(`src/layered/scoring.py` grades analysts on whether their driver calls were
right; metrics grade the book on P&L), the fund can ask *which* layer failed.

Runs offline with no keys (`--source synthetic`) and with real data
(`--source fred`); the LLM is an optional Phase-2 refinement of each analyst.

```bash
python3 test_layered.py                 # offline smoke test (no keys, no network)
python3 -m src.run_layered              # synthetic hawkish regime → a flattener
python3 -m src.run_layered --regime dovish          # mirror image → a steepener
python3 -m src.run_layered --source fred --start 2022-01-01 --end 2024-12-31
```

Layered layout: `contracts.py` (interfaces), `timeline.py` (no-lookahead gate),
`analysts/` (isolated experts + mandate specs), `pm/` (arbitrate/express +
transmission-map persona), `fund.py` (netting/allocation), `meeting.py` (the
meeting as a run loop), `scoring.py` (research attribution), `backtest.py` (weekly
loop), `synthetic.py` (offline data).

### Phase-1 diagnostics (`src/layered/diagnostics.py`)

Before trusting the analyst layer, four things get measured — each for the
**deterministic Phase-1** agents and the **LLM Phase-2** agents *side by side*,
so you can see what the LLM adds (skill? correlation? a future leak?):

1. **Faithfulness** — is the inflation agent really about inflation? *Input
   isolation* is structural (an access-logging probe confirms each analyst reads
   only its declared series, nothing past `asof`); *responsiveness vs
   contamination* correlates each agent's view against every driver's honest
   measurement (own high, cross low = faithful); a lexicon proxy checks the
   reasoning stays on-topic.
2. **Correctness** — horizon-aware hit rate + information score vs a persistence
   baseline and a coin flip (`edge_vs_persistence`, `edge_vs_random`).
3. **Lookahead** — the data-slice leak is caught by the isolation probe + the
   `AsOf` unit test; the subtle **LLM training-cutoff** leak is caught by a
   *prescience* test: information gain of the LLM over the no-future-info
   deterministic baseline, plus accuracy on dates where the LLM *overrode* it.
   Only meaningful on `--source fred` (synthetic series have no real future to
   memorize — the report says so).
4. **Correlation** — pairwise correlation of the analysts' signed-conviction
   streams; low off-diagonal = the independence the thesis is buying.

```bash
python3 -m src.run_diagnostics                        # synthetic, deterministic column
ANTHROPIC_API_KEY=... python3 -m src.run_diagnostics  # adds the LLM column
ANTHROPIC_API_KEY=... FRED_API_KEY=... python3 -m src.run_diagnostics \
    --source fred --start 2019-01-01 --end 2024-12-31 --out reports/phase1.md
python3 test_diagnostics.py                            # offline (stub LLM), no keys
```

## Status / roadmap

Flat ensemble (prior step):
- [x] Instruments + data layer (yfinance, FRED)
- [x] Metrics module
- [x] Attribution tracker + weight manager
- [x] Agent base class + persona loading
- [x] Ray Dalio macro regime agent (Phase 1 complete; wire LLM client)
- [ ] Port Warren Buffett + Aswath Damodaran agents from upstream (keep MIT header)
- [ ] Multi-asset risk manager (vol + cross-asset correlation limits)
- [ ] LLM Portfolio Manager consuming weighted signals + regime context
- [ ] Backtest engine (daily loop) + benchmark comparison study
- [ ] Persona distillation script (transcripts -> YAML)
- [ ] Fixed-income agent (Gundlach-style: duration, curve, credit)
- [ ] Commodities agent (COT positioning + trend)

Layered agent fund (next step):
- [x] Layer contracts: `DriverView` / `StrategyTrade` / `FundAllocation`
- [x] Time-integrity gate (`AsOf`) — no agent sees data it could not have had
- [x] Analyst layer: single-driver base + inflation / labor / balance-sheet / term-premium
- [x] PM layer: arbitrate + express, with a driver→instrument transmission map
- [x] Worked instance: macro-rates PM → DV01-neutral 2s10s flattener
- [x] Unifying layer: net exposures, allocate by conviction / risk / diversification
- [x] Research scoring: grade analysts on being right, separately from P&L
- [x] Weekly meeting loop + offline synthetic data + smoke test
- [x] Phase-1 diagnostics: faithfulness, correctness vs baseline, lookahead +
      LLM prescience, cross-agent correlation (deterministic vs LLM, side by side)
- [ ] Second strategy (equity long/short or credit basis) to activate cross-strategy diversification
- [ ] Multi-analyst-per-driver (preserve/consume disagreement as a first-class signal)
- [ ] Wire LLM Phase-2 refinement on a real FRED/yfinance run

## Setup

```bash
pip install pandas numpy requests yfinance pyyaml pydantic
export FRED_API_KEY=...        # free: fred.stlouisfed.org
export ANTHROPIC_API_KEY=...   # for LLM agents
python -m src.main --start 2024-01-01 --end 2024-12-31
```

## Headline experiment (planned)

Backtest three configurations on 2020–2025:
equal-weighted agents vs performance-weighted (this repo) vs 60/40 benchmark —
report Sharpe/Sortino/Calmar/max-DD and per-agent attribution.
