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

## Status / roadmap

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
