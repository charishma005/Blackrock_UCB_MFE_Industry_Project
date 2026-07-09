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
   by the whole book, not just per-ticker signals. Equity fundamentals come from
   a pluggable data layer with three interchangeable sources — `yfinance`
   (free, no point-in-time history), `financialdatasets.ai` (paid, point-in-time),
   and `wrds` (Compustat/CRSP, point-in-time). (`src/instruments.py`,
   `src/data/markets.py`, `src/data/equities*.py`, `src/agents/ray_dalio.py`)
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
- [x] Ray Dalio macro regime agent (wired to LLM client)
- [x] Port Warren Buffett + Aswath Damodaran agents from upstream (MIT header kept)
- [x] Multi-asset risk manager (vol + cross-asset correlation limits)
- [x] LLM Portfolio Manager consuming weighted signals + regime context
      (also a `mechanical` PM mode; select via `--pm-mode`)
- [x] Backtest engine (weekly/daily loop) + 3-way benchmark comparison study
- [x] Point-in-time equity data via `financialdatasets.ai` and WRDS
- [x] Persona specs committed (`src/agents/personas/*.yaml`)
- [ ] Persona distillation script (transcripts -> YAML) — specs are hand-authored for now
- [ ] Fixed-income agent (Gundlach-style: duration, curve, credit)
- [ ] Commodities agent (COT positioning + trend)

## Setup

```bash
pip install pandas numpy requests yfinance pyyaml pydantic anthropic
pip install "multi-asset-fund[wrds]"   # optional: only for --equity-data-source wrds

export FRED_API_KEY=...         # free: fred.stlouisfed.org (macro series)
export ANTHROPIC_API_KEY=...    # LLM agents + LLM portfolio manager
# optional, depending on --equity-data-source:
export FINANCIAL_DATASETS_API_KEY=...   # for financialdatasets.ai (point-in-time)
export WRDS_USERNAME=...                # for wrds (password via ~/.pgpass)
```

Run the headline experiment (equal- vs performance-weighted vs benchmark):

```bash
python3 -m src.run_backtest --start 2024-01-01 --end 2024-12-31 \
    --equity-data-source yfinance      # yfinance | financialdatasets | wrds
```

Useful flags: `--rebalance-freq` (pandas offset, e.g. `W-FRI`, `ME`),
`--model` (default `claude-haiku-4-5-20251001`), `--pm-mode` (`mechanical` |
`llm`), `--no-risk-manager`, `--cache-dir`, `--verbose`.

> **Look-ahead note:** `yfinance` has no point-in-time fundamentals, so
> Buffett/Damodaran signals are computed once from *today's* filings and are
> look-ahead biased — fine for a smoke test, not for a real headline run. Use
> `financialdatasets` or `wrds` for point-in-time results.

## Headline experiment

`src.run_backtest` runs a 3-way comparison — equal-weighted agents vs
performance-weighted (this repo) vs a naive buy-and-hold benchmark — and reports
Sharpe/Sortino/Calmar/max-DD plus per-agent attribution. A 2020–2025 run is
captured in `attribution_2020_2025.txt`.
