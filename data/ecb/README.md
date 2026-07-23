# `data/ecb/` — ECB monetary-policy statement corpus

`documents.jsonl` — one JSON per line in the FOMC corpus schema
(`doc_type`/`release_date`/`text` + provenance fields), consumed by
`FomcCorpus` via the `text_corpus` field of the `ea_rates` / `ea_equity`
personas.

## Provenance

Fetched by `scripts/fetch_ecb_statements.py` (2026-07-22) from the ECB's
monetary-policy-statement archive. Discovery: the JS listing page lazyloads
static per-year fragments
(`.../monetary-policy-statement/{YYYY}/html/index_include.en.html`) whose
`<dt isoDate>` entries carry date + link — stable back to 1998. We keep docs
titled "Introductory statement" (pre-2021) / "Monetary policy statement"
(2021+): these exist for **every** Governing Council monetary-policy meeting,
unlike the "Monetary policy decisions" press release (irregular pre-2012).

**205 docs, 2005-01-13 → 2026-06-11** (~8/yr; the window matches the vendored
FOMC corpus). Median ~6,600 words.

## Point-in-time contract

`release_date` = press-conference day (statement delivered ~14:30 CET on the
decision day). `FomcCorpus.as_of` serves the latest doc with
`release_date <= asof`, so a Friday decision date never sees a future
statement.

## Quirks

- Text **includes the Q&A transcript** where the page carries it ("with Q&A"
  titles) — the full press-conference record, not just the prepared statement.
  Truncate at run time with `--text-max-chars` if only the opening statement
  is wanted.
- Look-ahead caveat (project-wide, accepted for the narrative deliverable):
  these documents are inside the LLM's training window; see the
  fomc-recall-probe verdict in `EXPERIMENTS.md` before treating any in-window
  text result as alpha evidence.
