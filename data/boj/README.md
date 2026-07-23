# `data/boj/` — Bank of Japan monetary-policy statement corpus

`documents.jsonl` — one JSON per line in the FOMC corpus schema
(`doc_type`/`release_date`/`text` + provenance fields), consumed by
`FomcCorpus` via the `text_corpus` field of the `jp_rates` / `jp_equity`
personas.

## Provenance

Fetched by `scripts/fetch_boj_statements.py` (2026-07-22; run with
`uv run --with pypdf` — pypdf is a scripts-only dependency). Discovery: the
BoJ per-year decision lists (`/en/mopo/mpmdeci/mpr_{YYYY}/index.htm`), keeping
links whose filename matches the `k{yymmdd}` decision-statement pattern with a
loose policy-vocabulary title guard — titles vary by era ("Statement on
Monetary Policy", "Announcement of the Monetary Policy Meeting Decisions",
"New Framework for Strengthening Monetary Easing…").

**248 docs, 2005-01-19 → 2026-06-16** (~14/yr through 2015, 8/yr after — the
BoJ cut its meeting count in 2016). Median ~750 words.

## Point-in-time contract

`release_date` = the decision date encoded in the statement filename;
statements publish the same day (typically around midday JST, well before any
Friday-close decision point).

## Quirks

- **Format varies by era**: HTML pages in 2005-06 and 2018+, PDFs for roughly
  2007-2017 (149 of 248 docs are PDF-sourced; the statements are
  machine-generated and extract cleanly via pypdf — eyeballed across eras).
- English statements only; BoJ publishes them alongside the Japanese
  originals, same day.
- Look-ahead caveat (project-wide, accepted for the narrative deliverable):
  see the fomc-recall-probe verdict in `EXPERIMENTS.md` before treating any
  in-window text result as alpha evidence.
