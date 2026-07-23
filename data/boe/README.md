# `data/boe/` — Bank of England Monetary Policy Summary corpus

`documents.jsonl` — one JSON per line in the FOMC corpus schema
(`doc_type`/`release_date`/`text` + provenance fields), consumed by
`FomcCorpus` via the `text_corpus` field of the `uk_rates` / `uk_equity`
personas.

## Provenance

Fetched by `scripts/fetch_boe_statements.py` (2026-07-22). Discovery: the BoE
sitemap API (`/_api/sitemap/getsitemap` — the plain `/sitemap.xml` redirects
to an error page), filtered to `/monetary-policy-summary-and-minutes/` URLs
(early slugs are `mpc-august-2015`-style; guessing `august-2015` 404s).

**91 docs, 2015-08-06 → 2026-06-18** (8/yr). Median ~1,200 words; some pages
include the full MPC minutes below the summary (same release, kept).

## Point-in-time contract

The decision date is parsed from the summary text ("…meeting ending on
3 August 2016…"); `release_date` = that date **plus one day**, because the
MPC announcement is published at noon on the day after the meeting ends. The
sidebar `<time>` tags on the page are recent-news links, not the article
date — the fetcher deliberately ignores them.

## Coverage caveat (team decision, GA 2026-07-22)

**The UK text channel starts 2015-08** — before Aug 2015 the BoE published no
statement when rates were unchanged; the pre-2015 record is the MPC minutes
(~2-week lag), deliberately out of scope. Earlier asof dates get
`TextContext(available=False)`, which the pipeline handles as "no document
yet" — expected, not an error. Also the project-wide look-ahead caveat: see
the fomc-recall-probe verdict in `EXPERIMENTS.md`.
