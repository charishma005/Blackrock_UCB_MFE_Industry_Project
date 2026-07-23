# Project deck

`layered_agent_fund_phase1.pptx` — **14 story slides + 5 appendix slides**. Generated from
`make_deck.py`, so every number traces back to `reports/` instead of being retyped.

```bash
.venv/bin/pip install python-pptx
.venv/bin/python slides/make_deck.py
```

## The story

Three beats, in order: **the layers idea** → **the model added nothing** → **the input is what
mattered**.

1. When an AI fund loses money, nobody can say why · 2. So we gave the fund an org chart ·
3. An analyst may say "inflation is rising", not "buy the 10-year" · 4. Meet one analyst ·
5. The PM knows one thing the analysts don't · 6. We built every analyst twice · 7. **The model
added nothing (+0.0645 → +0.0648)** · 8. So we changed what it reads · 9. Words help where the
Fed does the talking · 10. But the words come with a bill · 11. Give it both · 12. What we have
not tested · 13. What we do next.

## Rules the deck follows — keep them if you edit it

- **The headline is the claim, not a label.** "The model added nothing", not "Correctness".
- **One idea per slide, ≤ 3 objects, ≤ 40 words of body.** A fourth box means it's two slides.
- **No internal jargon in the story.** Phase 1 → *the formula*. Phase 2 → *the model*.
  `own_corr` → *"does it still track its own driver"*. `edge_vs_random` → *"better than a coin
  flip"*. `override` → *"disagreed with the formula"*.
- **Machinery lives in exactly two places**: slide 4 (the one worked example) and the appendix.
- **Plain sentences**: "we ask whether the driver actually moved the way the analyst said, and
  we compare that to flipping a coin."

## Logos

Drop `berkeley.png` and `blackrock.png` into `slides/assets/` and re-run — they're picked up
automatically. Without them the deck falls back to typographic wordmarks, so it builds offline.
Use transparent wordmarks, not square lockups (they scale to a fixed height).

## Sources — and three traps

Numbers come from `reports/phase1_{vector,text,textvec}.md` (the 4-driver A/B),
`reports/phase1_7d_textvec.md` (the seven-analyst run) and the matching `*.audit.json`.

- **`phase1_7d_textvec.md` is the SEVEN-ANALYST run, not a 7-day run.** 2,191 calls = 313 × 7;
  the horizon is still 63d. It is text+vector only, so it is **not** an A/B.
- **Do not cite `phase1.md` or `phase1_fred.html`.** `phase1.md` is a superseded earlier
  sampling of the same vector config (override counts 25/20/28/5 vs the current 26/21/29/8) and
  has no audit.json. `phase1_fred.html` is rendered from it, so it will not reconcile with
  `phase1_comparison.html`.
- **Never quote the §3 `interpretation` column of the four older reports.** It prints "possible
  training-cutoff leak" on every row regardless of the numbers — a fixed template, fixed in
  `7b7b68c`. Only `phase1_7d_textvec.md` has the computed verdicts. Regenerate before
  circulating.

Two more caveats baked into the slide copy: `edge_vs_persistence` flatters the monthly drivers
(the baseline calls "flat" most weeks and a flat call never scores, so it hits 0.135 on
inflation — read `edge_vs_random` instead); and the 2019–2024 window has no post-cutoff control,
so the leak test is inconclusive by construction.

## Rendering to PDF (how to verify layout)

Geometry checks do **not** catch text overflowing its box — that was the real defect class. You
have to look at the pages.

PowerPoint's AppleScript export works but has three traps: it must write to a path its sandbox
accepts, the Bash sandbox must be disabled for the call, and **every open presentation must be
closed first** — otherwise `open` re-activates a stale in-memory copy and you silently export
the *old* deck.

```applescript
tell application "Microsoft PowerPoint"
  repeat while (count of presentations) > 0
    close presentation 1 saving no
  end repeat
  open POSIX file "…/slides/layered_agent_fund_phase1.pptx"
  delay 3
  save active presentation in POSIX file "…/slides/_render/final.pdf" as save as PDF
end tell
```

`brew install --cask libreoffice` → `soffice --headless --convert-to pdf` avoids all of this.

## Slides teammates will want

- **2 — the org chart.** The analyst strip is this fork's macro/rates pool with an
  "equity analysts — teammates, same slot" note. That's the hook for the equity work.
- **4 — meet one analyst.** The input/gate/analyst/claim flow is generic; an equity analyst
  drops into the same shape.
