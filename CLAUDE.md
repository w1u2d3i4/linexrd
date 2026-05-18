
---

## Autoresearch Pipeline (v2)

Two autonomous research workflows. Neither calls external APIs for review
or image generation. The only external actor is the human venue.

### Mode 1: Code Autoresearch (hyperparameter / architecture tuning)

**Trigger**: "start autoresearch", "tune", "optimize hyperparams", "code autoresearch"

**Two sub-modes inside** (see `.claude/skills/code-autoresearch/SKILL.md`):

- **Optuna outer loop** — if `autoresearch.search_space.json` exists and
  the `optuna` pip package is available, `autoresearch/tools/optuna_driver.py`
  runs TPE without consulting Claude per trial. Claude configures the
  search space and interprets results.
- **Claude-driven loop** — for structural / code changes. Same rules as
  before.

**Keep gate**: multi-seed statistical significance via
`autoresearch/tools/statsig.py`. A provisional keep must pass Welch + MWU +
bootstrap-CI vs the baseline seeds (`runs/baseline.jsonl`) before it is
logged with `runner.py log_experiment --status keep`.

**Rules**:
- Primary metric is king, but a single-seed bump is not a keep.
- `autoresearch.checks.sh` remains a hard correctness gate.
- Annotate every run with ASI (what you learned).
- Don't thrash: same idea failing twice → structurally different attempt.

**Files**: `autoresearch.md`, `autoresearch.sh`, `autoresearch.checks.sh`,
`autoresearch.search_space.json` (optional), `autoresearch.jsonl`,
`autoresearch.state.json`, `runs/baseline.jsonl`, `runs/statsig_round_*.json`.

### Mode 2: Paper Autoresearch (external-review quality loop)

**Trigger**: "paper autoresearch", "论文autoresearch", "improve paper with openreview"

**Protocol**:

1. If no `paper/main_v*.tex`: delegate drafting to `/paper-writing`
   (idea_e skill), then `/paper-hero-figure` if Fig 1 is a schematic.
2. Run local gates on the current version:
   - compile (tectonic / latexmk)
   - `autoresearch/tools/numcheck.py` — every number in the claim zone
     (abstract, tables, results) traces to `autoresearch.jsonl` /
     `results/*.json` / `CLAIMS_FROM_RESULTS.md`.
   - `autoresearch/tools/bibcheck.py` — every `\cite` key has a bib entry,
     every entry has required fields, no `[VERIFY]` / duplicates.
   - `autoresearch/tools/venuecheck.py --venue <preset>` — page count,
     anonymity extras (\institute, ack section, ORCID, grant numbers),
     template integrity, self-cite ratio, color-blind figure scan,
     limitations / checklist presence per venue.
3. `autoresearch/tools/export_for_openreview.py` assembles
   `paper/openreview/round_{N}/submission.pdf`, `submission.zip`,
   `UPLOAD_README.md`, `anonymity_report.md`, `fingerprint.json`.
4. **USER uploads to OpenReview / CMT / venue portal**, waits for reviews.
5. USER pastes review text into `paper/reviews/round{N}_openreview_raw.md`
   and says `reviews ready`.
6. `autoresearch/tools/import_review.py` parses the text into
   `review_round{N}.md` + `fix_plan_round{N}.md` and updates
   `PAPER_AUTORESEARCH_STATE.json`.
7. User checkpoint: approve `needs_data` / `needs_baseline` items.
8. Revise → new `main_v{N+1}.tex` → back to step 2.

**Rules**:
- Never fabricate numbers or reviewer scores.
- Never call an image / review / citation API from this skill.
- Hero figures follow Mode B: prompt + placeholder; user uploads the
  final image; `hero_figure_bundle.py ingest` swaps it in.
- `numcheck` / `bibcheck` / compile are blocking gates before export.
- Preserve `main_v0.tex`, `main_v1.tex`, ... forever.
- Loop stops at `mean_rating ≥ TARGET_SCORE` (default 6.0) or
  `MAX_ROUNDS = 5`.

**Files**: `paper/main_v{N}.tex`, `paper/reviews/round{N}_openreview_raw.md`,
`paper/reviews/review_round{N}.md`, `paper/fix_plan_round{N}.md`,
`paper/openreview/round_{N}/`, `paper/NUMCHECK_REPORT.md`,
`paper/BIBCHECK_REPORT.md`, `paper/VENUECHECK_REPORT.md`,
`paper/PAPER_AUTORESEARCH_STATE.json`.

### Mode 3: Rebuttal Autoresearch (CCF window response loop)

**Trigger**: "rebuttal", "写 rebuttal", "draft response", "回应 reviewer"
(or auto after `import_review.py`)

**Protocol** (see `.claude/skills/rebuttal-autoresearch/SKILL.md`):

1. Bucketize each weakness/question into
   `text_only | point_at_existing | quick_exp | new_baseline | new_data | concede`.
2. USER checkpoint: approve only `quick_exp` items that finish ≥48h
   before deadline. Recommend conceding `new_baseline` / `new_data`.
3. Run approved experiments with `asi.source = "rebuttal_round{N}"`,
   appending to `paper/rebuttal/round_{N}/new_experiments.jsonl`.
4. Draft `rebuttal_draft.md` per-reviewer (open with the answer, no
   thank-you preamble; use compact tables for new numbers).
5. Local gates via `autoresearch/tools/rebuttal_compile.py compile`:
   - point-to-point (every reviewer item has a non-empty response),
   - char count (per-reviewer venue cap),
   - numcheck (every number in the rebuttal traces to evidence).
6. USER approval → copy to `rebuttal_final.md`, update state to
   `rebuttal_submitted`.

**Files**: `paper/rebuttal/round_{N}/{rebuttal_draft.md,
point_to_point_table.md, char_count.txt, REBUTTAL_NUMCHECK_REPORT.md,
new_experiments.jsonl, rebuttal_final.md}`.

### CCF / venue-specific reproducibility pack

Run `autoresearch/tools/reproducibility_pack.py --venue <preset>` to
generate `paper/repro/{hparams.yaml, seeds.json, env.txt, cmds.sh,
README.md, checklist.md, results_summary.md, code/}`. Required by
NeurIPS/ICML/ICLR/ACL submissions; recommended for CVPR/AAAI.
