#!/usr/bin/env python3
"""
import_review — parse a raw OpenReview review dump that the user pastes in,
produce the merged review markdown + categorized fix plan that
paper-autoresearch expects, and update PAPER_AUTORESEARCH_STATE.json.

No network. Pure stdlib.

Expected raw formats (auto-detected):

  A. OpenReview-exported text with blocks separated by "Official Review"
     or "Reviewer <letter>" / "Reviewer #<n>" headers.
  B. Free-form: any markdown where each reviewer's section begins with a
     heading containing the word "review".
  C. Single-reviewer text.

Within each reviewer block we look for:

  Summary / Strengths / Weaknesses / Questions / Limitations
  Soundness / Presentation / Contribution / Rating / Overall / Confidence

Each weakness line is categorized along two axes:

  severity  := CRITICAL | MAJOR | MINOR
  category  := text_fix | needs_experiment | needs_data | needs_baseline

using a small keyword lexicon (see below). Claude can override these in the
fix plan after reading.

Usage
  python import_review.py \
      --raw paper/reviews/round2_openreview_raw.md \
      --round 2 --paper-dir paper/

Outputs
  paper/reviews/review_round2.md
  paper/fix_plan_round2.md
  paper/PAPER_AUTORESEARCH_STATE.json (updated)

Exit codes
  0  parsed and wrote files (paper still needs revision)
  10 parsed and target score met -> DONE
  2  usage error or malformed input
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------

CRITICAL_WORDS = (
    "fundamental", "fatal", "incorrect", "wrong", "unsound", "invalid",
    "contradicts", "contradiction", "reject", "no novelty", "not novel",
    "missing proof", "no proof", "mathematically wrong", "serious flaw",
)
MAJOR_WORDS = (
    "missing", "unclear", "weak", "insufficient", "needs", "should",
    "would benefit", "overclaim", "overstated", "concerning", "problematic",
    "significantly", "lacks", "no comparison", "no baseline", "ablation",
    "limited", "narrow",
)
# MINOR is default

NEEDS_EXP_WORDS = (
    "ablation", "additional experiment", "more experiments",
    "additional ablation", "sensitivity analysis",
    "run", "report", "measure", "try", "hyperparameter", "learning rate",
    "more seeds", "error bars", "significance",
)
NEEDS_DATA_WORDS = (
    "new dataset", "additional dataset", "out-of-distribution", "ood",
    "real-world data", "larger dataset", "held-out", "held out",
    "evaluate on",
)
NEEDS_BASELINE_WORDS = (
    "new baseline", "additional baseline", "compare to", "compare with",
    "compared against", "strong baseline", "stronger baseline",
    "should compare", "missing comparison", "sota", "state-of-the-art",
)

SCORE_PATTERNS = {
    "soundness":    re.compile(r"^[\s*•\-]*soundness[\s:]*(\d+(?:\.\d+)?)", re.I | re.M),
    "presentation": re.compile(r"^[\s*•\-]*presentation[\s:]*(\d+(?:\.\d+)?)", re.I | re.M),
    "contribution": re.compile(r"^[\s*•\-]*contribution[\s:]*(\d+(?:\.\d+)?)", re.I | re.M),
    "rating":       re.compile(r"^[\s*•\-]*(?:rating|overall|overall score)[\s:]*(\d+(?:\.\d+)?)", re.I | re.M),
    "confidence":   re.compile(r"^[\s*•\-]*confidence[\s:]*(\d+(?:\.\d+)?)", re.I | re.M),
}

SECTION_HEADINGS = (
    ("summary", re.compile(r"^(?:#+|\*\*)?\s*Summary\b", re.I | re.M)),
    ("strengths", re.compile(r"^(?:#+|\*\*)?\s*Strengths?\b", re.I | re.M)),
    ("weaknesses", re.compile(r"^(?:#+|\*\*)?\s*Weaknesses?\b", re.I | re.M)),
    ("questions", re.compile(r"^(?:#+|\*\*)?\s*Questions?\b", re.I | re.M)),
    ("limitations", re.compile(r"^(?:#+|\*\*)?\s*Limitations?\b", re.I | re.M)),
    ("ethics", re.compile(r"^(?:#+|\*\*)?\s*Ethics\b", re.I | re.M)),
)

REVIEWER_HEADER_RE = re.compile(
    r"(?im)^\s*(?:#+\s*)?(?:Official\s+Review|Reviewer\s*[#]?[A-Za-z0-9]+|"
    r"Review\s*[#]?[A-Za-z0-9]+)\b.*$"
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class Weakness:
    text: str
    severity: str = "MINOR"
    category: str = "text_fix"


@dataclass
class ReviewBlock:
    label: str
    scores: dict[str, float] = field(default_factory=dict)
    summary: str = ""
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[Weakness] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    limitations: str = ""
    raw: str = ""


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def split_reviewers(text: str) -> list[tuple[str, str]]:
    matches = list(REVIEWER_HEADER_RE.finditer(text))
    if not matches:
        return [("Reviewer", text.strip())]
    out: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        label = m.group(0).strip()
        body = text[start:end].strip()
        out.append((label, body))
    return out


# ---------------------------------------------------------------------------
# Per-block parsing
# ---------------------------------------------------------------------------

def _extract_scores(body: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    for key, pat in SCORE_PATTERNS.items():
        m = pat.search(body)
        if m:
            try:
                scores[key] = float(m.group(1))
            except ValueError:
                pass
    return scores


def _section_spans(body: str) -> list[tuple[str, int, int]]:
    hits: list[tuple[str, int]] = []
    for name, pat in SECTION_HEADINGS:
        for m in pat.finditer(body):
            hits.append((name, m.start()))
    hits.sort(key=lambda x: x[1])
    spans: list[tuple[str, int, int]] = []
    for i, (name, pos) in enumerate(hits):
        end = hits[i + 1][1] if i + 1 < len(hits) else len(body)
        spans.append((name, pos, end))
    return spans


def _bulletize(block: str) -> list[str]:
    out: list[str] = []
    current: list[str] = []
    for raw in block.splitlines():
        line = raw.rstrip()
        if not line.strip():
            if current:
                out.append(" ".join(current).strip())
                current = []
            continue
        if re.match(r"^\s*(?:[-*•\d]+[.)]?|\(\d+\))\s+", line):
            if current:
                out.append(" ".join(current).strip())
                current = []
            current.append(re.sub(r"^\s*(?:[-*•]|\d+[.)]|\(\d+\))\s+", "", line))
        else:
            if current:
                current.append(line.strip())
            # else: prose between bullets, drop
    if current:
        out.append(" ".join(current).strip())
    out = [x for x in out if len(x) > 3]
    return out


def _classify_severity(text: str) -> str:
    low = text.lower()
    if any(w in low for w in CRITICAL_WORDS):
        return "CRITICAL"
    if any(w in low for w in MAJOR_WORDS):
        return "MAJOR"
    return "MINOR"


def _classify_category(text: str) -> str:
    low = text.lower()
    if any(w in low for w in NEEDS_DATA_WORDS):
        return "needs_data"
    if any(w in low for w in NEEDS_BASELINE_WORDS):
        return "needs_baseline"
    if any(w in low for w in NEEDS_EXP_WORDS):
        return "needs_experiment"
    return "text_fix"


def parse_block(label: str, body: str) -> ReviewBlock:
    rb = ReviewBlock(label=label, raw=body)
    rb.scores = _extract_scores(body)
    spans = _section_spans(body)
    for name, start, end in spans:
        chunk = body[start:end].strip()
        # Strip the heading line itself
        chunk = re.sub(r"^[^\n]*\n", "", chunk, count=1)
        if name == "summary":
            rb.summary = " ".join(l.strip() for l in chunk.splitlines() if l.strip())[:1200]
        elif name == "strengths":
            rb.strengths = _bulletize(chunk)
        elif name == "weaknesses":
            weaknesses_text = _bulletize(chunk)
            for w in weaknesses_text:
                rb.weaknesses.append(Weakness(
                    text=w,
                    severity=_classify_severity(w),
                    category=_classify_category(w),
                ))
        elif name == "questions":
            rb.questions = _bulletize(chunk)
        elif name == "limitations":
            rb.limitations = " ".join(l.strip() for l in chunk.splitlines() if l.strip())[:800]
    return rb


# ---------------------------------------------------------------------------
# Merge + render
# ---------------------------------------------------------------------------

def mean(vs: list[float]) -> float | None:
    vs = [v for v in vs if isinstance(v, (int, float))]
    return sum(vs) / len(vs) if vs else None


def render_merged(blocks: list[ReviewBlock], round_no: int) -> str:
    lines = [f"# Round {round_no} review (imported from OpenReview)", ""]
    lines.append("## Scores")
    lines.append("")
    cols = ("rating", "soundness", "presentation", "contribution", "confidence")
    lines.append("| Reviewer | " + " | ".join(c.title() for c in cols) + " |")
    lines.append("|" + "|".join(["---"] * (len(cols) + 1)) + "|")
    for b in blocks:
        row = [b.label]
        for c in cols:
            v = b.scores.get(c)
            row.append(f"{v:g}" if v is not None else "—")
        lines.append("| " + " | ".join(row) + " |")
    means = {c: mean([b.scores.get(c, float("nan")) for b in blocks
                       if isinstance(b.scores.get(c), (int, float))]) for c in cols}
    lines.append("| **Mean** | " + " | ".join(
        f"{v:g}" if v is not None else "—" for v in means.values()
    ) + " |")
    lines.append("")
    for b in blocks:
        lines.append(f"## {b.label}")
        if b.summary:
            lines.append("")
            lines.append(f"**Summary.** {b.summary}")
        if b.strengths:
            lines.append("")
            lines.append("**Strengths.**")
            for s in b.strengths:
                lines.append(f"- {s}")
        if b.weaknesses:
            lines.append("")
            lines.append("**Weaknesses.**")
            for w in b.weaknesses:
                lines.append(f"- [{w.severity}/{w.category}] {w.text}")
        if b.questions:
            lines.append("")
            lines.append("**Questions.**")
            for q in b.questions:
                lines.append(f"- {q}")
        if b.limitations:
            lines.append("")
            lines.append(f"**Limitations.** {b.limitations}")
        lines.append("")
    return "\n".join(lines) + "\n"


def render_fix_plan(blocks: list[ReviewBlock], round_no: int) -> str:
    all_weaknesses: list[tuple[str, Weakness]] = []
    for b in blocks:
        for w in b.weaknesses:
            all_weaknesses.append((b.label, w))

    severity_rank = {"CRITICAL": 0, "MAJOR": 1, "MINOR": 2}
    all_weaknesses.sort(key=lambda t: (severity_rank[t[1].severity], t[1].category))

    lines = [f"# Fix plan — round {round_no}", ""]
    lines.append("Categories: `text_fix` = direct LaTeX edit, "
                 "`needs_experiment` = run with existing code+data, "
                 "`needs_data` = **ask user** before downloading, "
                 "`needs_baseline` = **ask user** before implementing.")
    lines.append("")
    for bucket in ("text_fix", "needs_experiment", "needs_data", "needs_baseline"):
        items = [t for t in all_weaknesses if t[1].category == bucket]
        if not items:
            continue
        header = {
            "text_fix": "## Text fixes",
            "needs_experiment": "## Experiments to run (no user checkpoint)",
            "needs_data": "## Needs new data — USER CHECKPOINT",
            "needs_baseline": "## Needs new baseline — USER CHECKPOINT",
        }[bucket]
        lines.append(header)
        lines.append("")
        for label, w in items:
            lines.append(f"- [{w.severity}] ({label}) {w.text}")
        lines.append("")
    lines.append("## Suggested ordering")
    lines.append("")
    lines.append("1. Address CRITICAL items first, regardless of bucket.")
    lines.append("2. Run `numcheck.py` and `bibcheck.py` before re-exporting.")
    lines.append("3. Increment main_vN.tex and commit.")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# State update
# ---------------------------------------------------------------------------

def update_state(paper_dir: Path, round_no: int, blocks: list[ReviewBlock],
                 merged_path: Path, fix_plan_path: Path,
                 target_score: float, mean_rating: float | None) -> dict:
    state_path = paper_dir / "PAPER_AUTORESEARCH_STATE.json"
    state: dict = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
        except json.JSONDecodeError:
            state = {}
    state["round"] = round_no
    state["last_review"] = str(merged_path)
    state["last_fix_plan"] = str(fix_plan_path)
    scores = state.setdefault("scores", [])
    entry = {
        "round": round_no,
        "reviewers": [{"label": b.label, "scores": b.scores} for b in blocks],
        "mean_rating": mean_rating,
        "timestamp": datetime.now().isoformat(),
    }
    # Replace if a row for this round already exists
    scores = [s for s in scores if s.get("round") != round_no]
    scores.append(entry)
    state["scores"] = scores
    if mean_rating is not None and mean_rating >= target_score:
        state["status"] = "done"
    else:
        state["status"] = "planning_fixes"
    state["timestamp"] = datetime.now().isoformat()
    state_path.write_text(json.dumps(state, indent=2))
    return state


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Import a pasted OpenReview review")
    ap.add_argument("--raw", required=True, type=Path,
                    help="Text file with the pasted review(s)")
    ap.add_argument("--round", required=True, type=int)
    ap.add_argument("--paper-dir", default=Path("paper"), type=Path)
    ap.add_argument("--target-score", type=float, default=6.0,
                    help="Mean overall-rating threshold to mark status=done")
    ns = ap.parse_args(argv)

    if not ns.raw.exists():
        print(f"[import] raw review not found: {ns.raw}", file=sys.stderr)
        return 2
    paper_dir: Path = ns.paper_dir
    (paper_dir / "reviews").mkdir(parents=True, exist_ok=True)

    text = ns.raw.read_text(errors="replace")
    reviewer_chunks = split_reviewers(text)
    blocks = [parse_block(label, body) for label, body in reviewer_chunks]

    merged_path = paper_dir / "reviews" / f"review_round{ns.round}.md"
    fix_plan_path = paper_dir / f"fix_plan_round{ns.round}.md"
    merged_path.write_text(render_merged(blocks, ns.round))
    fix_plan_path.write_text(render_fix_plan(blocks, ns.round))

    ratings = [b.scores.get("rating") for b in blocks
                if isinstance(b.scores.get("rating"), (int, float))]
    mean_rating = sum(ratings) / len(ratings) if ratings else None

    state = update_state(paper_dir, ns.round, blocks, merged_path,
                          fix_plan_path, ns.target_score, mean_rating)

    summary = {
        "round": ns.round,
        "reviewers": [b.label for b in blocks],
        "mean_rating": mean_rating,
        "target": ns.target_score,
        "status": state["status"],
        "merged": str(merged_path),
        "fix_plan": str(fix_plan_path),
        "weaknesses_by_bucket": {
            bucket: sum(1 for b in blocks for w in b.weaknesses
                         if w.category == bucket)
            for bucket in ("text_fix", "needs_experiment",
                            "needs_data", "needs_baseline")
        },
    }
    print(json.dumps(summary, indent=2))
    return 10 if state["status"] == "done" else 0


if __name__ == "__main__":
    sys.exit(main())
