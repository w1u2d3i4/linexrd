#!/usr/bin/env python3
"""
rebuttal_compile — local helpers for paper-autoresearch rebuttal flow.

What it does:

  1. point-to-point   build a markdown table that pairs every reviewer
                      weakness/question (from review_round{N}.md) with the
                      author response section in rebuttal_draft.md.
  2. char_count       per-reviewer + total character count vs venue cap.
  3. numcheck         re-run numcheck.py against the rebuttal text only —
                      every numeric claim added in the rebuttal must trace
                      to existing evidence.

No network. Pure stdlib. Reuses numcheck.py via subprocess if --numcheck.

Usage
  # full pipeline:
  python rebuttal_compile.py compile \
      --paper-dir paper/ \
      --round 2 \
      --char-cap 5000        # per reviewer (CVPR=5000, NeurIPS=6000)

  # individual subcommands:
  python rebuttal_compile.py p2p     --paper-dir paper/ --round 2
  python rebuttal_compile.py count   --paper-dir paper/ --round 2 --char-cap 5000
  python rebuttal_compile.py numcheck --paper-dir paper/ --round 2

Inputs
  paper/reviews/review_round{N}.md       (from import_review.py)
  paper/rebuttal/round_{N}/rebuttal_draft.md
        markdown structured as
            ## Reviewer 1
            ### W1
            response text…
            ### W2
            response text…
            ## Reviewer 2
            …

Outputs
  paper/rebuttal/round_{N}/point_to_point_table.md
  paper/rebuttal/round_{N}/char_count.txt
  paper/rebuttal/round_{N}/REBUTTAL_NUMCHECK_REPORT.md  (if --numcheck)

Exit codes
  0  all subcommands ok (and within char cap, and numcheck clean)
  1  one or more checks failed (over cap / unmatched numbers / missing pairing)
  2  usage error
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


REVIEWER_HEADER = re.compile(r"^##\s+(Reviewer\s+[0-9A-Za-z]+|Reviewer\s+#?\d+)",
                              re.MULTILINE)
WEAKNESS_BULLET = re.compile(
    r"^-\s*\[(?P<sev>CRITICAL|MAJOR|MINOR)/(?P<cat>[\w_]+)\]\s*(?P<text>.+?)\s*$",
    re.MULTILINE,
)
QUESTION_BULLET = re.compile(r"^-\s*(.+?)\s*$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Parse review_round{N}.md
# ---------------------------------------------------------------------------

@dataclass
class ReviewItem:
    reviewer: str
    kind: str          # "weakness" | "question"
    severity: str      # CRITICAL/MAJOR/MINOR (questions: "—")
    category: str      # text_fix/needs_experiment/...
    text: str
    short_id: str      # "W1" / "Q2"


def parse_review_md(path: Path) -> list[ReviewItem]:
    if not path.exists():
        return []
    text = path.read_text(errors="ignore")
    # Split on ## Reviewer headers
    out: list[ReviewItem] = []
    parts = re.split(r"(?m)^##\s+", text)
    for part in parts[1:]:
        first_line, _, body = part.partition("\n")
        reviewer = first_line.strip()
        # Extract weaknesses block
        wmatch = re.search(r"\*\*Weaknesses\.\*\*\s*\n(?P<body>.*?)(?:\n\*\*|\Z)",
                            body, re.DOTALL)
        widx = 0
        if wmatch:
            for m in WEAKNESS_BULLET.finditer(wmatch.group("body")):
                widx += 1
                out.append(ReviewItem(
                    reviewer=reviewer,
                    kind="weakness",
                    severity=m.group("sev"),
                    category=m.group("cat"),
                    text=m.group("text").strip(),
                    short_id=f"W{widx}",
                ))
        qmatch = re.search(r"\*\*Questions\.\*\*\s*\n(?P<body>.*?)(?:\n\*\*|\Z)",
                            body, re.DOTALL)
        qidx = 0
        if qmatch:
            for line in qmatch.group("body").splitlines():
                bm = re.match(r"\s*-\s*(.+)", line)
                if not bm:
                    continue
                qidx += 1
                out.append(ReviewItem(
                    reviewer=reviewer,
                    kind="question",
                    severity="—",
                    category="text_fix",
                    text=bm.group(1).strip(),
                    short_id=f"Q{qidx}",
                ))
    return out


# ---------------------------------------------------------------------------
# Parse rebuttal_draft.md
# ---------------------------------------------------------------------------

@dataclass
class RebuttalSection:
    reviewer: str
    short_id: str        # "W1" / "Q2"
    body: str


SECTION_HEADER = re.compile(r"^###\s+(?P<sid>[WQ]\d+)\b[^\n]*$", re.MULTILINE)


def parse_rebuttal(draft: Path) -> dict[str, dict[str, str]]:
    """Return {reviewer: {short_id: body}}"""
    if not draft.exists():
        return {}
    text = draft.read_text(errors="ignore")
    out: dict[str, dict[str, str]] = {}
    parts = re.split(r"(?m)^##\s+", text)
    for part in parts[1:]:
        first_line, _, body = part.partition("\n")
        reviewer = first_line.strip()
        out.setdefault(reviewer, {})
        sec_starts = list(SECTION_HEADER.finditer(body))
        for i, m in enumerate(sec_starts):
            sid = m.group("sid")
            start = m.end()
            end = sec_starts[i + 1].start() if i + 1 < len(sec_starts) else len(body)
            chunk = body[start:end].strip()
            out[reviewer][sid] = chunk
    return out


# ---------------------------------------------------------------------------
# Subcommand: point-to-point
# ---------------------------------------------------------------------------

def cmd_p2p(round_dir: Path, review_md: Path, draft: Path) -> int:
    items = parse_review_md(review_md)
    rebuttal = parse_rebuttal(draft)
    if not items:
        print(f"[rebuttal] no items parsed from {review_md}", file=sys.stderr)
        return 1
    lines = [
        "# Point-to-point response table",
        "",
        f"Source review: `{review_md}`",
        f"Source rebuttal: `{draft}`",
        "",
        "| Reviewer | ID | Sev | Category | Issue | Response (excerpt) | Status |",
        "|---|---|---|---|---|---|---|",
    ]
    missing = 0
    for it in items:
        resp = rebuttal.get(it.reviewer, {}).get(it.short_id, "")
        if not resp:
            # Try variant labels (e.g. "Reviewer 1" vs "Reviewer #1")
            for k in rebuttal:
                if k.replace("#", "").replace(" ", "").lower() == \
                   it.reviewer.replace("#", "").replace(" ", "").lower():
                    resp = rebuttal[k].get(it.short_id, "")
                    if resp:
                        break
        status = "✅" if resp else "❌ MISSING"
        if not resp:
            missing += 1
        excerpt = resp.strip().split("\n", 1)[0][:120].replace("|", "/")
        text_short = it.text[:120].replace("|", "/")
        lines.append(f"| {it.reviewer} | {it.short_id} | {it.severity} | "
                      f"{it.category} | {text_short} | {excerpt or '—'} | {status} |")
    out = round_dir / "point_to_point_table.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "p2p_table": str(out),
        "items": len(items),
        "missing_responses": missing,
    }, indent=2))
    return 1 if missing else 0


# ---------------------------------------------------------------------------
# Subcommand: char count
# ---------------------------------------------------------------------------

def visible_chars(text: str) -> int:
    """Approximate visible char count: strip markdown decoration."""
    t = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)   # headings
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)              # bold
    t = re.sub(r"\*([^*]+)\*", r"\1", t)                   # italic
    t = re.sub(r"`([^`]+)`", r"\1", t)                     # inline code
    t = re.sub(r"\s+", " ", t).strip()
    return len(t)


def cmd_count(round_dir: Path, draft: Path, char_cap: int) -> int:
    rebuttal = parse_rebuttal(draft)
    if not rebuttal:
        print(f"[rebuttal] empty/unparseable rebuttal: {draft}", file=sys.stderr)
        return 1
    out_lines: list[str] = ["per-reviewer character counts (visible chars):"]
    over = 0
    for reviewer, sections in rebuttal.items():
        full = "\n".join(sections.values())
        n = visible_chars(full)
        flag = f"  OVER CAP ({n} > {char_cap})" if n > char_cap else ""
        if n > char_cap:
            over += 1
        out_lines.append(f"  {reviewer}: {n} chars{flag}")
    out_lines.append("")
    out_lines.append(f"cap_per_reviewer={char_cap}, reviewers_over={over}")
    out = round_dir / "char_count.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(out_lines) + "\n")
    print(json.dumps({
        "char_count_file": str(out),
        "reviewers_over": over,
        "char_cap_per_reviewer": char_cap,
    }, indent=2))
    return 1 if over else 0


# ---------------------------------------------------------------------------
# Subcommand: numcheck on rebuttal
# ---------------------------------------------------------------------------

def cmd_numcheck(round_dir: Path, draft: Path, paper_dir: Path,
                  workdir: Path) -> int:
    """Re-run numcheck.py against the rebuttal text (treated as a synthetic
    .tex). The check verifies that every numeric literal in the rebuttal
    traces to existing evidence files; new numbers must come from existing
    runs, not invented for the rebuttal."""
    if not draft.exists():
        print(f"[rebuttal] draft missing: {draft}", file=sys.stderr)
        return 1
    # Build a synthetic tex with sectioning so claim-zone heuristic activates
    synth_tex = round_dir / "_rebuttal_synth.tex"
    body = draft.read_text(errors="ignore")
    synth = (
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "\\section{Results}\n"  # force claim zone
        f"{body}\n"
        "\\end{document}\n"
    )
    synth_tex.parent.mkdir(parents=True, exist_ok=True)
    synth_tex.write_text(synth)
    # Mirror into a temporary paper-dir-like structure so numcheck can scan it
    synth_paper = round_dir / "_rebuttal_paper"
    synth_paper.mkdir(exist_ok=True)
    (synth_paper / "main_v0.tex").write_text(synth)
    nc = Path(__file__).parent / "numcheck.py"
    if not nc.exists():
        print(f"[rebuttal] numcheck.py not found at {nc}", file=sys.stderr)
        return 2
    # Reuse evidence files of the paper
    evidence: list[str] = []
    for cand in [
        workdir / "autoresearch.jsonl",
        workdir / "results",
        paper_dir / "CLAIMS_FROM_RESULTS.md",
    ]:
        if cand.exists():
            evidence.append(str(cand))
    report_path = round_dir / "REBUTTAL_NUMCHECK_REPORT.md"
    cmd = [
        "python3", str(nc),
        "--paper-dir", str(synth_paper),
        "--evidence", *evidence,
        "--report", str(report_path),
    ]
    rc = subprocess.run(cmd).returncode
    print(json.dumps({
        "numcheck_report": str(report_path),
        "exit_code": rc,
    }, indent=2))
    return 0 if rc == 0 else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def round_dir(paper_dir: Path, n: int) -> Path:
    return paper_dir / "rebuttal" / f"round_{n}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="sub", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--paper-dir", required=True)
    common.add_argument("--round", "-r", type=int, required=True)
    common.add_argument("--workdir", default=".")

    p = sub.add_parser("p2p", parents=[common])
    p = sub.add_parser("count", parents=[common])
    p.add_argument("--char-cap", type=int, default=5000,
                    help="Per-reviewer cap (CVPR=5000, NeurIPS=6000, AAAI=4000)")
    p = sub.add_parser("numcheck", parents=[common])
    p = sub.add_parser("compile", parents=[common])
    p.add_argument("--char-cap", type=int, default=5000)

    args = ap.parse_args(argv)
    paper_dir = Path(args.paper_dir).resolve()
    workdir = Path(args.workdir).resolve()
    rdir = round_dir(paper_dir, args.round)
    review_md = paper_dir / "reviews" / f"review_round{args.round}.md"
    draft = rdir / "rebuttal_draft.md"

    if args.sub == "p2p":
        return cmd_p2p(rdir, review_md, draft)
    if args.sub == "count":
        return cmd_count(rdir, draft, args.char_cap)
    if args.sub == "numcheck":
        return cmd_numcheck(rdir, draft, paper_dir, workdir)
    if args.sub == "compile":
        rcs = [
            cmd_p2p(rdir, review_md, draft),
            cmd_count(rdir, draft, args.char_cap),
            cmd_numcheck(rdir, draft, paper_dir, workdir),
        ]
        return max(rcs)
    return 2


if __name__ == "__main__":
    sys.exit(main())
