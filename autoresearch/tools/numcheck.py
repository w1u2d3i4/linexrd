#!/usr/bin/env python3
"""
numcheck — cross-check every numeric literal in paper/*.tex against
experiment evidence (autoresearch.jsonl, results/*.json,
CLAIMS_FROM_RESULTS.md). Pure stdlib, no network.

Exit codes
  0  no unmatched number in a "claim zone" (abstract / tables / results section)
  1  one or more unmatched claim-zone numbers -> paper revision must fix them
  2  usage error

Usage
  python numcheck.py \
      --paper-dir paper/ \
      --evidence autoresearch.jsonl results/ CLAIMS_FROM_RESULTS.md \
      --tol-rel 0.01 --tol-abs 0.005 \
      --report paper/NUMCHECK_REPORT.md

The tool is a gate, not an oracle. It flags numbers that look like empirical
results and whose value it cannot locate anywhere in the evidence bundle.
Non-claim numbers (years, equation numbers, page counts) are skipped via
heuristics; false positives can be silenced by listing them in
paper/numcheck_ignore.txt (one "section:number" per line).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


# Match numbers in LaTeX prose, including:
#   leading-dot decimals (.626 == 0.626, very common in tables),
#   scientific notation,
#   optional sign, but NEVER consume a `-` that is a hyphen/range/dash
#   (1--2, 10--20, 400--4000, en-dashes, etc.)
NUM_RE = re.compile(
    r"(?<![\w\\.\-])"                      # not preceded by word, backslash, '.' or '-'
    r"([+-]?(?:\d+\.\d+|\.\d+|\d+)"         # signed integer / decimal / leading-dot decimal
    r"(?:[eE][+-]?\d+)?)"                  # optional scientific exponent
    r"(?![\w.])"                            # not followed by word or another dot
)
CITE_YEAR_CTX = re.compile(r"\\cite[a-z]*\{[^}]*\}|\\bibitem\{[^}]*\}")
PAGEREF_CTX = re.compile(r"\\(page)?ref\{[^}]*\}")
LABEL_CTX = re.compile(r"\\label\{[^}]*\}")
TABLE_BEGIN = re.compile(r"\\begin\{(table|tabular|table\*|tabularx)\}")
TABLE_END = re.compile(r"\\end\{(table|tabular|table\*|tabularx)\}")
COMMENT_RE = re.compile(r"(?<!\\)%.*$", re.MULTILINE)
MATHINLINE = re.compile(r"\$[^$]+\$")

CLAIM_SECTION_HINTS = (
    "abstract", "experiment", "result", "evaluation",
    "ablation", "comparison", "analysis",
)


@dataclass
class Number:
    value: float
    raw: str
    file: str
    line: int
    context: str
    in_table: bool = False
    in_claim_section: bool = False

    @property
    def is_claim(self) -> bool:
        return self.in_table or self.in_claim_section


@dataclass
class Evidence:
    value: float
    source: str
    path: str
    note: str = ""


@dataclass
class Report:
    numbers: list[Number] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    matched: list[tuple[Number, Evidence]] = field(default_factory=list)
    unmatched_claim: list[Number] = field(default_factory=list)
    unmatched_prose: list[Number] = field(default_factory=list)
    ignored: list[Number] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _is_year(v: float) -> bool:
    return 1500 <= v <= 2100 and float(int(v)) == v


def _is_small_count(v: float, raw: str) -> bool:
    # Integers 0..20 that are not percentages and not in tables are usually
    # not claims ("three baselines", "Section 4").
    if "." in raw or "e" in raw.lower():
        return False
    return 0 <= v <= 20


def _strip_noise(text: str) -> str:
    text = COMMENT_RE.sub("", text)
    text = CITE_YEAR_CTX.sub("<CITE>", text)
    text = PAGEREF_CTX.sub("<REF>", text)
    text = LABEL_CTX.sub("<LABEL>", text)
    return text


def _extract_from_tex(path: Path, is_claim_section: bool) -> list[Number]:
    out: list[Number] = []
    depth = 0
    for i, raw_line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        line = _strip_noise(raw_line)
        if TABLE_BEGIN.search(raw_line):
            depth += 1
        for m in NUM_RE.finditer(line):
            raw = m.group(1)
            try:
                val = float(raw)
            except ValueError:
                continue
            out.append(Number(
                value=val,
                raw=raw,
                file=str(path),
                line=i,
                context=raw_line.strip()[:200],
                in_table=depth > 0,
                in_claim_section=is_claim_section,
            ))
        if TABLE_END.search(raw_line) and depth > 0:
            depth -= 1
    return out


def collect_tex_numbers(paper_dir: Path) -> list[Number]:
    targets: list[tuple[Path, bool]] = []
    for sub in ("", "sections"):
        d = paper_dir / sub if sub else paper_dir
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.tex")):
            stem = p.stem.lower()
            is_claim = any(h in stem for h in CLAIM_SECTION_HINTS)
            targets.append((p, is_claim))
    numbers: list[Number] = []
    for p, is_claim in targets:
        numbers.extend(_extract_from_tex(p, is_claim))
    return numbers


# ---------------------------------------------------------------------------
# Evidence collection
# ---------------------------------------------------------------------------

def _walk_json(obj, out: list[Evidence], source: str, path_repr: str):
    if isinstance(obj, dict):
        for k, v in obj.items():
            _walk_json(v, out, source, f"{path_repr}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _walk_json(v, out, source, f"{path_repr}[{i}]")
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        try:
            f = float(obj)
        except (TypeError, ValueError):
            return
        if f != f:  # nan
            return
        out.append(Evidence(value=f, source=source, path=path_repr))


def collect_evidence(paths: Iterable[Path]) -> list[Evidence]:
    out: list[Evidence] = []
    for p in paths:
        if not p.exists():
            continue
        if p.is_dir():
            for q in sorted(p.rglob("*.json")):
                _ingest_json_file(q, out)
            for q in sorted(p.rglob("*.jsonl")):
                _ingest_jsonl_file(q, out)
            for q in sorted(p.rglob("*.md")):
                _ingest_md_file(q, out)
        elif p.suffix == ".jsonl":
            _ingest_jsonl_file(p, out)
        elif p.suffix == ".json":
            _ingest_json_file(p, out)
        elif p.suffix in {".md", ".txt"}:
            _ingest_md_file(p, out)
    return out


def _ingest_json_file(p: Path, out: list[Evidence]):
    try:
        data = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return
    _walk_json(data, out, source=str(p), path_repr=p.name)


def _ingest_jsonl_file(p: Path, out: list[Evidence]):
    try:
        for i, ln in enumerate(p.read_text().splitlines()):
            ln = ln.strip()
            if not ln:
                continue
            try:
                data = json.loads(ln)
            except json.JSONDecodeError:
                continue
            _walk_json(data, out, source=str(p), path_repr=f"{p.name}[{i}]")
    except OSError:
        return


_MD_NUM_RE = re.compile(r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")


def _ingest_md_file(p: Path, out: list[Evidence]):
    try:
        text = p.read_text()
    except OSError:
        return
    for i, line in enumerate(text.splitlines()):
        for m in _MD_NUM_RE.finditer(line):
            try:
                v = float(m.group(1))
            except ValueError:
                continue
            out.append(Evidence(
                value=v, source=str(p), path=f"{p.name}:L{i+1}",
                note=line.strip()[:120],
            ))


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def load_ignore(paper_dir: Path) -> set[tuple[str, str]]:
    ignore: set[tuple[str, str]] = set()
    f = paper_dir / "numcheck_ignore.txt"
    if not f.exists():
        return ignore
    for line in f.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        sec, num = line.split(":", 1)
        ignore.add((sec.strip(), num.strip()))
    return ignore


def match_numbers(numbers: list[Number], evidence: list[Evidence],
                  tol_rel: float, tol_abs: float,
                  ignore: set[tuple[str, str]]) -> Report:
    rep = Report(numbers=numbers, evidence=evidence)
    for n in numbers:
        key = (Path(n.file).stem, n.raw)
        if key in ignore:
            rep.ignored.append(n)
            continue
        # Fast skips for obvious non-claims even in claim sections
        if _is_year(n.value) and "%" not in n.context:
            rep.ignored.append(n)
            continue
        # Try matching against any evidence number
        matched_ev = _best_match(n.value, evidence, tol_rel, tol_abs)
        # Percent vs fraction (0.857 vs 85.7): also match /100 and *100
        if matched_ev is None:
            matched_ev = _best_match(n.value / 100.0, evidence, tol_rel, tol_abs) \
                         or _best_match(n.value * 100.0, evidence, tol_rel, tol_abs)
        if matched_ev is not None:
            rep.matched.append((n, matched_ev))
            continue
        if n.is_claim:
            if _is_small_count(n.value, n.raw) and not n.in_table:
                rep.ignored.append(n)
            else:
                rep.unmatched_claim.append(n)
        else:
            rep.unmatched_prose.append(n)
    return rep


def _best_match(v: float, evidence: list[Evidence], tol_rel: float,
                tol_abs: float) -> Evidence | None:
    best: Evidence | None = None
    best_err = float("inf")
    for ev in evidence:
        err = abs(ev.value - v)
        ref = max(abs(v), abs(ev.value), 1e-12)
        if err <= tol_abs or err / ref <= tol_rel:
            if err < best_err:
                best_err = err
                best = ev
    return best


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def render_report(rep: Report) -> str:
    lines = ["# Numcheck Report", ""]
    total = len(rep.numbers)
    lines.append(f"- Numbers scanned: **{total}**")
    lines.append(f"- Matched to evidence: **{len(rep.matched)}**")
    lines.append(f"- Ignored (years/counts/whitelist): **{len(rep.ignored)}**")
    lines.append(f"- Unmatched in prose (warning): **{len(rep.unmatched_prose)}**")
    lines.append(f"- **Unmatched in claim zone (blocker): {len(rep.unmatched_claim)}**")
    lines.append("")
    if rep.unmatched_claim:
        lines.append("## Blocking: unmatched numbers in abstract / tables / results")
        lines.append("")
        lines.append("| File | Line | Number | Context |")
        lines.append("| --- | --- | --- | --- |")
        for n in rep.unmatched_claim:
            ctx = n.context.replace("|", r"\|")
            lines.append(f"| {n.file} | {n.line} | `{n.raw}` | {ctx} |")
        lines.append("")
    if rep.unmatched_prose:
        lines.append("## Warning: unmatched numbers in prose sections")
        lines.append("")
        for n in rep.unmatched_prose[:50]:
            lines.append(f"- {n.file}:L{n.line} `{n.raw}` — {n.context}")
        if len(rep.unmatched_prose) > 50:
            lines.append(f"- ... ({len(rep.unmatched_prose) - 50} more)")
        lines.append("")
    lines.append("## How to silence a false positive")
    lines.append("")
    lines.append("Add `<section-stem>:<number>` to `paper/numcheck_ignore.txt` "
                 "(one per line). Example: `4_experiments:9` to whitelist the"
                 " integer 9 in `sections/4_experiments.tex`.")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Numeric claim checker")
    ap.add_argument("--paper-dir", default="paper", type=Path)
    ap.add_argument("--evidence", nargs="*", default=[],
                    help="Files or directories with ground-truth numbers "
                    "(autoresearch.jsonl, results/, CLAIMS_FROM_RESULTS.md, ...)")
    ap.add_argument("--tol-rel", type=float, default=0.01,
                    help="Relative tolerance (default 1%%)")
    ap.add_argument("--tol-abs", type=float, default=0.005,
                    help="Absolute tolerance (default 0.005)")
    ap.add_argument("--report", type=Path, default=None,
                    help="Path to write Markdown report (default paper/NUMCHECK_REPORT.md)")
    ap.add_argument("--quiet", action="store_true")
    ns = ap.parse_args(argv)

    paper_dir: Path = ns.paper_dir
    if not paper_dir.is_dir():
        print(f"[numcheck] paper dir not found: {paper_dir}", file=sys.stderr)
        return 2

    numbers = collect_tex_numbers(paper_dir)
    evidence_paths = [Path(p) for p in ns.evidence] or _default_evidence(paper_dir)
    evidence = collect_evidence(evidence_paths)
    ignore = load_ignore(paper_dir)
    rep = match_numbers(numbers, evidence, ns.tol_rel, ns.tol_abs, ignore)

    report_path = ns.report or paper_dir / "NUMCHECK_REPORT.md"
    report_path.write_text(render_report(rep))

    summary = {
        "scanned": len(rep.numbers),
        "matched": len(rep.matched),
        "ignored": len(rep.ignored),
        "unmatched_prose": len(rep.unmatched_prose),
        "unmatched_claim": len(rep.unmatched_claim),
        "report": str(report_path),
    }
    if not ns.quiet:
        print(json.dumps(summary, indent=2))
    return 1 if rep.unmatched_claim else 0


def _default_evidence(paper_dir: Path) -> list[Path]:
    root = paper_dir.parent
    out = []
    for cand in ("autoresearch.jsonl", "result.md", "result_final.md",
                 "results", "CLAIMS_FROM_RESULTS.md",
                 str(paper_dir / "CLAIMS_FROM_RESULTS.md")):
        p = (root / cand) if not os.path.isabs(cand) else Path(cand)
        out.append(p)
    return out


if __name__ == "__main__":
    sys.exit(main())
