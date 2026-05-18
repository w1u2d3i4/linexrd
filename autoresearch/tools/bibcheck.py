#!/usr/bin/env python3
"""
bibcheck — local bibliography hygiene gate. No network.

Checks
  1. Every \\cite{} / \\citep{} / \\citet{} / \\citeauthor{} / \\citeyear{}
     key referenced from paper/ has a matching entry in references.bib.
  2. Every .bib entry is actually cited somewhere (no bib bloat).
  3. Each cited entry has the minimum required fields (author, title, year,
     and one of: booktitle / journal / publisher / howpublished / school).
  4. No duplicate bib keys.
  5. No leftover [VERIFY] / FIXME / TODO markers in bib or .tex.
  6. Consistent year (year in bib key matches `year` field, if the key
     looks like "name2024word"; soft check).

Exit codes
  0  no issues
  1  at least one blocker (undefined key, missing field, duplicate key,
     [VERIFY] marker still present)
  2  usage error

Usage
  python bibcheck.py --paper-dir paper/ --bib paper/references.bib \\
         --report paper/BIBCHECK_REPORT.md

A JSON summary is printed to stdout.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


CITE_RE = re.compile(
    r"\\cite(?:t|p|author|year|alt|num|alp)?\*?"
    r"(?:\[[^\]]*\])?(?:\[[^\]]*\])?\{([^}]+)\}"
)
VERIFY_RE = re.compile(r"\[VERIFY\]|\bFIXME\b|\bTODO\b|\bXXX\b")
COMMENT_RE = re.compile(r"(?<!\\)%.*$", re.MULTILINE)

BIB_ENTRY_RE = re.compile(
    r"@(?P<type>[A-Za-z]+)\s*\{\s*(?P<key>[^,\s]+)\s*,"
)
BIB_FIELD_RE = re.compile(
    r"(?P<name>[A-Za-z]+)\s*=\s*(?:\{(?P<braced>[^{}]*(?:\{[^{}]*\}[^{}]*)*)\}|"
    r"\"(?P<quoted>[^\"]*)\"|(?P<bare>\w+))",
    re.DOTALL,
)

REQUIRED_BY_TYPE = {
    "article": [{"author"}, {"title"}, {"year"}, {"journal"}],
    "inproceedings": [{"author"}, {"title"}, {"year"}, {"booktitle"}],
    "conference": [{"author"}, {"title"}, {"year"}, {"booktitle"}],
    "book": [{"author", "editor"}, {"title"}, {"year"}, {"publisher"}],
    "incollection": [{"author"}, {"title"}, {"year"}, {"booktitle"}],
    "techreport": [{"author"}, {"title"}, {"year"}, {"institution"}],
    "mastersthesis": [{"author"}, {"title"}, {"year"}, {"school"}],
    "phdthesis": [{"author"}, {"title"}, {"year"}, {"school"}],
    "misc": [{"author"}, {"title"}, {"year"}],
    "unpublished": [{"author"}, {"title"}, {"note"}],
}
# Fallback requirement for any type not listed above.
DEFAULT_REQUIREMENT = [{"author"}, {"title"}, {"year"}]


@dataclass
class BibEntry:
    key: str
    type: str
    fields: dict[str, str]
    line: int


@dataclass
class Report:
    cited: set[str] = field(default_factory=set)
    defined: dict[str, BibEntry] = field(default_factory=dict)
    duplicates: list[str] = field(default_factory=list)
    undefined: list[tuple[str, str, int]] = field(default_factory=list)
    uncited: list[str] = field(default_factory=list)
    missing_fields: list[tuple[str, list[str]]] = field(default_factory=list)
    verify_markers: list[tuple[str, int, str]] = field(default_factory=list)
    key_year_mismatch: list[tuple[str, str]] = field(default_factory=list)
    # --strict-traceable additions:
    traceable_failures: list[tuple[str, str]] = field(default_factory=list)
    anonymous_authors: list[str] = field(default_factory=list)
    recency_warning: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_bib(path: Path) -> tuple[dict[str, BibEntry], list[str]]:
    text = path.read_text(errors="replace")
    entries: dict[str, BibEntry] = {}
    duplicates: list[str] = []
    # Split on @ at start of entry; handle nested braces by scanning.
    i = 0
    while i < len(text):
        m = BIB_ENTRY_RE.search(text, i)
        if not m:
            break
        key = m.group("key").strip()
        etype = m.group("type").lower()
        # Locate the body between braces starting right after the opening '{'
        body_start = m.end()
        # We already consumed `@type{key,`; find matching brace pair
        # starting from the `{` that followed @type.
        depth = 1
        j = body_start
        start_brace = text.rfind("{", m.start(), body_start)
        if start_brace == -1:
            i = m.end()
            continue
        j = start_brace + 1
        while j < len(text) and depth > 0:
            ch = text[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            j += 1
        body = text[body_start:j - 1]
        fields = {}
        for fm in BIB_FIELD_RE.finditer(body):
            name = fm.group("name").lower()
            val = fm.group("braced") or fm.group("quoted") or fm.group("bare") or ""
            fields[name] = val.strip()
        line = text[: m.start()].count("\n") + 1
        entry = BibEntry(key=key, type=etype, fields=fields, line=line)
        if key in entries:
            duplicates.append(key)
        else:
            entries[key] = entry
        i = j
    return entries, duplicates


def collect_cites(paper_dir: Path) -> list[tuple[str, str, int]]:
    out: list[tuple[str, str, int]] = []
    for p in list(paper_dir.rglob("*.tex")):
        text = p.read_text(errors="replace")
        text_nc = COMMENT_RE.sub("", text)
        for idx, line in enumerate(text_nc.splitlines(), 1):
            for m in CITE_RE.finditer(line):
                keys = [k.strip() for k in m.group(1).split(",") if k.strip()]
                for k in keys:
                    out.append((k, str(p), idx))
    return out


def find_verify_markers(paper_dir: Path, bib_path: Path | None) -> list[tuple[str, int, str]]:
    out: list[tuple[str, int, str]] = []
    targets = list(paper_dir.rglob("*.tex"))
    if bib_path:
        targets.append(bib_path)
    for p in targets:
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if VERIFY_RE.search(line):
                out.append((str(p), i, line.strip()[:160]))
    return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_KEY_YEAR_RE = re.compile(r"(\d{4})")


def validate(paper_dir: Path, bib_path: Path, strict_traceable: bool = False,
             current_year: int | None = None) -> Report:
    rep = Report()
    entries, duplicates = parse_bib(bib_path)
    rep.defined = entries
    rep.duplicates = duplicates

    cite_hits = collect_cites(paper_dir)
    cited = {k for k, _, _ in cite_hits}
    rep.cited = cited

    for key, file, line in cite_hits:
        if key not in entries:
            rep.undefined.append((key, file, line))

    recent_cnt = 0
    cited_with_year = 0

    for key, entry in entries.items():
        if key not in cited:
            rep.uncited.append(key)

        # Required fields
        reqs = REQUIRED_BY_TYPE.get(entry.type, DEFAULT_REQUIREMENT)
        missing = []
        for alt in reqs:
            if not any(f in entry.fields and entry.fields[f].strip() for f in alt):
                missing.append(" or ".join(sorted(alt)))
        if missing:
            rep.missing_fields.append((key, missing))

        # Key year vs field year
        m = _KEY_YEAR_RE.search(key)
        if m and entry.fields.get("year"):
            yr = re.search(r"\d{4}", entry.fields["year"])
            if yr and yr.group(0) != m.group(1):
                rep.key_year_mismatch.append((key, yr.group(0)))

        # --strict-traceable additions
        if strict_traceable:
            _traceable_check(key, entry, rep, cited, current_year,
                             recent_counter=lambda yr_int: None)

    rep.verify_markers = find_verify_markers(paper_dir, bib_path)

    # Recency stats: only for cited entries with a parseable year
    if strict_traceable and current_year is not None:
        for key in cited:
            entry = entries.get(key)
            if not entry:
                continue
            yr_match = re.search(r"\d{4}", entry.fields.get("year", ""))
            if not yr_match:
                continue
            cited_with_year += 1
            yr = int(yr_match.group(0))
            if current_year - yr <= 5:
                recent_cnt += 1
        rep.recency_warning = {
            "cited_with_year": cited_with_year,
            "within_last_5_years": recent_cnt,
            "ratio_pct": round(100 * recent_cnt / cited_with_year, 1)
            if cited_with_year else 0.0,
            "threshold_pct": 30.0,
        }
    return rep


# ---------------------------------------------------------------------------
# --strict-traceable helpers
# ---------------------------------------------------------------------------

_ANON_RE = re.compile(r"^\s*(anonymous|et\s*\.?\s*al\.?)\s*$", re.IGNORECASE)
_VENUE_KEYS = ("journal", "booktitle", "publisher", "school",
               "institution", "url", "howpublished", "note")


def _traceable_check(key: str, entry: "BibEntry", rep: Report,
                     cited: set, current_year, recent_counter) -> None:
    """Apply the Norman two-of-three rule + Anonymous blocker.

    Only flags *cited* entries — uncited noise is not a blocker."""
    if key not in cited:
        return
    flds = entry.fields

    # Two-of-three traceability: author, year, venue
    has_author = bool(flds.get("author", "").strip()
                      or flds.get("editor", "").strip())
    has_year = bool(re.search(r"\d{4}", flds.get("year", "")))
    has_venue = any(flds.get(k, "").strip() for k in _VENUE_KEYS)
    score = sum([has_author, has_year, has_venue])
    if score < 2:
        missing_axes = []
        if not has_author:
            missing_axes.append("author")
        if not has_year:
            missing_axes.append("year")
        if not has_venue:
            missing_axes.append("venue/url")
        rep.traceable_failures.append((key, ", ".join(missing_axes)))

    # Anonymous / et al. blocker
    author_value = flds.get("author", "").strip()
    if author_value and _ANON_RE.match(author_value):
        rep.anonymous_authors.append(key)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render(rep: Report) -> str:
    out = ["# Bibcheck Report", ""]
    out.append(f"- Cited keys: **{len(rep.cited)}**")
    out.append(f"- Defined entries: **{len(rep.defined)}**")
    out.append(f"- Duplicate keys: **{len(rep.duplicates)}**")
    out.append(f"- Undefined (blocker): **{len(rep.undefined)}**")
    out.append(f"- Uncited entries (warning): **{len(rep.uncited)}**")
    out.append(f"- Missing required fields (blocker): **{len(rep.missing_fields)}**")
    out.append(f"- VERIFY / FIXME markers (blocker): **{len(rep.verify_markers)}**")
    out.append(f"- Key-year mismatches (warning): **{len(rep.key_year_mismatch)}**")
    if rep.traceable_failures or rep.anonymous_authors or rep.recency_warning:
        out.append(f"- Traceability failures (blocker): **{len(rep.traceable_failures)}**")
        out.append(f"- Anonymous / et-al authors (blocker): **{len(rep.anonymous_authors)}**")
        if rep.recency_warning:
            rw = rep.recency_warning
            out.append(
                f"- 5-yr recency ratio: **{rw['ratio_pct']}%** "
                f"({rw['within_last_5_years']}/{rw['cited_with_year']}, "
                f"threshold {rw['threshold_pct']}%)"
            )
    out.append("")
    if rep.undefined:
        out.append("## Undefined citation keys")
        out.append("")
        for key, file, line in rep.undefined:
            out.append(f"- `{key}` @ {file}:L{line}")
        out.append("")
    if rep.duplicates:
        out.append("## Duplicate bib keys")
        out.append("")
        for k in rep.duplicates:
            out.append(f"- `{k}`")
        out.append("")
    if rep.missing_fields:
        out.append("## Entries missing required fields")
        out.append("")
        for k, ms in rep.missing_fields:
            out.append(f"- `{k}` missing: {', '.join(ms)}")
        out.append("")
    if rep.verify_markers:
        out.append("## Leftover VERIFY / FIXME / TODO markers")
        out.append("")
        for f, ln, ctx in rep.verify_markers:
            out.append(f"- {f}:L{ln} — {ctx}")
        out.append("")
    if rep.uncited:
        out.append("## Uncited bib entries (consider deleting)")
        out.append("")
        out.append(", ".join(f"`{k}`" for k in rep.uncited))
        out.append("")
    if rep.key_year_mismatch:
        out.append("## Key-year vs field-year mismatch")
        out.append("")
        for k, y in rep.key_year_mismatch:
            out.append(f"- `{k}` has year field {y}")
        out.append("")
    if rep.traceable_failures:
        out.append("## Traceability failures (strict-traceable)")
        out.append("")
        out.append("Each cited entry must have at least two of "
                   "{author, year, venue/url} populated.")
        out.append("")
        for k, axes in rep.traceable_failures:
            out.append(f"- `{k}` missing: {axes}")
        out.append("")
    if rep.anonymous_authors:
        out.append("## Anonymous / et-al author fields (strict-traceable)")
        out.append("")
        for k in rep.anonymous_authors:
            out.append(f"- `{k}` has `author = Anonymous` or `et al.`")
        out.append("")
    if rep.recency_warning:
        rw = rep.recency_warning
        if rw["cited_with_year"] and rw["ratio_pct"] < rw["threshold_pct"]:
            out.append("## Recency warning (strict-traceable)")
            out.append("")
            out.append(
                f"Only {rw['within_last_5_years']}/{rw['cited_with_year']} "
                f"({rw['ratio_pct']}%) of cited entries are from the last "
                f"5 years; threshold is {rw['threshold_pct']}%. Consider "
                f"adding more recent references for active subfields."
            )
            out.append("")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Local bibliography hygiene checker")
    ap.add_argument("--paper-dir", default="paper", type=Path)
    ap.add_argument("--bib", default=None, type=Path,
                    help="references.bib path (default: paper/references.bib)")
    ap.add_argument("--report", default=None, type=Path,
                    help="Markdown report (default: paper/BIBCHECK_REPORT.md)")
    ap.add_argument("--strict-traceable", action="store_true",
                    help="Norman two-of-three field rule + Anonymous "
                         "blocker + recency warning. Additional blockers.")
    ap.add_argument("--current-year", type=int, default=None,
                    help="Reference year for 5-yr recency calc "
                         "(default: current calendar year).")
    ap.add_argument("--quiet", action="store_true")
    ns = ap.parse_args(argv)

    paper_dir = ns.paper_dir
    if not paper_dir.is_dir():
        print(f"[bibcheck] paper dir not found: {paper_dir}", file=sys.stderr)
        return 2
    bib = ns.bib or paper_dir / "references.bib"
    if not bib.exists():
        print(f"[bibcheck] bib file not found: {bib}", file=sys.stderr)
        return 2

    cur_year = ns.current_year
    if cur_year is None and ns.strict_traceable:
        from datetime import date
        cur_year = date.today().year

    rep = validate(paper_dir, bib,
                   strict_traceable=ns.strict_traceable,
                   current_year=cur_year)
    report_path = ns.report or paper_dir / "BIBCHECK_REPORT.md"
    report_path.write_text(render(rep))

    blockers = (
        len(rep.undefined) + len(rep.duplicates) + len(rep.missing_fields)
        + len(rep.verify_markers)
        + len(rep.traceable_failures) + len(rep.anonymous_authors)
    )
    summary = {
        "cited": len(rep.cited),
        "defined": len(rep.defined),
        "undefined": len(rep.undefined),
        "uncited": len(rep.uncited),
        "duplicates": len(rep.duplicates),
        "missing_fields": len(rep.missing_fields),
        "verify_markers": len(rep.verify_markers),
        "key_year_mismatch": len(rep.key_year_mismatch),
        "traceable_failures": len(rep.traceable_failures),
        "anonymous_authors": len(rep.anonymous_authors),
        "recency_warning": rep.recency_warning,
        "blockers": blockers,
        "report": str(report_path),
    }
    if not ns.quiet:
        print(json.dumps(summary, indent=2))
    return 1 if blockers else 0


if __name__ == "__main__":
    sys.exit(main())
