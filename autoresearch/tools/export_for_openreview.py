#!/usr/bin/env python3
"""
export_for_openreview — assemble a hand-off bundle the user can upload to
OpenReview (or any double-blind venue) without this pipeline calling any
external API.

What the tool does, in order:

  1. Locate the latest `paper/main_v*.tex` (or the file passed via --tex).
  2. Compile it with tectonic (preferred) or latexmk into a PDF.
  3. Run anonymity checks: no real author names, emails, affiliations,
     acknowledgements, GitHub URLs, or \\thanks lines leaking identity.
  4. Extract title, abstract and keywords from the source.
  5. Write paper/openreview/round_{N}/
       submission.pdf
       submission.zip            - pdf + sources + figures + bib
       UPLOAD_README.md          - what to paste into which OpenReview field
       anonymity_report.md       - what the anonymity scan flagged
       fingerprint.json          - sha1 of the compiled PDF + tex; used by
                                   import_review.py to tie reviews back to
                                   the exact round they were written for

Pure stdlib. Compilation uses whatever is on $PATH.

Exit codes
  0  bundle written, no anonymity blockers
  1  anonymity blockers found (bundle still written so user can inspect)
  2  usage error / compile failure
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


TITLE_RE = re.compile(r"\\title\s*\{([^}]*)\}", re.DOTALL)
ABSTRACT_RE = re.compile(
    r"\\begin\{abstract\}(?P<body>.*?)\\end\{abstract\}", re.DOTALL
)
KEYWORDS_RE = re.compile(
    r"\\(?:keywords|Keywords|IEEEkeywords)\s*\{(?P<body>.*?)\}", re.DOTALL
)
THANKS_RE = re.compile(r"\\thanks\s*\{([^}]*)\}", re.DOTALL)
AUTHOR_RE = re.compile(r"\\author\s*\{([^}]*)\}", re.DOTALL)

EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
GITHUB_RE = re.compile(r"github\.com/[\w.-]+/[\w.-]+", re.IGNORECASE)
URL_RE = re.compile(r"https?://[^\s}]+")
ACK_RE = re.compile(
    r"\\section\*?\{[^}]*(Acknowledg|Ackno)[^}]*\}", re.IGNORECASE
)

ANON_HINTS = (
    "anonymous", "anon.", "under review", "double-blind",
    "submission to", "for review only",
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class AnonymityIssue:
    severity: str  # blocker | warning
    kind: str
    file: str
    line: int
    text: str


@dataclass
class Bundle:
    paper_dir: Path
    tex: Path
    round: int
    out_dir: Path
    title: str = ""
    abstract: str = ""
    keywords: str = ""
    issues: list[AnonymityIssue] = field(default_factory=list)
    pdf: Path | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_latest_tex(paper_dir: Path) -> tuple[Path, int]:
    candidates: list[tuple[int, Path]] = []
    for p in paper_dir.glob("main_v*.tex"):
        m = re.search(r"main_v(\d+)\.tex$", p.name)
        if m:
            candidates.append((int(m.group(1)), p))
    if not candidates:
        fallback = paper_dir / "main.tex"
        if fallback.exists():
            return fallback, 0
        raise FileNotFoundError(
            f"no main_v*.tex or main.tex in {paper_dir}"
        )
    candidates.sort()
    return candidates[-1][1], candidates[-1][0]


def compile_tex(tex: Path, paper_dir: Path) -> Path:
    pdf = tex.with_suffix(".pdf")
    if shutil.which("tectonic"):
        proc = subprocess.run(
            ["tectonic", "--keep-logs", "--synctex=none", "-Z", "search-path=.",
             str(tex.name)],
            cwd=str(paper_dir), capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"tectonic failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
            )
    elif shutil.which("latexmk"):
        proc = subprocess.run(
            ["latexmk", "-pdf", "-interaction=nonstopmode",
             "-halt-on-error", tex.name],
            cwd=str(paper_dir), capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"latexmk failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
            )
    else:
        raise RuntimeError(
            "neither `tectonic` nor `latexmk` found on PATH; install one and retry"
        )
    if not pdf.exists():
        raise RuntimeError(f"expected {pdf} but compilation did not produce it")
    return pdf


def _read_all_tex(paper_dir: Path) -> list[tuple[Path, str]]:
    out: list[tuple[Path, str]] = []
    for p in sorted(paper_dir.rglob("*.tex")):
        try:
            out.append((p, p.read_text(errors="replace")))
        except OSError:
            continue
    return out


def scan_anonymity(paper_dir: Path) -> list[AnonymityIssue]:
    issues: list[AnonymityIssue] = []
    for path, text in _read_all_tex(paper_dir):
        lines = text.splitlines()
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            # Skip commented-out lines for the quick scan
            content = re.sub(r"(?<!\\)%.*$", "", stripped)
            if not content:
                continue
            for m in EMAIL_RE.finditer(content):
                issues.append(AnonymityIssue(
                    "blocker", "email", str(path), i, m.group(0)))
            for m in GITHUB_RE.finditer(content):
                issues.append(AnonymityIssue(
                    "blocker", "github_link", str(path), i, m.group(0)))
            for m in URL_RE.finditer(content):
                if "anonymous" in m.group(0).lower():
                    continue
                issues.append(AnonymityIssue(
                    "warning", "url", str(path), i, m.group(0)))
            if ACK_RE.search(content):
                issues.append(AnonymityIssue(
                    "blocker", "acknowledgements_section",
                    str(path), i, stripped[:160]))
        # Author / thanks-level scan on the full file, since they span lines
        for m in AUTHOR_RE.finditer(text):
            body = m.group(1).strip()
            if not body:
                continue
            if not any(h in body.lower() for h in ANON_HINTS):
                line_no = text[: m.start()].count("\n") + 1
                issues.append(AnonymityIssue(
                    "blocker", "author_block", str(path), line_no,
                    body[:160]))
        for m in THANKS_RE.finditer(text):
            line_no = text[: m.start()].count("\n") + 1
            body = m.group(1).strip()
            if not body:
                continue
            issues.append(AnonymityIssue(
                "blocker", "thanks", str(path), line_no, body[:160]))
    return issues


def extract_frontmatter(tex_path: Path, paper_dir: Path) -> tuple[str, str, str]:
    # Search in main tex and its \input'd sections.
    blobs = [tex_path.read_text(errors="replace")]
    for _, t in _read_all_tex(paper_dir):
        blobs.append(t)
    joined = "\n".join(blobs)
    title = (TITLE_RE.search(joined) or _N).group(1).strip() if TITLE_RE.search(joined) else ""
    abs_m = ABSTRACT_RE.search(joined)
    abstract = abs_m.group("body").strip() if abs_m else ""
    kw_m = KEYWORDS_RE.search(joined)
    keywords = kw_m.group("body").strip() if kw_m else ""
    # Collapse whitespace for pasting into OpenReview
    abstract = re.sub(r"\s+", " ", abstract).strip()
    title = re.sub(r"\s+", " ", title).strip()
    keywords = re.sub(r"\s+", " ", keywords).strip()
    return title, abstract, keywords


class _Null:
    @staticmethod
    def group(_):
        return ""


_N = _Null()


def compute_fingerprint(pdf: Path, tex: Path) -> dict:
    h_pdf = hashlib.sha1(pdf.read_bytes()).hexdigest()
    h_tex = hashlib.sha1(tex.read_bytes()).hexdigest()
    return {
        "pdf_sha1": h_pdf,
        "tex_sha1": h_tex,
        "pdf_bytes": pdf.stat().st_size,
        "timestamp": datetime.now().isoformat(),
        "tex_path": str(tex),
    }


def build_zip(paper_dir: Path, pdf: Path, out: Path):
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(pdf, arcname="submission.pdf")
        for pattern in ("*.tex", "*.bib", "*.cls", "*.sty", "*.bst",
                         "math_commands.tex"):
            for p in paper_dir.rglob(pattern):
                try:
                    zf.write(p, arcname=str(p.relative_to(paper_dir)))
                except ValueError:
                    pass
        figs = paper_dir / "figures"
        if figs.is_dir():
            for p in figs.rglob("*"):
                if p.is_file() and p.suffix.lower() in {
                    ".pdf", ".png", ".jpg", ".jpeg", ".svg", ".tex", ".py"
                }:
                    zf.write(p, arcname=str(p.relative_to(paper_dir)))


def render_upload_readme(b: Bundle) -> str:
    lines = [
        f"# OpenReview upload checklist — round {b.round}",
        "",
        f"Prepared: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Source tex: `{b.tex}`",
        f"Compiled PDF: `{b.pdf}`",
        "",
        "## Copy-paste fields",
        "",
        "**Title**",
        "",
        f"```\n{b.title}\n```",
        "",
        "**Abstract**",
        "",
        f"```\n{b.abstract}\n```",
        "",
        "**Keywords**",
        "",
        f"```\n{b.keywords}\n```",
        "",
        "## Upload",
        "",
        "1. Go to the OpenReview submission form for your venue.",
        "2. Paste the title, abstract, keywords above into the respective fields.",
        "3. Upload `submission.pdf`. (If the venue also accepts a source zip, use `submission.zip`.)",
        "4. Confirm the anonymity block below has no blockers.",
        "5. Submit.",
        "",
        "## When reviews arrive",
        "",
        "1. Copy each reviewer's full text (scores + comments) into a single file, e.g. `paper/reviews/round{N}_openreview_raw.md`.",
        "2. Run:",
        "",
        "```bash",
        "python autoresearch/tools/import_review.py \\",
        f"    --raw paper/reviews/round{b.round}_openreview_raw.md \\",
        f"    --round {b.round} --paper-dir paper/",
        "```",
        "",
        "3. The importer will write `paper/reviews/review_round{N}.md`, "
        "categorize each weakness, and update `PAPER_AUTORESEARCH_STATE.json`. "
        "Then tell Claude Code `paper autoresearch` to resume.",
        "",
        "## Anonymity scan",
        "",
    ]
    if not b.issues:
        lines.append("No anonymity issues flagged.")
    else:
        blockers = [i for i in b.issues if i.severity == "blocker"]
        warnings = [i for i in b.issues if i.severity == "warning"]
        lines.append(f"- Blockers: **{len(blockers)}**")
        lines.append(f"- Warnings: **{len(warnings)}**")
        lines.append("")
        if blockers:
            lines.append("### Blockers — fix before uploading")
            lines.append("")
            for x in blockers:
                lines.append(
                    f"- [{x.kind}] {x.file}:L{x.line} — `{x.text}`"
                )
            lines.append("")
        if warnings:
            lines.append("### Warnings — inspect")
            lines.append("")
            for x in warnings:
                lines.append(
                    f"- [{x.kind}] {x.file}:L{x.line} — `{x.text}`"
                )
    return "\n".join(lines) + "\n"


def render_anonymity_report(issues: list[AnonymityIssue]) -> str:
    lines = ["# Anonymity report", ""]
    if not issues:
        lines.append("No issues.")
        return "\n".join(lines) + "\n"
    for sev in ("blocker", "warning"):
        hits = [i for i in issues if i.severity == sev]
        if not hits:
            continue
        lines.append(f"## {sev.title()}s ({len(hits)})")
        lines.append("")
        for i in hits:
            lines.append(f"- [{i.kind}] {i.file}:L{i.line} — `{i.text}`")
        lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Assemble OpenReview submission bundle")
    ap.add_argument("--paper-dir", default="paper", type=Path)
    ap.add_argument("--tex", default=None, type=Path,
                    help="Source tex (default: latest paper/main_v*.tex)")
    ap.add_argument("--round", type=int, default=None,
                    help="Round number (default: inferred from main_vN.tex)")
    ap.add_argument("--out", default=None, type=Path,
                    help="Output dir (default: paper/openreview/round_{N})")
    ap.add_argument("--skip-compile", action="store_true",
                    help="Assume paper/main_vN.pdf already exists")
    ns = ap.parse_args(argv)

    paper_dir: Path = ns.paper_dir
    if not paper_dir.is_dir():
        print(f"[export] paper dir not found: {paper_dir}", file=sys.stderr)
        return 2

    try:
        tex_path, inferred_round = (ns.tex, ns.round or 0) if ns.tex \
            else find_latest_tex(paper_dir)
    except FileNotFoundError as e:
        print(f"[export] {e}", file=sys.stderr)
        return 2
    round_no = ns.round if ns.round is not None else inferred_round

    out_dir = ns.out or (paper_dir / "openreview" / f"round_{round_no}")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        if ns.skip_compile:
            pdf = tex_path.with_suffix(".pdf")
            if not pdf.exists():
                raise RuntimeError(
                    f"--skip-compile set but {pdf} does not exist"
                )
        else:
            pdf = compile_tex(tex_path, paper_dir)
    except RuntimeError as e:
        print(f"[export] compile error: {e}", file=sys.stderr)
        return 2

    title, abstract, keywords = extract_frontmatter(tex_path, paper_dir)
    issues = scan_anonymity(paper_dir)

    b = Bundle(
        paper_dir=paper_dir, tex=tex_path, round=round_no, out_dir=out_dir,
        title=title, abstract=abstract, keywords=keywords,
        issues=issues, pdf=pdf,
    )

    # Copy PDF, build zip, write reports
    submission_pdf = out_dir / "submission.pdf"
    shutil.copy2(pdf, submission_pdf)
    build_zip(paper_dir, pdf, out_dir / "submission.zip")
    (out_dir / "UPLOAD_README.md").write_text(render_upload_readme(b))
    (out_dir / "anonymity_report.md").write_text(render_anonymity_report(issues))
    fp = compute_fingerprint(submission_pdf, tex_path)
    fp["round"] = round_no
    fp["title"] = title
    (out_dir / "fingerprint.json").write_text(json.dumps(fp, indent=2))

    blockers = [i for i in issues if i.severity == "blocker"]
    print(json.dumps({
        "round": round_no,
        "pdf": str(submission_pdf),
        "zip": str(out_dir / "submission.zip"),
        "readme": str(out_dir / "UPLOAD_README.md"),
        "fingerprint": fp["pdf_sha1"],
        "anonymity_blockers": len(blockers),
        "anonymity_warnings": len(issues) - len(blockers),
        "title": title,
        "keywords": keywords,
    }, indent=2))
    return 1 if blockers else 0


if __name__ == "__main__":
    sys.exit(main())
