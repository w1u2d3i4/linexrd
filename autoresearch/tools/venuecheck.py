#!/usr/bin/env python3
"""
venuecheck — venue-specific compliance gate for paper-autoresearch.

Runs *before* export_for_openreview.py. Blocks export if the paper violates
hard rules of the target CCF / international venue (page limit, anonymity
beyond what export_for_openreview catches, template integrity, color-blind
friendliness of figures, self-cite ratio).

Pure stdlib + optional Pillow (only for color-blind check on PNGs).

Usage
  python venuecheck.py \
      --paper-dir paper/ \
      --venue cvpr2026 \
      --report paper/VENUECHECK_REPORT.md

Or with an explicit config file:
  python venuecheck.py --paper-dir paper/ --config paper/venue.yaml

Exit codes
  0  no blockers
  1  one or more blockers (export must not proceed)
  2  usage error / missing paper directory
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# Built-in venue presets (CCF-A heavy hitters + a few CCF-B)
# ---------------------------------------------------------------------------

VENUE_PRESETS: dict[str, dict] = {
    # CCF-A
    "cvpr2026":      {"page_main": 8,  "ref_unlimited": True,  "double_blind": True,
                       "limitations_required": False, "checklist": False},
    "iccv2025":      {"page_main": 8,  "ref_unlimited": True,  "double_blind": True},
    "eccv2024":      {"page_main": 14, "ref_unlimited": True,  "double_blind": True},  # ECCV style is 14 incl ref
    "neurips2025":   {"page_main": 9,  "ref_unlimited": True,  "double_blind": True,
                       "limitations_required": True, "checklist": True},
    "iclr2026":      {"page_main": 9,  "ref_unlimited": True,  "double_blind": True,
                       "checklist": True},
    "icml2026":      {"page_main": 8,  "ref_unlimited": True,  "double_blind": True,
                       "checklist": True},
    "aaai2026":      {"page_main": 7,  "page_appendix": 2,     "double_blind": True},
    "ijcai2025":     {"page_main": 7,  "ref_unlimited": True,  "double_blind": True},
    "acl2026":       {"page_main": 8,  "ref_unlimited": True,  "double_blind": True,
                       "limitations_required": True},
    "emnlp2025":     {"page_main": 8,  "ref_unlimited": True,  "double_blind": True,
                       "limitations_required": True},
    "mm2025":        {"page_main": 9,  "ref_unlimited": True,  "double_blind": True},
    # CCF-B / Chinese venues
    "nlpcc2025":     {"page_main": 12, "ref_unlimited": True,  "double_blind": True},
    "cvm2026":       {"page_main": 12, "ref_unlimited": True,  "double_blind": False},
}

# Which `\documentclass` options correspond to "anonymous mode"
ANON_DOCLASS_HINT = {
    "cvpr": ["review"],
    "iccv": ["review"],
    "neurips": ["preprint"],   # absence of "final" implies anonymous
    "iclr": [],
}


# ---------------------------------------------------------------------------
# Issue type
# ---------------------------------------------------------------------------

@dataclass
class Issue:
    severity: str   # "blocker" | "warning"
    kind: str
    detail: str
    location: str = ""

    def to_md(self) -> str:
        return f"- **[{self.severity.upper()}]** `{self.kind}` — {self.detail}" \
               + (f"  (`{self.location}`)" if self.location else "")


# ---------------------------------------------------------------------------
# Locate latest tex / pdf
# ---------------------------------------------------------------------------

def latest_main(paper_dir: Path) -> tuple[Path | None, Path | None]:
    """Return (latest .tex, matching .pdf if present)."""
    candidates = sorted(paper_dir.glob("main_v*.tex"))
    if candidates:
        tex = candidates[-1]
        pdf = tex.with_suffix(".pdf")
        return tex, (pdf if pdf.exists() else None)
    tex = paper_dir / "main.tex"
    if tex.exists():
        pdf = paper_dir / "main.pdf"
        return tex, (pdf if pdf.exists() else None)
    return None, None


# ---------------------------------------------------------------------------
# Page count — try pdfinfo, fall back to /Type /Page heuristic on raw bytes
# ---------------------------------------------------------------------------

def page_count(pdf_path: Path) -> int | None:
    if not pdf_path.exists():
        return None
    try:
        out = subprocess.run(
            ["pdfinfo", str(pdf_path)],
            capture_output=True, text=True, timeout=20,
        )
        if out.returncode == 0:
            for line in out.stdout.splitlines():
                if line.lower().startswith("pages:"):
                    return int(line.split(":", 1)[1].strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    try:
        data = pdf_path.read_bytes()
        n = len(re.findall(rb"/Type\s*/Page(?![s/])", data))
        return n if n > 0 else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Reference page boundary — detect where the main body ends
# ---------------------------------------------------------------------------

REF_HEADERS = (
    "References", "REFERENCES", "Bibliography",
    "\\bibliographystyle", "\\bibliography{",
    "\\printbibliography", "\\begin{thebibliography}",
)


def detect_main_body_pages(pdf_path: Path) -> int | None:
    """Approximate main-body page count = pages up to (but not including) the
    page that first contains 'References'. Falls back to total pages."""
    if not pdf_path.exists():
        return None
    try:
        out = subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), "-"],
            capture_output=True, text=True, timeout=30,
        )
        if out.returncode != 0:
            return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    # \f is the form-feed page separator pdftotext emits.
    pages = out.stdout.split("\f")
    for idx, page in enumerate(pages):
        for h in ("References", "REFERENCES", "Bibliography", "BIBLIOGRAPHY"):
            # Match a line that *starts* with the header (allow leading ws)
            if re.search(rf"(?m)^\s*{h}\s*$", page):
                return idx  # 0-based: idx is # of pages before refs
    return len(pages)


# ---------------------------------------------------------------------------
# Template integrity
# ---------------------------------------------------------------------------

def sha1(path: Path) -> str:
    h = hashlib.sha1()
    h.update(path.read_bytes())
    return h.hexdigest()


def check_template(paper_dir: Path, expected: dict) -> list[Issue]:
    issues: list[Issue] = []
    for fname, want_sha in (expected or {}).items():
        p = paper_dir / fname
        if not p.exists():
            issues.append(Issue("warning", "template_missing",
                                f"template file {fname} not found", str(p)))
            continue
        got = sha1(p)
        if got != want_sha:
            issues.append(Issue("blocker", "template_modified",
                                f"{fname}: sha1 {got[:10]} != expected {want_sha[:10]}",
                                str(p)))
    return issues


# ---------------------------------------------------------------------------
# Anonymity extras — beyond what export_for_openreview.py catches
# ---------------------------------------------------------------------------

ACK_PATTERN = re.compile(
    r"\\section\*?\s*\{[^}]*(?:Acknowledg|致谢|Ackno)[^}]*\}",
    re.IGNORECASE,
)
INSTITUTE_PATTERN = re.compile(r"\\institute\s*\{[^}]+\}")
AFFIL_PATTERN = re.compile(r"\\affiliation\s*\{[^}]+\}")
ORCID_PATTERN = re.compile(r"\\orcid\s*\{[^}]+\}")
GRANT_PATTERN = re.compile(r"\b(?:grant|funded by|supported by)\b[^.]*\d{3,}",
                            re.IGNORECASE)


def check_anonymity_extras(tex: Path, double_blind: bool) -> list[Issue]:
    if not double_blind:
        return []
    issues: list[Issue] = []
    text = tex.read_text(errors="ignore")
    # Strip comments
    text_nc = re.sub(r"(?<!\\)%.*", "", text)

    if ACK_PATTERN.search(text_nc):
        issues.append(Issue("blocker", "ack_section",
                            "acknowledgements section found — must be removed for double-blind",
                            tex.name))
    for m in INSTITUTE_PATTERN.finditer(text_nc):
        issues.append(Issue("blocker", "institute_block",
                            f"\\institute{{...}} block: {m.group(0)[:80]}",
                            tex.name))
    for m in AFFIL_PATTERN.finditer(text_nc):
        issues.append(Issue("blocker", "affiliation_block",
                            f"\\affiliation{{...}} block: {m.group(0)[:80]}",
                            tex.name))
    for m in ORCID_PATTERN.finditer(text_nc):
        issues.append(Issue("blocker", "orcid",
                            f"ORCID exposes identity: {m.group(0)[:80]}",
                            tex.name))
    for m in GRANT_PATTERN.finditer(text_nc):
        issues.append(Issue("warning", "grant_number",
                            f"grant number may identify funding/lab: {m.group(0)[:120]}",
                            tex.name))
    return issues


# ---------------------------------------------------------------------------
# Self-cite ratio
# ---------------------------------------------------------------------------

BIB_AUTHOR_RE = re.compile(r"author\s*=\s*[\"{](?P<v>[^\"}]+)[\"}]",
                            re.IGNORECASE | re.DOTALL)
BIB_ENTRY_RE = re.compile(r"@\w+\s*\{\s*(?P<key>[\w:.\-]+),", re.MULTILINE)


def parse_bib_authors(bib_path: Path) -> dict[str, list[str]]:
    """Map bib key -> list of author last names."""
    if not bib_path.exists():
        return {}
    text = bib_path.read_text(errors="ignore")
    # Naive split on @ at start of line
    chunks = re.split(r"(?m)^\s*@", text)
    out: dict[str, list[str]] = {}
    for chunk in chunks:
        if not chunk.strip():
            continue
        km = BIB_ENTRY_RE.search("@" + chunk[:200])
        if not km:
            continue
        key = km.group("key")
        am = BIB_AUTHOR_RE.search(chunk)
        if not am:
            out[key] = []
            continue
        # Split on " and "; take last name (last token after comma or space)
        names: list[str] = []
        for raw in re.split(r"\s+and\s+", am.group("v")):
            raw = raw.strip().rstrip(",")
            if "," in raw:
                last = raw.split(",", 1)[0].strip()
            else:
                toks = raw.split()
                last = toks[-1] if toks else ""
            if last:
                names.append(last.lower())
        out[key] = names
    return out


def check_self_cite(paper_dir: Path, our_authors: list[str],
                     max_ratio: float) -> list[Issue]:
    if not our_authors:
        return []
    bib = paper_dir / "references.bib"
    authors_map = parse_bib_authors(bib)
    if not authors_map:
        return []
    # Count how many \cite keys appear in main_v*.tex
    tex, _ = latest_main(paper_dir)
    if tex is None:
        return []
    text = tex.read_text(errors="ignore")
    cited_keys: set[str] = set()
    for m in re.finditer(r"\\cite[a-z]*\{([^}]+)\}", text):
        for k in m.group(1).split(","):
            k = k.strip()
            if k:
                cited_keys.add(k)
    if not cited_keys:
        return []
    our_set = {a.lower() for a in our_authors}
    self_n = 0
    for k in cited_keys:
        authors = authors_map.get(k, [])
        if any(a in our_set for a in authors):
            self_n += 1
    ratio = self_n / len(cited_keys)
    if ratio > max_ratio:
        return [Issue(
            "warning", "self_cite_ratio",
            f"self-citations {self_n}/{len(cited_keys)} = {ratio:.0%} "
            f"exceeds max {max_ratio:.0%} (anonymous reviewers may infer identity)",
            "references.bib",
        )]
    return []


# ---------------------------------------------------------------------------
# Color-blind / grayscale check (optional, requires Pillow)
# ---------------------------------------------------------------------------

def check_figures_colorblind(paper_dir: Path) -> list[Issue]:
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        return [Issue("warning", "pillow_missing",
                      "Pillow not installed; skipping color-blind figure check", "")]
    figs_dir = paper_dir / "figures"
    if not figs_dir.exists():
        return []
    issues: list[Issue] = []
    for img_path in sorted(figs_dir.glob("*.png")):
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            issues.append(Issue("warning", "figure_open_fail",
                                f"could not open {img_path.name}: {e}", str(img_path)))
            continue
        # Heuristic: count pure-red and pure-green pixels in same image.
        # If both >0.5% AND blue-channel low, it's a red/green-only chart.
        small = img.resize((120, 120))
        px = list(small.getdata())
        red = sum(1 for r, g, b in px if r > 180 and g < 80 and b < 80)
        green = sum(1 for r, g, b in px if g > 180 and r < 80 and b < 80)
        total = len(px)
        if red / total > 0.005 and green / total > 0.005:
            issues.append(Issue("warning", "red_green_palette",
                                f"{img_path.name}: significant red+green only "
                                f"(R≈{red/total:.1%}, G≈{green/total:.1%}); "
                                "consider color-blind safe palette",
                                str(img_path)))
        # Grayscale legibility: convert to gray, compute std of gray in fig.
        gray = small.convert("L")
        gpx = list(gray.getdata())
        if max(gpx) - min(gpx) < 40:
            issues.append(Issue("warning", "low_contrast",
                                f"{img_path.name}: very low gray contrast "
                                f"(range {max(gpx)-min(gpx)}/255); will print poorly",
                                str(img_path)))
    return issues


# ---------------------------------------------------------------------------
# Limitations / checklist (NeurIPS/ICML/ICLR/ACL)
# ---------------------------------------------------------------------------

def check_limitations(tex: Path, required: bool) -> list[Issue]:
    if not required:
        return []
    text = tex.read_text(errors="ignore")
    if not re.search(r"\\section\*?\s*\{[^}]*(?:Limitations|不足|局限)[^}]*\}",
                      text, re.IGNORECASE):
        return [Issue("blocker", "limitations_missing",
                      "venue requires a Limitations section in the main paper",
                      tex.name)]
    return []


def check_checklist(paper_dir: Path, required: bool) -> list[Issue]:
    if not required:
        return []
    candidates = [
        paper_dir / "checklist.tex",
        paper_dir / "neurips_checklist.tex",
        paper_dir / "reproducibility_checklist.tex",
        paper_dir / "checklist.md",
    ]
    if any(c.exists() for c in candidates):
        return []
    return [Issue("warning", "checklist_missing",
                  "venue expects a reproducibility/paper checklist; "
                  "none found (run reproducibility_pack.py to generate one)",
                  str(paper_dir))]


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(args) -> dict:
    cfg: dict = {}
    if args.config:
        p = Path(args.config)
        if not p.exists():
            print(f"[venuecheck] config not found: {p}", file=sys.stderr)
            sys.exit(2)
        text = p.read_text()
        # Tiny YAML parser: only flat key: value (no nested mappings beyond
        # template_sha1: { fname: sha }). Avoids pyyaml dep.
        cfg = _mini_yaml(text)
    if args.venue:
        preset = VENUE_PRESETS.get(args.venue.lower())
        if not preset:
            print(f"[venuecheck] unknown venue '{args.venue}'. "
                  f"Known: {sorted(VENUE_PRESETS)}", file=sys.stderr)
            sys.exit(2)
        for k, v in preset.items():
            cfg.setdefault(k, v)
        cfg.setdefault("venue", args.venue)
    if args.our_authors:
        cfg["our_authors"] = [a.strip() for a in args.our_authors.split(",") if a.strip()]
    cfg.setdefault("max_self_cite_ratio", 0.30)
    return cfg


def _mini_yaml(text: str) -> dict:
    out: dict = {}
    cur_section: str | None = None
    cur_dict: dict | None = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.startswith("  ") and cur_dict is not None:
            k, _, v = line.strip().partition(":")
            cur_dict[k.strip()] = _coerce(v.strip())
            continue
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        k = k.strip()
        v = v.strip()
        if not v:
            cur_section = k
            cur_dict = {}
            out[k] = cur_dict
            continue
        cur_section, cur_dict = None, None
        out[k] = _coerce(v)
    return out


def _coerce(v: str):
    if v.lower() in ("true", "yes"):
        return True
    if v.lower() in ("false", "no"):
        return False
    try:
        if "." in v:
            return float(v)
        return int(v)
    except ValueError:
        return v.strip("\"'")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def render_report(paper_dir: Path, cfg: dict, issues: list[Issue],
                   facts: dict, report_path: Path) -> None:
    blockers = [i for i in issues if i.severity == "blocker"]
    warnings = [i for i in issues if i.severity == "warning"]
    lines = [
        "# venuecheck report",
        "",
        f"- venue: `{cfg.get('venue', 'custom')}`",
        f"- paper: `{paper_dir}`",
        f"- main-body pages (auto-detected): `{facts.get('body_pages')}`",
        f"- total pdf pages: `{facts.get('total_pages')}`",
        f"- main-body limit: `{cfg.get('page_main')}`",
        f"- ref unlimited: `{cfg.get('ref_unlimited', False)}`",
        f"- double-blind: `{cfg.get('double_blind', True)}`",
        "",
        f"**Blockers: {len(blockers)}, Warnings: {len(warnings)}**",
        "",
    ]
    if blockers:
        lines.append("## Blockers (must fix before export)")
        lines.extend(i.to_md() for i in blockers)
        lines.append("")
    if warnings:
        lines.append("## Warnings (review before export)")
        lines.extend(i.to_md() for i in warnings)
        lines.append("")
    if not issues:
        lines.append("All checks passed. Safe to run `export_for_openreview.py`.")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paper-dir", required=True)
    ap.add_argument("--venue", help=f"Preset venue (one of: {sorted(VENUE_PRESETS)})")
    ap.add_argument("--config", help="venue.yaml path")
    ap.add_argument("--report", default=None,
                     help="Output report path (default: paper-dir/VENUECHECK_REPORT.md)")
    ap.add_argument("--our-authors",
                     help="Comma-separated last names for self-cite check, e.g. 'Wang,Liu'")
    args = ap.parse_args(argv)

    paper_dir = Path(args.paper_dir).resolve()
    if not paper_dir.exists():
        print(f"[venuecheck] paper-dir does not exist: {paper_dir}", file=sys.stderr)
        return 2

    cfg = load_config(args)
    tex, pdf = latest_main(paper_dir)
    if tex is None:
        print(f"[venuecheck] no main_v*.tex / main.tex in {paper_dir}", file=sys.stderr)
        return 2

    issues: list[Issue] = []
    facts: dict = {"tex": str(tex)}

    # --- Page count
    if pdf is not None:
        total = page_count(pdf)
        body = detect_main_body_pages(pdf)
        facts["total_pages"] = total
        facts["body_pages"] = body
        page_main = cfg.get("page_main")
        if page_main and body is not None and body > page_main:
            issues.append(Issue(
                "blocker", "page_overflow",
                f"main body has {body} pages, venue allows {page_main}",
                pdf.name,
            ))
        if not cfg.get("ref_unlimited", True) and total and page_main \
           and total > page_main + cfg.get("page_appendix", 0):
            issues.append(Issue(
                "blocker", "total_overflow",
                f"total {total}p exceeds main+appendix budget", pdf.name,
            ))
    else:
        issues.append(Issue("warning", "pdf_missing",
                            "no compiled PDF found; page checks skipped",
                            str(paper_dir)))

    # --- Anonymity extras
    issues.extend(check_anonymity_extras(tex, cfg.get("double_blind", True)))

    # --- Template integrity
    issues.extend(check_template(paper_dir, cfg.get("template_sha1", {}) or {}))

    # --- Self-cite
    issues.extend(check_self_cite(paper_dir,
                                    cfg.get("our_authors", []),
                                    cfg.get("max_self_cite_ratio", 0.30)))

    # --- Figures
    issues.extend(check_figures_colorblind(paper_dir))

    # --- Required limitations / checklist
    issues.extend(check_limitations(tex, cfg.get("limitations_required", False)))
    issues.extend(check_checklist(paper_dir, cfg.get("checklist", False)))

    report_path = Path(args.report) if args.report else paper_dir / "VENUECHECK_REPORT.md"
    render_report(paper_dir, cfg, issues, facts, report_path)

    blockers = [i for i in issues if i.severity == "blocker"]
    print(json.dumps({
        "blockers": len(blockers),
        "warnings": len(issues) - len(blockers),
        "report": str(report_path),
    }, indent=2))
    return 1 if blockers else 0


if __name__ == "__main__":
    sys.exit(main())
