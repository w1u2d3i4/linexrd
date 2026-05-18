#!/usr/bin/env python3
"""
hero_figure_bundle — Mode-B hand-off for generating the paper's hero figure
via an external image model (GPT-image-1, DALL-E, Midjourney, ...) without
calling any API from this pipeline.

Pipeline the user is expected to follow:

  1. Run this tool. It reads PAPER_PLAN.md and assembles:

       paper/figures/fig1_prompt.md          - ready-to-paste prompt
       paper/figures/fig1_data_snapshot.json - numbers / comparisons the
                                               prompt references
       paper/figures/fig1_caption.tex        - caption + \\includegraphics stub
       paper/figures/fig1_placeholder.pdf    - grey placeholder so LaTeX still
                                               compiles before the real image
                                               arrives

  2. The user opens fig1_prompt.md, copies it into ChatGPT / GPT-image, and
     downloads the generated image as paper/figures/fig1_hero.png (or .pdf).

  3. They run `python hero_figure_bundle.py ingest` which removes the
     placeholder, validates the uploaded file, and refreshes the LaTeX stub
     to point at the real file.

No network calls. Pure stdlib + matplotlib (matplotlib is already a
pipeline dependency).

Usage
  python hero_figure_bundle.py prepare --plan PAPER_PLAN.md --paper-dir paper/
  python hero_figure_bundle.py ingest  --paper-dir paper/ --image fig1_hero.png
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Plan parsing
# ---------------------------------------------------------------------------

FIG1_ROW_RE = re.compile(
    r"^\|\s*Fig\s*1[^|]*\|.*$", re.IGNORECASE | re.MULTILINE
)
HERO_SECTION_RE = re.compile(
    r"(?:###\s*Hero\s*Figure[^\n]*\n(?P<body>.+?))(?=\n##\s|\Z)",
    re.IGNORECASE | re.DOTALL,
)
TITLE_RE = re.compile(r"^\*\*Title\*\*\s*:\s*(.+)$", re.MULTILINE)
CONTRIB_RE = re.compile(
    r"^\*\*One-sentence contribution\*\*\s*:\s*(.+)$", re.MULTILINE
)
VENUE_RE = re.compile(r"^\*\*Venue\*\*\s*:\s*(.+)$", re.MULTILINE)


def parse_plan(plan_path: Path) -> dict:
    text = plan_path.read_text()
    info = {
        "title": _search(TITLE_RE, text, "[title]"),
        "contribution": _search(CONTRIB_RE, text, "[one-sentence contribution]"),
        "venue": _search(VENUE_RE, text, "[venue]"),
        "fig1_row": "",
        "hero_block": "",
    }
    m = FIG1_ROW_RE.search(text)
    if m:
        info["fig1_row"] = m.group(0).strip()
    m = HERO_SECTION_RE.search(text)
    if m:
        info["hero_block"] = m.group("body").strip()
    return info


def _search(pat: re.Pattern[str], text: str, default: str) -> str:
    m = pat.search(text)
    return m.group(1).strip() if m else default


# ---------------------------------------------------------------------------
# Data snapshot
# ---------------------------------------------------------------------------

def _collect_snapshot(paper_dir: Path, results_path: Path | None) -> dict:
    snap: dict = {"claims": [], "metrics": {}, "baselines": []}
    claims = paper_dir / "CLAIMS_FROM_RESULTS.md"
    if claims.exists():
        snap["claims"] = [
            ln.strip("- ").strip()
            for ln in claims.read_text().splitlines()
            if ln.strip().startswith(("-", "*")) and len(ln.strip()) > 3
        ][:20]
    results = results_path or paper_dir.parent / "autoresearch.jsonl"
    if results.exists() and results.suffix == ".jsonl":
        best = None
        for ln in results.read_text().splitlines():
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if r.get("status") == "keep" and isinstance(r.get("metric"), (int, float)):
                if best is None or r["metric"] > best["metric"]:
                    best = r
        if best:
            snap["metrics"]["best_metric"] = best["metric"]
            snap["metrics"]["best_description"] = best.get("description", "")
    return snap


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """\
# Fig. 1 — GPT-image prompt for {title}

Paste everything below the `---` line into ChatGPT (GPT-image-1, 1536x1024,
quality=high). Attach `fig1_data_snapshot.json` if the chat interface lets
you upload a data file.

---

Create a single-panel, publication-quality overview figure for a {venue}
paper. The paper's one-sentence contribution is:

> {contribution}

The figure must:

1. Show the method's **core mechanism** visually on the left and the
   **key comparison vs. prior work** on the right. Both halves share the
   same horizontal baseline.
2. Use a minimalist vector-style illustration. White background, thin
   strokes, two-colour palette (a single accent colour for our method and a
   neutral grey for baselines). No 3D effects, no gradients, no glow.
3. Keep all in-figure text in **Times New Roman or similar serif**, 10pt
   minimum when printed at 3.3 inches wide.
4. Use colours that remain distinguishable in greyscale and that respect
   deuteranopia (avoid red-green contrast).
5. Add short textual callouts to at most 4 visual elements. Do not place a
   decorative title inside the figure — the paper caption will carry that.

Content anchors (from the paper plan):

{fig1_row}

{hero_block}

Quantitative anchors to reflect visually (fractional bars or sparklines,
never literal printed numbers unless the anchor below explicitly asks):

{snapshot_bullets}

Output: a single landscape image (3:2), transparent background preferred,
no watermark, no additional panels.
"""


def _bulletize(snap: dict) -> str:
    out: list[str] = []
    if snap.get("metrics"):
        for k, v in snap["metrics"].items():
            out.append(f"- {k}: {v}")
    for c in snap.get("claims", [])[:8]:
        out.append(f"- claim: {c}")
    if not out:
        out.append("- (none provided; the data_snapshot file is empty)")
    return "\n".join(out)


CAPTION_TEMPLATE = r"""% Auto-generated by hero_figure_bundle.py
\begin{figure}[t]
    \centering
    \includegraphics[width=0.98\linewidth]{figures/__HERO_PATH__}
    \caption{\textbf{Hero figure placeholder.} Replace with the image produced
        from \texttt{figures/fig1\_prompt.md}. The caption should state the
        comparison the figure makes and what the reader should notice, in
        one or two sentences, fully self-contained.}
    \label{fig:hero}
\end{figure}
"""


def _write_placeholder_pdf(out_path: Path):
    try:
        import matplotlib  # type: ignore
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        out_path.write_bytes(b"%PDF-1.4\n% placeholder (matplotlib missing)\n")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.55, "Hero figure placeholder",
            ha="center", va="center", fontsize=16)
    ax.text(0.5, 0.42, "(run hero_figure_bundle ingest once the\nreal image arrives)",
            ha="center", va="center", fontsize=9, color="gray")
    for sp in ax.spines.values():
        sp.set_color("lightgray")
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# prepare / ingest
# ---------------------------------------------------------------------------

def cmd_prepare(args) -> int:
    paper_dir: Path = args.paper_dir
    plan_path: Path = args.plan
    if not paper_dir.is_dir():
        print(f"[hero] paper dir not found: {paper_dir}", file=sys.stderr)
        return 2
    if not plan_path.exists():
        print(f"[hero] plan not found: {plan_path}", file=sys.stderr)
        return 2
    figs = paper_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    info = parse_plan(plan_path)
    snap = _collect_snapshot(paper_dir, args.results)

    prompt = PROMPT_TEMPLATE.format(
        title=info["title"],
        venue=info["venue"],
        contribution=info["contribution"],
        fig1_row=info["fig1_row"] or "(no Fig 1 row found in plan)",
        hero_block=info["hero_block"] or "(no hero figure section in plan)",
        snapshot_bullets=_bulletize(snap),
    )
    (figs / "fig1_prompt.md").write_text(prompt)
    (figs / "fig1_data_snapshot.json").write_text(
        json.dumps(snap, indent=2)
    )
    cap = CAPTION_TEMPLATE.replace("__HERO_PATH__", "fig1_placeholder.pdf")
    (figs / "fig1_caption.tex").write_text(cap)
    _write_placeholder_pdf(figs / "fig1_placeholder.pdf")

    print(json.dumps({
        "status": "prepared",
        "prompt": str(figs / "fig1_prompt.md"),
        "snapshot": str(figs / "fig1_data_snapshot.json"),
        "caption": str(figs / "fig1_caption.tex"),
        "placeholder": str(figs / "fig1_placeholder.pdf"),
        "next": "open fig1_prompt.md, paste into GPT-image, save result "
                 "as figures/fig1_hero.png, then run `ingest`.",
    }, indent=2))
    return 0


def cmd_ingest(args) -> int:
    paper_dir: Path = args.paper_dir
    figs = paper_dir / "figures"
    src: Path = args.image
    if not src.exists():
        print(f"[hero] image not found: {src}", file=sys.stderr)
        return 2
    if src.suffix.lower() not in {".png", ".pdf", ".jpg", ".jpeg"}:
        print(f"[hero] unsupported extension: {src.suffix}", file=sys.stderr)
        return 2
    target = figs / f"fig1_hero{src.suffix.lower()}"
    shutil.copy2(src, target)
    # Rewrite caption to point at the real file
    cap = CAPTION_TEMPLATE.replace("__HERO_PATH__", target.name)
    cap = cap.replace("Hero figure placeholder.",
                      "").replace(
        "Replace with the image produced from \\texttt{figures/fig1\\_prompt.md}. "
        "The caption should state the comparison the figure makes and what the "
        "reader should notice, in one or two sentences, fully self-contained.",
        "TODO: write a self-contained caption stating the comparison this figure "
        "makes and what the reader should notice.",
    )
    (figs / "fig1_caption.tex").write_text(cap)
    # Drop placeholder
    ph = figs / "fig1_placeholder.pdf"
    if ph.exists():
        ph.unlink()
    print(json.dumps({
        "status": "ingested",
        "target": str(target),
        "caption": str(figs / "fig1_caption.tex"),
        "next": "update the caption in figures/fig1_caption.tex, then "
                 "\\input{figures/fig1_caption.tex} in main.tex (or copy "
                 "into sections/1_introduction.tex).",
    }, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Hero figure Mode-B bundle")
    sub = ap.add_subparsers(dest="sub", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--plan", required=True, type=Path,
                   help="Path to PAPER_PLAN.md")
    p.add_argument("--paper-dir", default=Path("paper"), type=Path)
    p.add_argument("--results", default=None, type=Path,
                   help="Optional path to autoresearch.jsonl / results json")

    p = sub.add_parser("ingest")
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--paper-dir", default=Path("paper"), type=Path)

    ns = ap.parse_args(argv)
    if ns.sub == "prepare":
        return cmd_prepare(ns)
    if ns.sub == "ingest":
        return cmd_ingest(ns)
    return 2


if __name__ == "__main__":
    sys.exit(main())
