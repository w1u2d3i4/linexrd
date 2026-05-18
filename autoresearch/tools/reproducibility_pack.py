#!/usr/bin/env python3
"""
reproducibility_pack — assemble the supplementary / reproducibility bundle
that NeurIPS / ICML / ICLR / CVPR / AAAI / ACL increasingly require.

What it produces under --out (default: paper/repro/):

  hparams.yaml          extracted from autoresearch.sh HYPERPARAMS section,
                         augmented with autoresearch.search_space.json best
                         trial if present
  seeds.json            seeds list pulled from runs/baseline.jsonl
  env.txt               `pip freeze` snapshot (best-effort)
  cmds.sh               minimal 5-line "how to reproduce" command sequence
  README.md             repro instructions + directory layout + dep snapshot
  checklist.md          venue-specific reproducibility checklist (auto-fill)
  code/                 copy of train/eval/model entry points listed in
                         autoresearch.md "Files in Scope"
  results_summary.md    last 20 keep-experiments from autoresearch.jsonl

Pure stdlib. No required deps.

Usage
  python reproducibility_pack.py \
      --workdir . \
      --paper-dir paper/ \
      --venue neurips2025 \
      --out paper/repro/

Exit codes
  0  bundle written
  1  bundle written but with warnings (missing inputs)
  2  usage error
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime


# ---------------------------------------------------------------------------
# Venue-specific checklist templates
# ---------------------------------------------------------------------------

# Items keyed by venue, value = list of (Q, default_answer, evidence_pointer)
# default_answer is one of "Yes", "No", "N/A", or None (unfilled)
NEURIPS_CHECKLIST = [
    ("Claims in abstract & intro reflect contributions and scope",
     "Yes", "limitations + abstract"),
    ("Discussion of limitations of the work",
     "Yes", "Limitations section"),
    ("Theoretical assumptions stated and proofs sketched",
     "N/A", "no theory"),
    ("All information needed to reproduce main experimental results",
     "Yes", "repro/cmds.sh + repro/hparams.yaml + repro/code/"),
    ("Open access to code, with sufficient instructions",
     "Yes", "repro/code/ and repro/README.md"),
    ("All training/test details specified",
     "Yes", "repro/hparams.yaml"),
    ("Error bars / confidence intervals / variability reported",
     "Yes", "runs/statsig_round_*.json (multi-seed)"),
    ("Compute resources reported",
     "Yes", "repro/README.md (Compute Used)"),
    ("Conformance with NeurIPS Code of Ethics",
     "Yes", "no human subjects, no PII"),
    ("Broader impact discussed",
     None, "Broader Impact section"),
    ("Safeguards for high-risk releases",
     "N/A", "no high-risk artifacts"),
    ("Credits to creators of assets used",
     "Yes", "references.bib"),
    ("New assets documented",
     None, "n/a unless dataset/model released"),
    ("Crowdsourcing / human subjects",
     "N/A", "none used"),
    ("IRB approval if applicable",
     "N/A", "none used"),
    ("LLM usage declared",
     None, "describe LLM-assisted writing here"),
]

ICML_CHECKLIST = NEURIPS_CHECKLIST  # very similar
ICLR_CHECKLIST = NEURIPS_CHECKLIST

CVPR_CHECKLIST = [
    ("Compute resources reported (Compute Reporting Form)",
     "Yes", "repro/README.md"),
    ("Random seeds and hyperparameters specified",
     "Yes", "repro/hparams.yaml"),
    ("Code or pseudo-code provided",
     "Yes", "repro/code/"),
    ("Datasets / licenses identified",
     None, "fill in"),
    ("All quantitative claims include error bars or seeds count",
     "Yes", "repro/results_summary.md"),
]

ACL_CHECKLIST = [
    ("Limitations section in main paper",
     "Yes", "Limitations section"),
    ("Compute / model size reported",
     "Yes", "repro/README.md"),
    ("Hyperparameter search reported (best AND search range)",
     "Yes", "repro/hparams.yaml"),
    ("Number of runs and aggregation method (mean ± std) reported",
     "Yes", "repro/results_summary.md"),
    ("Use of existing scientific artifacts cited",
     "Yes", "references.bib"),
    ("Documentation of generated artifacts",
     None, "fill in if releasing model/data"),
    ("AI assistant usage declared (per ACL policy)",
     None, "fill in"),
]

CHECKLISTS: dict[str, list[tuple[str, str | None, str]]] = {
    "neurips": NEURIPS_CHECKLIST, "neurips2025": NEURIPS_CHECKLIST,
    "icml": ICML_CHECKLIST, "icml2026": ICML_CHECKLIST,
    "iclr": ICLR_CHECKLIST, "iclr2026": ICLR_CHECKLIST,
    "cvpr": CVPR_CHECKLIST, "cvpr2026": CVPR_CHECKLIST,
    "iccv": CVPR_CHECKLIST, "iccv2025": CVPR_CHECKLIST,
    "aaai": CVPR_CHECKLIST, "aaai2026": CVPR_CHECKLIST,
    "acl": ACL_CHECKLIST, "acl2026": ACL_CHECKLIST,
    "emnlp": ACL_CHECKLIST, "emnlp2025": ACL_CHECKLIST,
}


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

HP_RE = re.compile(
    r"#\s*-+\s*HYPERPARAMS.*?-+\s*\n(?P<body>.*?)\n#\s*-+\s*END HYPERPARAMS",
    re.DOTALL,
)
HP_LINE = re.compile(r"^\s*([A-Z_][A-Z0-9_]*)\s*=\s*(.+?)\s*(?:#.*)?$",
                      re.MULTILINE)


def extract_hparams(workdir: Path) -> tuple[dict, list[str]]:
    """Parse autoresearch.sh between HYPERPARAMS sentinels."""
    sh = workdir / "autoresearch.sh"
    warns: list[str] = []
    if not sh.exists():
        warns.append("autoresearch.sh not found")
        return {}, warns
    text = sh.read_text(errors="ignore")
    m = HP_RE.search(text)
    if not m:
        warns.append("HYPERPARAMS sentinels missing in autoresearch.sh")
        return {}, warns
    body = m.group("body")
    out: dict[str, object] = {}
    for line in body.splitlines():
        mm = HP_LINE.match(line)
        if not mm:
            continue
        k, v = mm.group(1), mm.group(2).strip().strip('"').strip("'")
        # coerce
        try:
            if "." in v or "e" in v.lower():
                out[k] = float(v)
            else:
                out[k] = int(v)
        except ValueError:
            out[k] = v
    return out, warns


def extract_search_space(workdir: Path) -> dict:
    p = workdir / "autoresearch.search_space.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def extract_seeds(workdir: Path) -> list[int]:
    p = workdir / "runs" / "baseline.jsonl"
    if not p.exists():
        return []
    seeds: list[int] = []
    for ln in p.read_text().splitlines():
        if not ln.strip():
            continue
        try:
            r = json.loads(ln)
            s = r.get("seed")
            if isinstance(s, int):
                seeds.append(s)
        except Exception:
            continue
    return sorted(set(seeds))


def extract_kept_results(workdir: Path, k: int = 20) -> list[dict]:
    p = workdir / "autoresearch.jsonl"
    if not p.exists():
        return []
    rows: list[dict] = []
    for ln in p.read_text().splitlines():
        if not ln.strip():
            continue
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if r.get("status") == "keep":
            rows.append(r)
    return rows[-k:]


def extract_files_in_scope(workdir: Path) -> list[Path]:
    md = workdir / "autoresearch.md"
    if not md.exists():
        return []
    text = md.read_text(errors="ignore")
    m = re.search(r"##\s*Files in Scope(?P<body>.*?)(?:\n##\s|\Z)",
                   text, re.DOTALL | re.IGNORECASE)
    if not m:
        return []
    out: list[Path] = []
    for ln in m.group("body").splitlines():
        bm = re.match(r"\s*-\s*`([^`]+)`", ln)
        if bm:
            p = workdir / bm.group(1)
            if p.exists() and p.is_file():
                out.append(p)
    return out


def pip_freeze() -> str | None:
    try:
        out = subprocess.run(["python", "-m", "pip", "freeze"],
                              capture_output=True, text=True, timeout=30)
        if out.returncode == 0:
            return out.stdout
    except Exception:
        pass
    return None


def gpu_info() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "-L"],
                              capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_yaml(d: dict, path: Path) -> None:
    """Tiny flat-yaml writer (no pyyaml dep). Strings get quoted."""
    lines: list[str] = []
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(f"{k}:")
            for kk, vv in v.items():
                lines.append(f"  {kk}: {_yamlfmt(vv)}")
        elif isinstance(v, list):
            lines.append(f"{k}:")
            for item in v:
                lines.append(f"  - {_yamlfmt(item)}")
        else:
            lines.append(f"{k}: {_yamlfmt(v)}")
    path.write_text("\n".join(lines) + "\n")


def _yamlfmt(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if v is None:
        return "null"
    if isinstance(v, dict):
        # Flow-style mapping (valid YAML)
        inner = ", ".join(f"{k}: {_yamlfmt(vv)}" for k, vv in v.items())
        return "{" + inner + "}"
    if isinstance(v, list):
        inner = ", ".join(_yamlfmt(x) for x in v)
        return "[" + inner + "]"
    s = str(v)
    if any(c in s for c in ': #"\n'):
        return '"' + s.replace('"', '\\"') + '"'
    return s


def write_checklist(venue: str, path: Path) -> bool:
    items = CHECKLISTS.get(venue.lower())
    if not items:
        return False
    lines = [
        f"# Reproducibility / Paper Checklist — {venue}",
        "",
        "Auto-filled by `reproducibility_pack.py`. Review every item before submission;",
        "answers marked _(default)_ should be confirmed manually.",
        "",
        "| # | Item | Answer | Evidence |",
        "|---|------|--------|----------|",
    ]
    for i, (q, ans, ev) in enumerate(items, 1):
        a = ans if ans is not None else "**TODO**"
        suffix = " _(default)_" if ans is not None else ""
        lines.append(f"| {i} | {q} | {a}{suffix} | {ev} |")
    path.write_text("\n".join(lines) + "\n")
    return True


def write_results_summary(rows: list[dict], path: Path) -> None:
    lines = [
        "# Top recent keep-experiments",
        "",
        "Pulled from `autoresearch.jsonl` (last 20 with status=keep).",
        "",
        "| timestamp | metric | value | what changed |",
        "|---|---|---|---|",
    ]
    for r in rows:
        ts = r.get("timestamp", "")[:19]
        m = r.get("metric_name", r.get("primary_metric", "?"))
        v = r.get("metric", r.get("value", "?"))
        asi = r.get("asi", {})
        what = asi.get("change") if isinstance(asi, dict) else (asi or "")
        what = (str(what) or "").replace("|", "/")[:80]
        lines.append(f"| {ts} | {m} | {v} | {what} |")
    path.write_text("\n".join(lines) + "\n")


def write_readme(out: Path, workdir: Path, venue: str, seeds: list[int],
                  kept_count: int, gpu: str) -> None:
    cmd = (workdir / "autoresearch.sh").relative_to(workdir) \
        if (workdir / "autoresearch.sh").exists() else "autoresearch.sh"
    lines = [
        f"# Reproducibility bundle — {venue}",
        "",
        f"Generated: {datetime.utcnow().isoformat(timespec='seconds')}Z",
        "",
        "## 5-line repro",
        "",
        "```bash",
        "# 1. install env (see env.txt for exact pin)",
        "pip install -r requirements.txt",
        "# 2. set seed",
        f"export SEED={seeds[0] if seeds else 0}",
        "# 3. run",
        f"bash {cmd}",
        "# 4. expect a `METRIC name=value` line on stdout",
        "# 5. multi-seed verification:",
        f"python autoresearch/tools/statsig.py baseline --command 'bash {cmd}' \\",
        f"       --seeds {' '.join(str(s) for s in (seeds or [0,1,2]))} "
        f"--metric <primary> --out runs/baseline.jsonl",
        "```",
        "",
        "## Layout",
        "",
        "```",
        "repro/",
        "├── README.md               this file",
        "├── checklist.md            venue-specific paper checklist",
        "├── hparams.yaml            extracted hyperparameters",
        "├── seeds.json              seeds used in baseline",
        "├── env.txt                 `pip freeze` snapshot",
        "├── cmds.sh                 minimal command sequence",
        "├── results_summary.md      last keep-experiments",
        "└── code/                   train / eval / model entry points",
        "```",
        "",
        "## Compute Used",
        "",
        f"- GPU: `{gpu}`",
        f"- Seeds: `{seeds or '[unknown]'}` (n={len(seeds)})",
        f"- Kept experiments aggregated: `{kept_count}`",
        "",
        "## Notes",
        "",
        "- Numbers in the paper trace to `autoresearch.jsonl` and `results/`",
        "  (verified by `numcheck.py`).",
        "- Multi-seed significance reported via `statsig.py compare`.",
        "- All randomness is seeded via the `SEED` env var inside `autoresearch.sh`.",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n")


def write_cmds_sh(out: Path, workdir: Path, seeds: list[int]) -> None:
    sh_path = (workdir / "autoresearch.sh")
    cmd = sh_path.name if sh_path.exists() else "autoresearch.sh"
    text = (
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "# Minimal repro: 5 commands.\n"
        "pip install -r requirements.txt\n"
        f"export SEED={seeds[0] if seeds else 0}\n"
        f"bash {cmd}\n"
        "python autoresearch/tools/statsig.py baseline \\\n"
        f"  --command 'bash {cmd}' --seeds "
        f"{' '.join(str(s) for s in (seeds or [0,1,2]))} \\\n"
        "  --metric val_accuracy --out runs/baseline.jsonl\n"
        "python autoresearch/tools/numcheck.py --paper-dir paper/ \\\n"
        "  --evidence autoresearch.jsonl results/ paper/CLAIMS_FROM_RESULTS.md\n"
    )
    p = out / "cmds.sh"
    p.write_text(text)
    p.chmod(0o755)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", default=".")
    ap.add_argument("--paper-dir", default="paper")
    ap.add_argument("--venue", default="neurips2025",
                     help=f"Venue key (one of: {sorted(CHECKLISTS)})")
    ap.add_argument("--out", default=None,
                     help="Output dir (default: paper-dir/repro/)")
    args = ap.parse_args(argv)

    workdir = Path(args.workdir).resolve()
    paper_dir = (workdir / args.paper_dir).resolve()
    out = Path(args.out).resolve() if args.out else (paper_dir / "repro")
    out.mkdir(parents=True, exist_ok=True)
    (out / "code").mkdir(exist_ok=True)

    warns: list[str] = []

    # 1. hparams
    hp, hpwarns = extract_hparams(workdir)
    warns.extend(hpwarns)
    space = extract_search_space(workdir)
    yaml_doc: dict = {"hyperparameters": hp}
    if space:
        yaml_doc["search_space"] = space
    yaml_doc["venue"] = args.venue
    yaml_doc["generated"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    write_yaml(yaml_doc, out / "hparams.yaml")

    # 2. seeds
    seeds = extract_seeds(workdir)
    (out / "seeds.json").write_text(json.dumps(seeds, indent=2))

    # 3. env
    env = pip_freeze()
    if env is None:
        warns.append("pip freeze failed; env.txt is a placeholder")
        env = "# pip freeze unavailable\n"
    (out / "env.txt").write_text(env)

    # 4. cmds.sh
    write_cmds_sh(out, workdir, seeds)

    # 5. results summary
    rows = extract_kept_results(workdir)
    write_results_summary(rows, out / "results_summary.md")

    # 6. checklist
    if not write_checklist(args.venue, out / "checklist.md"):
        warns.append(f"no checklist template for venue '{args.venue}'")
        # Generic empty checklist
        (out / "checklist.md").write_text(
            f"# Reproducibility checklist — {args.venue}\n\n"
            "No template available. Fill in venue-specific items manually.\n"
        )

    # 7. code copies
    files = extract_files_in_scope(workdir)
    if not files:
        warns.append("no Files-in-Scope listed in autoresearch.md; "
                     "code/ left empty")
    for f in files:
        try:
            target = out / "code" / f.relative_to(workdir)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
        except Exception as e:
            warns.append(f"copy failed for {f}: {e}")

    # 8. README + GPU info
    write_readme(out, workdir, args.venue, seeds, len(rows), gpu_info())

    print(json.dumps({
        "out": str(out),
        "venue": args.venue,
        "seeds": seeds,
        "hparams_keys": list(hp.keys()),
        "warnings": warns,
        "kept_experiments_summarized": len(rows),
        "code_files_copied": [str(f.relative_to(workdir)) for f in files],
    }, indent=2))
    return 1 if warns else 0


if __name__ == "__main__":
    sys.exit(main())
