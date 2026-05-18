#!/usr/bin/env python3
"""
autoresearch runner — Claude Code implementation of pi-autoresearch tools.

Implements the 3 core tools as CLI commands:
  - init_experiment: configure session
  - run_experiment: run benchmark command, capture output, parse METRIC lines
  - log_experiment: record result, auto-commit or auto-revert

Plus:
  - dashboard: show results table
  - status: show session summary

This is the infrastructure. The AI agent (Claude Code) is the loop:
  1. Read autoresearch.md to understand context
  2. Decide what to change in autoresearch.sh
  3. Call run_experiment to run the benchmark
  4. Call log_experiment to record keep/discard
  5. Update autoresearch.md with what was learned
  6. Repeat
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import statistics
from datetime import datetime
from pathlib import Path
from typing import Optional


JSONL_FILE = "autoresearch.jsonl"
STATE_FILE = "autoresearch.state.json"
CHECKS_SCRIPT = "autoresearch.checks.sh"

METRIC_RE = re.compile(r"^METRIC\s+([\w.]+)=(\S+)\s*$", re.MULTILINE)


def get_workdir() -> Path:
    return Path(os.environ.get("AUTORESEARCH_WORKDIR", os.getcwd())).resolve()


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def load_state(workdir: Path) -> dict:
    p = workdir / STATE_FILE
    return json.load(open(p)) if p.exists() else {}


def save_state(workdir: Path, state: dict):
    with open(workdir / STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def append_result(workdir: Path, result: dict):
    with open(workdir / JSONL_FILE, "a") as f:
        f.write(json.dumps(result) + "\n")


def load_results(workdir: Path) -> list:
    p = workdir / JSONL_FILE
    if not p.exists():
        return []
    return [json.loads(l) for l in open(p) if l.strip()]


def parse_metrics(output: str) -> dict:
    metrics = {}
    for m in METRIC_RE.finditer(output):
        try:
            v = float(m.group(2))
            if v == v:  # not NaN
                metrics[m.group(1)] = v
        except ValueError:
            pass
    return metrics


def compute_confidence(results: list, direction: str) -> Optional[float]:
    kept = [r for r in results if r.get("status") == "keep" and r.get("metric", 0) > 0]
    if len(kept) < 3:
        return None
    values = [r["metric"] for r in kept]
    baseline = values[0]
    median_val = statistics.median(values)
    mad = statistics.median([abs(v - median_val) for v in values])
    if mad == 0:
        return None
    best = max(values) if direction == "higher" else min(values)
    delta = abs(best - baseline)
    return round(delta / mad, 2)


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def git_run(workdir, *args):
    try:
        return subprocess.run(["git"] + list(args), cwd=workdir, capture_output=True, text=True, check=True).stdout.strip()
    except subprocess.CalledProcessError:
        return None


def git_commit(workdir, message):
    git_run(workdir, "add", "-A")
    git_run(workdir, "commit", "-m", message, "--allow-empty")
    return git_run(workdir, "rev-parse", "--short", "HEAD")


def git_revert_code(workdir):
    """Revert code changes but preserve autoresearch files."""
    ar_files = ["autoresearch.jsonl", "autoresearch.state.json", "autoresearch.md",
                "autoresearch.sh", "autoresearch.checks.sh", "autoresearch.ideas.md"]
    saved = {}
    for f in ar_files:
        fp = workdir / f
        if fp.exists():
            saved[f] = fp.read_bytes()

    git_run(workdir, "checkout", "HEAD", "--", ".")

    for f, content in saved.items():
        (workdir / f).write_bytes(content)


# ---------------------------------------------------------------------------
# init_experiment
# ---------------------------------------------------------------------------

def cmd_init(args):
    workdir = get_workdir()
    state = {
        "name": args.name,
        "metric_name": args.metric_name,
        "metric_unit": args.metric_unit or "",
        "direction": args.direction or "higher",
        "baseline": None,
        "best_metric": None,
        "best_commit": None,
        "segment": load_state(workdir).get("segment", -1) + 1,
        "created_at": datetime.now().isoformat(),
    }
    save_state(workdir, state)
    print(json.dumps({
        "status": "initialized",
        "name": args.name,
        "metric": args.metric_name,
        "direction": args.direction or "higher",
        "segment": state["segment"],
    }, indent=2))


# ---------------------------------------------------------------------------
# run_experiment
# ---------------------------------------------------------------------------

def cmd_run(args):
    workdir = get_workdir()
    state = load_state(workdir)
    if not state:
        print('{"error": "No session. Run init_experiment first."}')
        sys.exit(1)

    command = args.command
    timeout = args.timeout or 7200  # 2h default for ML training

    print(f"Running: {command}")
    start = time.time()

    try:
        proc = subprocess.run(
            command, shell=True, cwd=str(workdir),
            capture_output=True, text=True, timeout=timeout
        )
        duration = time.time() - start
        output = proc.stdout + proc.stderr
        crashed = proc.returncode != 0
        timed_out = False
    except subprocess.TimeoutExpired:
        duration = time.time() - start
        output = ""
        crashed = True
        timed_out = True

    # Parse metrics
    parsed = parse_metrics(output)
    primary = parsed.get(state["metric_name"])

    # Run checks
    checks_pass = None
    checks_output = ""
    checks_script = workdir / CHECKS_SCRIPT
    if checks_script.exists() and not crashed:
        try:
            cp = subprocess.run(
                ["bash", str(checks_script)], cwd=str(workdir),
                capture_output=True, text=True,
                timeout=args.checks_timeout or 300
            )
            checks_pass = cp.returncode == 0
            checks_output = (cp.stdout + cp.stderr)[-2000:]
        except subprocess.TimeoutExpired:
            checks_pass = False
            checks_output = "checks timed out"

    # Truncate output for agent consumption
    tail_lines = output.strip().split("\n")[-80:]
    tail_output = "\n".join(tail_lines)

    result = {
        "command": command,
        "exit_code": proc.returncode if not timed_out else -1,
        "duration_seconds": round(duration, 1),
        "passed": not crashed,
        "crashed": crashed,
        "timed_out": timed_out,
        "parsed_metrics": parsed,
        "parsed_primary": primary,
        "metric_name": state["metric_name"],
        "metric_unit": state.get("metric_unit", ""),
        "checks_pass": checks_pass,
        "checks_output": checks_output[-500:] if checks_output else "",
        "tail_output": tail_output,
    }

    # Save to temp for log_experiment to pick up
    with open(workdir / ".autoresearch_last_run.json", "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps({
        "passed": not crashed,
        "metric": primary,
        "metrics": parsed,
        "duration": f"{duration:.0f}s",
        "checks_pass": checks_pass,
    }, indent=2))


# ---------------------------------------------------------------------------
# log_experiment
# ---------------------------------------------------------------------------

def cmd_log(args):
    workdir = get_workdir()
    state = load_state(workdir)
    if not state:
        print('{"error": "No session."}')
        sys.exit(1)

    # Load last run details
    last_run_path = workdir / ".autoresearch_last_run.json"
    run_details = {}
    if last_run_path.exists():
        with open(last_run_path) as f:
            run_details = json.load(f)

    metric = args.metric
    status = args.status  # keep / discard / crash / checks_failed
    description = args.description

    # Parse ASI
    asi = {}
    if args.asi:
        try:
            asi = json.loads(args.asi)
        except json.JSONDecodeError:
            asi = {"raw": args.asi}

    # Parse secondary metrics
    metrics = {}
    if args.metrics:
        try:
            metrics = json.loads(args.metrics)
        except json.JSONDecodeError:
            pass

    # Git operations
    commit_hash = git_run(workdir, "rev-parse", "--short", "HEAD") or "unknown"

    if status == "keep":
        new_hash = git_commit(workdir, f"autoresearch: {description} [{state['metric_name']}={metric}]")
        if new_hash:
            commit_hash = new_hash
    elif status in ("discard", "crash", "checks_failed"):
        git_revert_code(workdir)

    # Update state
    if state.get("baseline") is None and status == "keep":
        state["baseline"] = metric
        state["best_metric"] = metric
        state["best_commit"] = commit_hash
    elif status == "keep":
        direction = state.get("direction", "higher")
        is_better = (metric > state.get("best_metric", float("-inf"))) if direction == "higher" \
            else (metric < state.get("best_metric", float("inf")))
        if is_better:
            state["best_metric"] = metric
            state["best_commit"] = commit_hash

    # Confidence
    results = load_results(workdir)
    confidence = compute_confidence(
        results + [{"metric": metric, "status": status}],
        state.get("direction", "higher")
    )
    state["confidence"] = confidence

    # Build record
    record = {
        "timestamp": datetime.now().isoformat(),
        "commit": commit_hash,
        "metric": metric,
        "metrics": metrics or run_details.get("parsed_metrics", {}),
        "status": status,
        "description": description,
        "segment": state.get("segment", 0),
        "confidence": confidence,
        "duration_seconds": run_details.get("duration_seconds"),
        "asi": asi,
    }
    append_result(workdir, record)
    save_state(workdir, state)

    # Print summary
    n = len(results) + 1
    icon = {"keep": "✓", "discard": "✗", "crash": "☠", "checks_failed": "⚠"}[status]
    improvement = ""
    if state.get("baseline") and state.get("best_metric"):
        bl = state["baseline"]
        best = state["best_metric"]
        if bl != 0:
            pct = ((best - bl) / abs(bl)) * 100
            improvement = f" ({'+' if pct > 0 else ''}{pct:.1f}%)"

    print(json.dumps({
        "run": n,
        "status": status,
        "metric": metric,
        "best": state.get("best_metric"),
        "improvement": improvement.strip(),
        "confidence": confidence,
        "description": description,
    }, indent=2))


# ---------------------------------------------------------------------------
# dashboard
# ---------------------------------------------------------------------------

def cmd_dashboard(args):
    workdir = get_workdir()
    state = load_state(workdir)
    results = load_results(workdir)

    if not state:
        print("No session.")
        return

    direction = state.get("direction", "higher")
    print(f"\n{'='*70}")
    print(f"  autoresearch: {state.get('name', '?')}")
    print(f"  {state['metric_name']} ({direction} is better)")
    print(f"  runs: {len(results)}  baseline: {state.get('baseline', 'N/A')}  best: {state.get('best_metric', 'N/A')}")
    if state.get("confidence"):
        print(f"  confidence: {state['confidence']}x")
    print(f"{'='*70}\n")

    if not results:
        return

    print(f"  {'#':>3}  {'st':<5}  {'metric':>10}  {'dur':>6}  description")
    print(f"  {'—'*3}  {'—'*5}  {'—'*10}  {'—'*6}  {'—'*30}")
    for i, r in enumerate(results):
        s = {"keep": " ✓", "discard": " ✗", "crash": " ☠", "checks_failed": " ⚠"}.get(r["status"], " ?")
        m = r.get("metric", 0)
        d = r.get("duration_seconds") or 0
        desc = r.get("description", "")[:40]
        print(f"  {i+1:>3}  {s:<5}  {m:>10.4f}  {d:>5.0f}s  {desc}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="autoresearch tools for Claude Code")
    sub = parser.add_subparsers(dest="subcmd")

    p = sub.add_parser("init_experiment")
    p.add_argument("--name", "-n", required=True)
    p.add_argument("--metric_name", "-m", required=True)
    p.add_argument("--metric_unit", "-u", default="")
    p.add_argument("--direction", "-d", default="higher", choices=["higher", "lower"])

    p = sub.add_parser("run_experiment")
    p.add_argument("--command", "-c", required=True)
    p.add_argument("--timeout", "-t", type=int, default=7200)
    p.add_argument("--checks_timeout", type=int, default=300)

    p = sub.add_parser("log_experiment")
    p.add_argument("--metric", type=float, required=True)
    p.add_argument("--status", required=True, choices=["keep", "discard", "crash", "checks_failed"])
    p.add_argument("--description", "-D", required=True)
    p.add_argument("--metrics", default=None, help="JSON dict of secondary metrics")
    p.add_argument("--asi", default=None, help="JSON dict of actionable side info")

    sub.add_parser("dashboard")
    sub.add_parser("status")

    args = parser.parse_args()

    if args.subcmd == "init_experiment":
        cmd_init(args)
    elif args.subcmd == "run_experiment":
        cmd_run(args)
    elif args.subcmd == "log_experiment":
        cmd_log(args)
    elif args.subcmd == "dashboard":
        cmd_dashboard(args)
    elif args.subcmd == "status":
        cmd_dashboard(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
