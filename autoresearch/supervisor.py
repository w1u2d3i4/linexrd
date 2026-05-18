#!/usr/bin/env python3
"""
autoresearch supervisor — manages training experiments with periodic checkpointing.

Workflow:
1. Launches training as a background subprocess
2. Monitors metrics file at configurable intervals
3. Writes status reports for the orchestrating agent to read
4. Supports stop/resume/adjust commands

Usage (called by Claude Code agent, not directly by user):
    # Start an experiment
    python autoresearch/supervisor.py start \
        --command "python src/train.py --stage 3 --epochs 50" \
        --metrics_file results/train_metrics.json \
        --check_interval 1800 \
        --description "stage3 baseline"

    # Check status (non-blocking)
    python autoresearch/supervisor.py check

    # Stop current experiment
    python autoresearch/supervisor.py stop

    # Get decision report (formatted for agent consumption)
    python autoresearch/supervisor.py report
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

SUPERVISOR_STATE = "autoresearch.supervisor.json"
EXPERIMENT_LOG = "autoresearch.experiments.jsonl"
REPORT_FILE = "autoresearch.report.md"
PID_FILE = "autoresearch.pid"
OUTPUT_LOG = "autoresearch.output.log"


def get_workdir() -> Path:
    return Path(os.environ.get("AUTORESEARCH_WORKDIR", os.getcwd())).resolve()


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def load_supervisor_state(workdir: Path) -> dict:
    path = workdir / SUPERVISOR_STATE
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_supervisor_state(workdir: Path, state: dict):
    path = workdir / SUPERVISOR_STATE
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def append_experiment(workdir: Path, record: dict):
    path = workdir / EXPERIMENT_LOG
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_experiments(workdir: Path) -> list:
    path = workdir / EXPERIMENT_LOG
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------

def is_running(workdir: Path) -> tuple[bool, Optional[int]]:
    """Check if a training process is currently running."""
    pid_path = workdir / PID_FILE
    if not pid_path.exists():
        return False, None

    pid = int(pid_path.read_text().strip())
    try:
        os.kill(pid, 0)  # signal 0 = check existence
        return True, pid
    except (ProcessLookupError, PermissionError):
        pid_path.unlink(missing_ok=True)
        return False, None


def stop_process(workdir: Path) -> bool:
    """Stop the running training process."""
    running, pid = is_running(workdir)
    if not running or pid is None:
        return False

    try:
        # Send SIGTERM first, wait, then SIGKILL
        os.kill(pid, signal.SIGTERM)
        for _ in range(30):  # wait up to 30s
            time.sleep(1)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
        else:
            os.kill(pid, signal.SIGKILL)

        (workdir / PID_FILE).unlink(missing_ok=True)
        return True
    except (ProcessLookupError, PermissionError):
        (workdir / PID_FILE).unlink(missing_ok=True)
        return True


# ---------------------------------------------------------------------------
# Metrics reading
# ---------------------------------------------------------------------------

def read_latest_metrics(workdir: Path, metrics_file: str) -> Optional[dict]:
    """Read the latest metrics from the training process output."""
    mf = workdir / metrics_file
    if not mf.exists():
        return None
    try:
        with open(mf) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def read_output_tail(workdir: Path, n_lines: int = 50) -> str:
    """Read the last N lines of the training output log."""
    log_path = workdir / OUTPUT_LOG
    if not log_path.exists():
        return "(no output yet)"
    try:
        with open(log_path, "rb") as f:
            # Read from end
            f.seek(0, 2)
            size = f.tell()
            read_size = min(size, n_lines * 200)  # rough estimate
            f.seek(max(0, size - read_size))
            content = f.read().decode("utf-8", errors="replace")
            lines = content.split("\n")
            return "\n".join(lines[-n_lines:])
    except IOError:
        return "(error reading log)"


# ---------------------------------------------------------------------------
# Checkpoint analysis
# ---------------------------------------------------------------------------

def analyze_training_progress(state: dict, metrics: Optional[dict], output_tail: str) -> dict:
    """Analyze training progress and generate a decision-ready report."""
    analysis = {
        "timestamp": datetime.now().isoformat(),
        "experiment_id": state.get("experiment_id", "unknown"),
        "description": state.get("description", ""),
        "elapsed_minutes": 0,
        "status": "unknown",
        "metrics": metrics or {},
        "recommendations": [],
        "output_tail_lines": output_tail.split("\n")[-20:],
    }

    # Calculate elapsed time
    start_time = state.get("start_time")
    if start_time:
        started = datetime.fromisoformat(start_time)
        elapsed = datetime.now() - started
        analysis["elapsed_minutes"] = round(elapsed.total_seconds() / 60, 1)

    # Check for common issues in output
    output_lower = output_tail.lower()
    if "nan" in output_lower and "loss" in output_lower:
        analysis["status"] = "WARNING_NAN"
        analysis["recommendations"].append("STOP: Training loss is NaN. Reduce learning rate or check data.")
    elif "out of memory" in output_lower or "cuda oom" in output_lower:
        analysis["status"] = "WARNING_OOM"
        analysis["recommendations"].append("STOP: GPU OOM. Reduce batch size or model size.")
    elif "error" in output_lower and "traceback" in output_lower:
        analysis["status"] = "CRASHED"
        analysis["recommendations"].append("STOP: Training crashed. Check error in output log.")
    else:
        analysis["status"] = "RUNNING"

    # Analyze metrics trend
    if metrics:
        history = metrics.get("history", [])
        if len(history) >= 3:
            recent = history[-3:]
            metric_key = state.get("primary_metric", "loss")

            values = [h.get(metric_key) for h in recent if h.get(metric_key) is not None]
            if len(values) >= 3:
                # Check if metric is improving
                direction = state.get("direction", "lower")
                if direction == "lower":
                    improving = values[-1] < values[0]
                    stagnant = abs(values[-1] - values[0]) / (abs(values[0]) + 1e-8) < 0.01
                else:
                    improving = values[-1] > values[0]
                    stagnant = abs(values[-1] - values[0]) / (abs(values[0]) + 1e-8) < 0.01

                if stagnant:
                    analysis["recommendations"].append(
                        f"CONSIDER: Metric {metric_key} stagnant over last 3 checkpoints "
                        f"({values}). Consider adjusting learning rate or stopping early."
                    )
                elif not improving:
                    analysis["recommendations"].append(
                        f"CONSIDER: Metric {metric_key} degrading ({values}). "
                        f"May be overfitting or learning rate too high."
                    )
                else:
                    analysis["recommendations"].append(
                        f"OK: Metric {metric_key} improving ({values}). Continue training."
                    )

        # Report current best
        current = metrics.get("current", {})
        if current:
            analysis["current_metrics"] = current

    if not analysis["recommendations"]:
        analysis["recommendations"].append("INFO: No metrics available yet. Continue monitoring.")

    return analysis


# ---------------------------------------------------------------------------
# Report generation (for agent consumption)
# ---------------------------------------------------------------------------

def generate_report(workdir: Path) -> str:
    """Generate a markdown report for the orchestrating agent."""
    state = load_supervisor_state(workdir)
    if not state:
        return "# Autoresearch Status\n\nNo active experiment."

    running, pid = is_running(workdir)
    metrics = read_latest_metrics(workdir, state.get("metrics_file", "results/train_metrics.json"))
    output_tail = read_output_tail(workdir, 30)

    analysis = analyze_training_progress(state, metrics, output_tail)

    # Build report
    lines = [
        "# Autoresearch Checkpoint Report",
        "",
        f"**Experiment**: {state.get('description', 'unnamed')}",
        f"**Status**: {'RUNNING' if running else 'STOPPED'} (PID: {pid or 'N/A'})",
        f"**Elapsed**: {analysis['elapsed_minutes']:.0f} minutes",
        f"**Command**: `{state.get('command', 'N/A')}`",
        f"**Check interval**: {state.get('check_interval', 1800) // 60} minutes",
        "",
        "## Analysis",
        "",
    ]

    for rec in analysis["recommendations"]:
        lines.append(f"- {rec}")

    if analysis.get("current_metrics"):
        lines.extend(["", "## Current Metrics", ""])
        for k, v in analysis["current_metrics"].items():
            lines.append(f"- **{k}**: {v}")

    lines.extend([
        "",
        "## Recent Output (last 20 lines)",
        "",
        "```",
    ])
    lines.extend(analysis["output_tail_lines"])
    lines.extend([
        "```",
        "",
        "## Decision Required",
        "",
        "Options:",
        "1. **CONTINUE** — training is progressing well, check again at next interval",
        "2. **STOP** — stop training and adjust parameters",
        "3. **STOP_AND_EVAL** — stop training and run full evaluation",
        "4. **ADJUST** — stop, modify hyperparameters, restart",
        "",
        "## Experiment History",
        "",
    ])

    experiments = load_experiments(workdir)
    if experiments:
        lines.append(f"| # | Description | Duration | Status | Metric |")
        lines.append(f"|---|-------------|----------|--------|--------|")
        for i, exp in enumerate(experiments[-10:]):  # last 10
            dur = exp.get("duration_minutes", "?")
            desc = exp.get("description", "")[:30]
            status = exp.get("final_status", "?")
            metric = exp.get("final_metric", "?")
            lines.append(f"| {i+1} | {desc} | {dur}m | {status} | {metric} |")
    else:
        lines.append("(no previous experiments)")

    report = "\n".join(lines)

    # Write report file
    (workdir / REPORT_FILE).write_text(report)
    return report


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_start(args):
    """Start a new training experiment in the background."""
    workdir = get_workdir()

    # Check if already running
    running, pid = is_running(workdir)
    if running:
        print(f"[supervisor] Experiment already running (PID {pid}). Stop it first.")
        sys.exit(1)

    # Save state
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    state = {
        "experiment_id": experiment_id,
        "command": args.command,
        "description": args.description or "",
        "metrics_file": args.metrics_file or "results/train_metrics.json",
        "check_interval": args.check_interval or 1800,
        "primary_metric": args.primary_metric or "loss",
        "direction": args.direction or "lower",
        "start_time": datetime.now().isoformat(),
        "status": "running",
    }
    save_supervisor_state(workdir, state)

    # Launch training process
    log_path = workdir / OUTPUT_LOG
    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            args.command,
            shell=True,
            cwd=str(workdir),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,  # new process group so we can kill tree
        )

    # Save PID
    (workdir / PID_FILE).write_text(str(proc.pid))

    print(f"[supervisor] Started experiment '{args.description}' (PID {proc.pid})")
    print(f"[supervisor] Command: {args.command}")
    print(f"[supervisor] Check interval: {args.check_interval or 1800}s ({(args.check_interval or 1800) // 60}min)")
    print(f"[supervisor] Output log: {log_path}")
    print(f"[supervisor] Metrics file: {args.metrics_file or 'results/train_metrics.json'}")
    print(f"[supervisor] Use 'python autoresearch/supervisor.py check' to monitor")


def cmd_check(args):
    """Check current experiment status and print report."""
    workdir = get_workdir()
    report = generate_report(workdir)
    print(report)


def cmd_stop(args):
    """Stop the current experiment."""
    workdir = get_workdir()
    state = load_supervisor_state(workdir)

    running, pid = is_running(workdir)
    if not running:
        print("[supervisor] No experiment running.")
        return

    # Read final metrics before stopping
    metrics = read_latest_metrics(workdir, state.get("metrics_file", "results/train_metrics.json"))

    success = stop_process(workdir)
    if success:
        print(f"[supervisor] Stopped experiment (PID {pid})")

        # Calculate duration
        start_time = state.get("start_time")
        duration_minutes = 0
        if start_time:
            elapsed = datetime.now() - datetime.fromisoformat(start_time)
            duration_minutes = round(elapsed.total_seconds() / 60, 1)

        # Log experiment
        reason = args.reason or "manual_stop"
        record = {
            "experiment_id": state.get("experiment_id"),
            "description": state.get("description"),
            "command": state.get("command"),
            "start_time": state.get("start_time"),
            "stop_time": datetime.now().isoformat(),
            "duration_minutes": duration_minutes,
            "final_status": reason,
            "final_metric": metrics.get("current", {}).get(state.get("primary_metric")) if metrics else None,
            "final_metrics": metrics.get("current") if metrics else None,
        }
        append_experiment(workdir, record)

        state["status"] = "stopped"
        state["stop_time"] = datetime.now().isoformat()
        state["stop_reason"] = reason
        save_supervisor_state(workdir, state)

        print(f"[supervisor] Duration: {duration_minutes:.0f} minutes")
        print(f"[supervisor] Reason: {reason}")
    else:
        print("[supervisor] Failed to stop experiment.")


def cmd_report(args):
    """Generate and print the decision report."""
    workdir = get_workdir()
    report = generate_report(workdir)
    print(report)


def cmd_history(args):
    """Show experiment history."""
    workdir = get_workdir()
    experiments = load_experiments(workdir)

    if not experiments:
        print("[supervisor] No experiments recorded.")
        return

    print(f"\n{'='*70}")
    print(f"  Experiment History ({len(experiments)} total)")
    print(f"{'='*70}\n")

    print(f"  {'#':>3}  {'status':<15}  {'duration':>8}  {'metric':>10}  {'description'}")
    print(f"  {'---':>3}  {'------':<15}  {'--------':>8}  {'------':>10}  {'-----------'}")

    for i, exp in enumerate(experiments):
        dur = f"{exp.get('duration_minutes', 0):.0f}m"
        desc = exp.get("description", "")[:35]
        status = exp.get("final_status", "?")
        metric = exp.get("final_metric")
        metric_str = f"{metric:.4f}" if metric is not None else "N/A"
        print(f"  {i+1:>3}  {status:<15}  {dur:>8}  {metric_str:>10}  {desc}")

    print()


# ---------------------------------------------------------------------------
# Monitor loop (runs as daemon, called by agent periodically)
# ---------------------------------------------------------------------------

def cmd_monitor(args):
    """
    Run a single monitor check cycle.
    Designed to be called periodically by the orchestrating agent.
    Returns exit code:
      0 = running normally
      1 = needs attention (warning/stagnation)
      2 = critical (crash/NaN/OOM)
      3 = training completed
    """
    workdir = get_workdir()
    state = load_supervisor_state(workdir)

    if not state:
        print("NO_EXPERIMENT")
        sys.exit(3)

    running, pid = is_running(workdir)
    if not running:
        print("COMPLETED")
        # Generate final report
        generate_report(workdir)
        sys.exit(3)

    metrics = read_latest_metrics(workdir, state.get("metrics_file", "results/train_metrics.json"))
    output_tail = read_output_tail(workdir, 30)
    analysis = analyze_training_progress(state, metrics, output_tail)

    # Print compact status for agent
    print(f"STATUS={analysis['status']}")
    print(f"ELAPSED={analysis['elapsed_minutes']:.0f}min")
    if analysis.get("current_metrics"):
        for k, v in analysis["current_metrics"].items():
            print(f"METRIC {k}={v}")
    for rec in analysis["recommendations"]:
        print(f"REC: {rec}")

    # Exit code based on status
    if analysis["status"] in ("WARNING_NAN", "WARNING_OOM", "CRASHED"):
        sys.exit(2)
    elif any("stagnant" in r.lower() or "degrading" in r.lower() for r in analysis["recommendations"]):
        sys.exit(1)
    else:
        sys.exit(0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="autoresearch supervisor")
    sub = parser.add_subparsers(dest="cmd")

    # start
    p_start = sub.add_parser("start", help="Start a training experiment")
    p_start.add_argument("--command", "-c", required=True, help="Training command")
    p_start.add_argument("--description", "-D", default="", help="Experiment description")
    p_start.add_argument("--metrics_file", "-m", default="results/train_metrics.json",
                         help="Path to metrics JSON (written by training script)")
    p_start.add_argument("--check_interval", "-i", type=int, default=1800,
                         help="Check interval in seconds (default: 1800 = 30min)")
    p_start.add_argument("--primary_metric", default="loss", help="Primary metric name")
    p_start.add_argument("--direction", "-d", default="lower", choices=["higher", "lower"])

    # check
    sub.add_parser("check", help="Check current status")

    # stop
    p_stop = sub.add_parser("stop", help="Stop current experiment")
    p_stop.add_argument("--reason", "-r", default="manual_stop", help="Stop reason")

    # report
    sub.add_parser("report", help="Generate decision report")

    # monitor
    sub.add_parser("monitor", help="Single monitor check (for agent polling)")

    # history
    sub.add_parser("history", help="Show experiment history")

    args = parser.parse_args()

    if args.cmd == "start":
        cmd_start(args)
    elif args.cmd == "check":
        cmd_check(args)
    elif args.cmd == "stop":
        cmd_stop(args)
    elif args.cmd == "report":
        cmd_report(args)
    elif args.cmd == "monitor":
        cmd_monitor(args)
    elif args.cmd == "history":
        cmd_history(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
