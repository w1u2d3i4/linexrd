#!/usr/bin/env python3
"""
optuna_driver — optional TPE/CMA outer loop around autoresearch.sh.

Why this exists: the base loop asks Claude to hand-pick each hyperparameter
trial, which wastes tokens and rarely beats a bog-standard Bayesian sampler.
This driver lets a pip-installable local library (Optuna) run the search
while Claude stays in the loop for interpretation, architecture changes,
and paper writing. No external APIs.

Search space format (`autoresearch.search_space.json`):

    {
      "LR":          {"type": "loguniform", "low": 1e-5, "high": 1e-2},
      "BATCH_SIZE":  {"type": "categorical", "choices": [64, 128, 256]},
      "DROPOUT":     {"type": "uniform", "low": 0.0, "high": 0.5},
      "EPOCHS":      {"type": "int", "low": 5, "high": 30, "step": 1}
    }

Keys must match variable names inside the HYPERPARAMS block of
`autoresearch.sh`:

    # ------- HYPERPARAMS (agent modifies these) -------
    LR=1e-3
    BATCH_SIZE=128
    DROPOUT=0.1
    EPOCHS=10
    # ------- END HYPERPARAMS -------

Usage
  # Propose trials and save results
  python optuna_driver.py run \\
      --space autoresearch.search_space.json \\
      --script autoresearch.sh \\
      --metric val_accuracy --direction higher \\
      --n-trials 25 --study-name lr_bs_sweep

  # Show best trials
  python optuna_driver.py best --study-name lr_bs_sweep --top 5

  # Re-run the best trial once for reporting in the paper
  python optuna_driver.py replay --study-name lr_bs_sweep --trial 7

Each trial writes a record into `autoresearch.jsonl` so the existing
`runner.py dashboard` shows Optuna trials next to Claude-driven ones.

Exit codes
  0  success
  2  usage error / Optuna not installed / space file malformed
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


HYPER_START = re.compile(r"^#\s*-+\s*HYPERPARAMS[^\n]*$", re.MULTILINE)
HYPER_END = re.compile(r"^#\s*-+\s*END\s+HYPERPARAMS[^\n]*$", re.MULTILINE)
ASSIGN_RE = re.compile(r"^(\w+)\s*=\s*(.+?)\s*$", re.MULTILINE)
METRIC_RE = re.compile(r"^METRIC\s+([\w.]+)=(\S+)\s*$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

def load_space(path: Path) -> dict:
    space = json.loads(path.read_text())
    if not isinstance(space, dict) or not space:
        raise ValueError("search space must be a non-empty object")
    for name, spec in space.items():
        if not isinstance(spec, dict) or "type" not in spec:
            raise ValueError(f"{name}: missing `type`")
        t = spec["type"]
        if t not in {"uniform", "loguniform", "int", "categorical"}:
            raise ValueError(f"{name}: unsupported type {t!r}")
        if t == "categorical":
            if "choices" not in spec or not spec["choices"]:
                raise ValueError(f"{name}: categorical needs `choices`")
        else:
            for k in ("low", "high"):
                if k not in spec:
                    raise ValueError(f"{name}: {t} needs `{k}`")
    return space


def suggest(trial, name: str, spec: dict):
    t = spec["type"]
    if t == "uniform":
        return trial.suggest_float(name, spec["low"], spec["high"])
    if t == "loguniform":
        return trial.suggest_float(name, spec["low"], spec["high"], log=True)
    if t == "int":
        return trial.suggest_int(name, spec["low"], spec["high"],
                                   step=int(spec.get("step", 1)))
    if t == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(f"unknown type {t!r}")


# ---------------------------------------------------------------------------
# Rewrite HYPERPARAMS block
# ---------------------------------------------------------------------------

def patch_script(script_path: Path, assignments: dict[str, object]) -> None:
    text = script_path.read_text()
    m1 = HYPER_START.search(text)
    m2 = HYPER_END.search(text)
    if not m1 or not m2 or m2.start() <= m1.end():
        raise RuntimeError(
            f"{script_path} is missing the '# --- HYPERPARAMS' / "
            "'# --- END HYPERPARAMS' markers; cannot safely patch."
        )
    block_start = m1.end()
    block_end = m2.start()
    block = text[block_start:block_end]
    # Replace existing assignments; keep unknown lines intact; append new ones.
    lines = block.splitlines(keepends=True)
    seen: set[str] = set()
    out_lines: list[str] = []
    for ln in lines:
        am = ASSIGN_RE.match(ln)
        if am and am.group(1) in assignments:
            key = am.group(1)
            seen.add(key)
            out_lines.append(f"{key}={_bash_value(assignments[key])}\n")
        else:
            out_lines.append(ln)
    for key, val in assignments.items():
        if key not in seen:
            out_lines.append(f"{key}={_bash_value(val)}\n")
    new_block = "".join(out_lines)
    new_text = text[:block_start] + new_block + text[block_end:]
    script_path.write_text(new_text)


def _bash_value(v: object) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return f"{v:g}" if isinstance(v, float) else str(v)
    s = str(v)
    if re.search(r"\s|\$|`|\"|'", s):
        return '"' + s.replace('"', '\\"') + '"'
    return s


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run_script(script: Path, workdir: Path, timeout: int) -> tuple[int, dict[str, float], str]:
    proc = subprocess.run(
        ["bash", str(script)], cwd=str(workdir),
        capture_output=True, text=True, timeout=timeout,
    )
    output = proc.stdout + proc.stderr
    metrics: dict[str, float] = {}
    for m in METRIC_RE.finditer(output):
        try:
            metrics[m.group(1)] = float(m.group(2))
        except ValueError:
            pass
    return proc.returncode, metrics, output[-4000:]


def log_to_jsonl(workdir: Path, record: dict):
    with (workdir / "autoresearch.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _require_optuna():
    try:
        import optuna  # noqa: F401
    except Exception:
        print(
            "[optuna_driver] `optuna` is not installed.\n"
            "  pip install optuna\n"
            "(no network beyond the pip install itself; Optuna runs locally)",
            file=sys.stderr,
        )
        sys.exit(2)


def cmd_run(args) -> int:
    _require_optuna()
    import optuna

    script = Path(args.script).resolve()
    if not script.exists():
        print(f"[optuna_driver] script not found: {script}", file=sys.stderr)
        return 2
    workdir = script.parent
    space = load_space(Path(args.space))

    storage = args.storage or f"sqlite:///{workdir / 'autoresearch.optuna.db'}"
    direction = "maximize" if args.direction == "higher" else "minimize"
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=args.study_name, storage=storage, sampler=sampler,
        direction=direction, load_if_exists=True,
    )
    backup = script.with_suffix(script.suffix + ".optuna_backup")
    shutil.copy2(script, backup)

    def objective(trial):
        choices = {name: suggest(trial, name, spec) for name, spec in space.items()}
        patch_script(script, choices)
        start = time.time()
        try:
            rc, metrics, tail = run_script(script, workdir, args.timeout)
        except subprocess.TimeoutExpired:
            trial.set_user_attr("status", "timeout")
            log_to_jsonl(workdir, {
                "timestamp": datetime.now().isoformat(),
                "status": "crash",
                "description": f"optuna trial {trial.number} timeout",
                "metric": None,
                "metrics": {}, "confidence": None,
                "duration_seconds": args.timeout,
                "asi": {"source": "optuna", "trial": trial.number,
                         "choices": choices},
            })
            raise optuna.TrialPruned()
        duration = time.time() - start
        primary = metrics.get(args.metric)
        status = "keep" if (rc == 0 and primary is not None) else "crash"
        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("exit_code", rc)
        trial.set_user_attr("tail", tail[-1500:])
        log_to_jsonl(workdir, {
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "description": f"optuna trial {trial.number}",
            "metric": primary,
            "metrics": metrics,
            "confidence": None,
            "duration_seconds": round(duration, 1),
            "asi": {"source": "optuna", "trial": trial.number,
                     "choices": choices, "exit_code": rc},
        })
        if primary is None:
            raise optuna.TrialPruned()
        return primary

    try:
        study.optimize(objective, n_trials=args.n_trials,
                       show_progress_bar=False,
                       catch=(RuntimeError,))
    finally:
        # Restore script to baseline so the repo is not left in a trial state.
        shutil.copy2(backup, script)
        backup.unlink(missing_ok=True)

    best = study.best_trial
    print(json.dumps({
        "study": args.study_name,
        "trials": len(study.trials),
        "best_value": best.value,
        "best_params": best.params,
        "storage": storage,
    }, indent=2))
    return 0


def cmd_best(args) -> int:
    _require_optuna()
    import optuna
    storage = args.storage or f"sqlite:///{Path(args.workdir) / 'autoresearch.optuna.db'}"
    study = optuna.load_study(study_name=args.study_name, storage=storage)
    trials = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value,
        reverse=study.direction == optuna.study.StudyDirection.MAXIMIZE,
    )[: args.top]
    print(json.dumps([
        {"number": t.number, "value": t.value, "params": t.params,
         "metrics": t.user_attrs.get("metrics", {})}
        for t in trials
    ], indent=2))
    return 0


def cmd_replay(args) -> int:
    _require_optuna()
    import optuna
    script = Path(args.script).resolve()
    workdir = script.parent
    storage = args.storage or f"sqlite:///{workdir / 'autoresearch.optuna.db'}"
    study = optuna.load_study(study_name=args.study_name, storage=storage)
    trial = next((t for t in study.trials if t.number == args.trial), None)
    if trial is None:
        print(f"[optuna_driver] no trial {args.trial} in study",
               file=sys.stderr)
        return 2
    backup = script.with_suffix(script.suffix + ".optuna_backup")
    shutil.copy2(script, backup)
    try:
        patch_script(script, trial.params)
        rc, metrics, tail = run_script(script, workdir, args.timeout)
        log_to_jsonl(workdir, {
            "timestamp": datetime.now().isoformat(),
            "status": "keep" if rc == 0 else "crash",
            "description": f"optuna replay trial {trial.number}",
            "metric": metrics.get(args.metric),
            "metrics": metrics, "confidence": None,
            "duration_seconds": None,
            "asi": {"source": "optuna_replay", "trial": trial.number,
                     "params": trial.params, "exit_code": rc},
        })
        print(json.dumps({
            "trial": trial.number, "exit_code": rc,
            "metrics": metrics, "params": trial.params,
        }, indent=2))
    finally:
        shutil.copy2(backup, script)
        backup.unlink(missing_ok=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Optuna outer loop for autoresearch")
    sub = ap.add_subparsers(dest="sub", required=True)

    p = sub.add_parser("run")
    p.add_argument("--space", required=True)
    p.add_argument("--script", default="autoresearch.sh")
    p.add_argument("--metric", required=True)
    p.add_argument("--direction", default="higher", choices=["higher", "lower"])
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--study-name", required=True)
    p.add_argument("--storage", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout", type=int, default=7200)

    p = sub.add_parser("best")
    p.add_argument("--study-name", required=True)
    p.add_argument("--storage", default=None)
    p.add_argument("--workdir", default=".")
    p.add_argument("--top", type=int, default=5)

    p = sub.add_parser("replay")
    p.add_argument("--study-name", required=True)
    p.add_argument("--trial", type=int, required=True)
    p.add_argument("--script", default="autoresearch.sh")
    p.add_argument("--metric", required=True)
    p.add_argument("--storage", default=None)
    p.add_argument("--timeout", type=int, default=7200)

    ns = ap.parse_args(argv)
    if ns.sub == "run":
        return cmd_run(ns)
    if ns.sub == "best":
        return cmd_best(ns)
    if ns.sub == "replay":
        return cmd_replay(ns)
    return 2


if __name__ == "__main__":
    sys.exit(main())
