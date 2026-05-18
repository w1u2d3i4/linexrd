#!/usr/bin/env python3
"""
statsig — multi-seed statistical significance gate for code-autoresearch.

Runs `autoresearch.sh` (or any benchmark command) N times with different seeds,
collects METRIC lines, compares the "candidate" metric distribution to a
"baseline" distribution via:

  * Welch's t-test (two-sided), or paired t-test if --paired
  * Mann-Whitney U (two-sided, tie-corrected),
       or Wilcoxon signed-rank if --paired
  * Bootstrap CI on the mean delta (B=10000),
       paired-by-seed if --paired

The tool blocks a `keep` decision unless:

  * mean improves in the direction of `--direction` AND
  * min(p_t, p_u) <= alpha_eff (alpha / bonferroni K) AND
  * bootstrap CI of the delta does not cross zero

Pure stdlib. Uses scipy only if available for tighter p-values, otherwise
falls back to a pure-python implementation.

Usage
  # 1. Seed the baseline once
  python statsig.py baseline --command "bash autoresearch.sh" \
         --seeds 0 1 2 3 4 --metric val_accuracy --direction higher \
         --out runs/baseline.jsonl

  # 2. After changing HYPERPARAMS, compare (independent two-sample)
  python statsig.py compare --command "bash autoresearch.sh" \
         --seeds 0 1 2 3 4 --metric val_accuracy --direction higher \
         --baseline runs/baseline.jsonl \
         --report runs/statsig_report.json

  # 2b. Same-seed paired comparison (lower variance, requires baseline.jsonl
  #     to have the same seeds as --seeds)
  python statsig.py compare ... --paired

  # 2c. Bonferroni when comparing against K baselines simultaneously
  python statsig.py compare ... --bonferroni 3

Exit codes
  0  candidate is significantly better than baseline -> keep
  1  not significantly better -> discard
  2  usage error or the command crashed on all seeds
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable


METRIC_RE = re.compile(r"^METRIC\s+([\w.]+)=(\S+)\s*$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_seed(command: str, seed: int, workdir: Path, timeout: int,
             env_seed_var: str) -> tuple[int, float | None, str]:
    env = os.environ.copy()
    env[env_seed_var] = str(seed)
    start = time.time()
    try:
        proc = subprocess.run(
            command, shell=True, cwd=str(workdir),
            capture_output=True, text=True, timeout=timeout, env=env,
        )
    except subprocess.TimeoutExpired:
        return -1, None, f"timeout after {timeout}s"
    output = proc.stdout + proc.stderr
    metrics: dict[str, float] = {}
    for m in METRIC_RE.finditer(output):
        try:
            metrics[m.group(1)] = float(m.group(2))
        except ValueError:
            pass
    duration = time.time() - start
    return proc.returncode, metrics, f"{duration:.0f}s"


def run_all(command: str, seeds: list[int], metric: str, workdir: Path,
            timeout: int, env_seed_var: str) -> list[dict]:
    records: list[dict] = []
    for seed in seeds:
        print(f"[statsig] seed={seed} ...", flush=True)
        rc, metrics, dur = run_seed(command, seed, workdir, timeout, env_seed_var)
        primary = metrics.get(metric) if isinstance(metrics, dict) else None
        records.append({
            "seed": seed,
            "exit_code": rc,
            "duration": dur,
            "metric": primary,
            "all_metrics": metrics if isinstance(metrics, dict) else {},
        })
        print(f"[statsig]   metric={primary} exit={rc} dur={dur}", flush=True)
    return records


def _values(records: list[dict]) -> list[float]:
    return [r["metric"] for r in records
            if isinstance(r.get("metric"), (int, float))]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def welch_t_test(a: list[float], b: list[float]) -> tuple[float, float]:
    """Returns (t, two-sided p). Pure python; normal approximation for df."""
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    na, nb = len(a), len(b)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return float("inf"), 0.0 if ma != mb else 1.0
    t = (ma - mb) / se
    # Welch-Satterthwaite df
    try:
        df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    except ZeroDivisionError:
        df = na + nb - 2
    p = 2 * _student_t_sf(abs(t), df)
    return t, p


def _student_t_sf(t: float, df: float) -> float:
    # Survival function via regularized incomplete beta.
    try:
        from math import lgamma
        x = df / (df + t * t)
        return 0.5 * _regularized_incomplete_beta(x, df / 2, 0.5)
    except Exception:
        # crude fallback: normal approximation
        return 0.5 * math.erfc(t / math.sqrt(2))


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    # Continued fraction via Lentz, adequate for df>=2.
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    f = 1.0
    c = 1.0
    d = 0.0
    eps = 1e-12
    for m in range(1, 200):
        # even step
        numer = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1 + numer * d
        if abs(d) < eps:
            d = eps
        c = 1 + numer / c
        if abs(c) < eps:
            c = eps
        d = 1 / d
        f *= d * c
        # odd step
        numer = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        d = 1 + numer * d
        if abs(d) < eps:
            d = eps
        c = 1 + numer / c
        if abs(c) < eps:
            c = eps
        d = 1 / d
        delta = d * c
        f *= delta
        if abs(delta - 1) < 1e-10:
            break
    return front * (f - 1)


def mann_whitney_u(a: list[float], b: list[float]) -> tuple[float, float]:
    """Two-sided Mann-Whitney U with normal approximation."""
    if not a or not b:
        return float("nan"), float("nan")
    try:
        from scipy.stats import mannwhitneyu  # type: ignore
        r = mannwhitneyu(a, b, alternative="two-sided")
        return float(r.statistic), float(r.pvalue)
    except Exception:
        pass
    combined = sorted([(v, "a") for v in a] + [(v, "b") for v in b])
    ranks: list[float] = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    ra = sum(r for r, (_, g) in zip(ranks, combined) if g == "a")
    na, nb = len(a), len(b)
    u1 = ra - na * (na + 1) / 2
    u2 = na * nb - u1
    u = min(u1, u2)
    mu = na * nb / 2
    sigma = math.sqrt(na * nb * (na + nb + 1) / 12)
    if sigma == 0:
        return u, 1.0
    z = (u - mu) / sigma
    p = 2 * (1 - _norm_cdf(abs(z)))
    return u, p


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bootstrap_mean_delta(a: list[float], b: list[float], B: int = 10000,
                         seed: int = 0, ci: float = 0.95) -> tuple[float, float, float]:
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(B):
        sa = [a[rng.randrange(len(a))] for _ in range(len(a))]
        sb = [b[rng.randrange(len(b))] for _ in range(len(b))]
        deltas.append(statistics.mean(sa) - statistics.mean(sb))
    deltas.sort()
    lo_idx = int((1 - ci) / 2 * B)
    hi_idx = int((1 - (1 - ci) / 2) * B) - 1
    return statistics.mean(deltas), deltas[lo_idx], deltas[hi_idx]


def bootstrap_paired_delta(diffs: list[float], B: int = 10000,
                            seed: int = 0, ci: float = 0.95
                            ) -> tuple[float, float, float]:
    """Bootstrap CI on the mean of paired differences (per-seed deltas)."""
    rng = random.Random(seed)
    n = len(diffs)
    means: list[float] = []
    for _ in range(B):
        s = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.mean(s))
    means.sort()
    lo_idx = int((1 - ci) / 2 * B)
    hi_idx = int((1 - (1 - ci) / 2) * B) - 1
    return statistics.mean(means), means[lo_idx], means[hi_idx]


def paired_t_test(diffs: list[float]) -> tuple[float, float]:
    """One-sample t-test on diffs vs 0. Two-sided p."""
    n = len(diffs)
    if n < 2:
        return float("nan"), float("nan")
    m = statistics.mean(diffs)
    sd = statistics.stdev(diffs)
    if sd == 0:
        return float("inf"), 0.0 if m != 0 else 1.0
    t = m / (sd / math.sqrt(n))
    p = 2 * _student_t_sf(abs(t), n - 1)
    return t, p


def wilcoxon_signed_rank(diffs: list[float]) -> tuple[float, float]:
    """Two-sided Wilcoxon signed-rank test on paired diffs."""
    nz = [d for d in diffs if d != 0]
    n = len(nz)
    if n < 1:
        return float("nan"), float("nan")
    try:
        from scipy.stats import wilcoxon  # type: ignore
        r = wilcoxon(nz, alternative="two-sided", zero_method="wilcox")
        return float(r.statistic), float(r.pvalue)
    except Exception:
        pass
    abs_d = sorted([(abs(d), d) for d in nz])
    # Average ranks for ties
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs_d[j + 1][0] == abs_d[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    w_pos = sum(r for r, (_, d) in zip(ranks, abs_d) if d > 0)
    w_neg = sum(r for r, (_, d) in zip(ranks, abs_d) if d < 0)
    w = min(w_pos, w_neg)
    mu = n * (n + 1) / 4
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    if sigma == 0:
        return w, 1.0
    z = (w - mu) / sigma
    p = 2 * (1 - _norm_cdf(abs(z)))
    return w, p


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

def _fmt_pm(mean: float, std: float, sig: int = 3) -> str:
    """LaTeX-ready 'mean ± std' string with significant digits."""
    if not math.isfinite(mean) or not math.isfinite(std):
        return f"${mean} \\pm {std}$"
    # Choose decimals from std magnitude
    if std == 0:
        decimals = sig
    else:
        decimals = max(0, sig - 1 - int(math.floor(math.log10(abs(std)))))
        decimals = min(decimals, 6)
    return f"${mean:.{decimals}f} \\pm {std:.{decimals}f}$"


def decide(candidate: list[float], baseline: list[float],
           direction: str, alpha: float,
           paired: bool = False, bonferroni: int = 1,
           cand_seeds: list[int] | None = None,
           base_seeds: list[int] | None = None) -> dict:
    if len(candidate) < 2 or len(baseline) < 2:
        return {
            "decision": "insufficient_seeds",
            "candidate_n": len(candidate),
            "baseline_n": len(baseline),
        }
    mean_c, mean_b = statistics.mean(candidate), statistics.mean(baseline)
    std_c = statistics.stdev(candidate)
    std_b = statistics.stdev(baseline)
    delta = mean_c - mean_b
    better = delta > 0 if direction == "higher" else delta < 0

    alpha_eff = alpha / max(1, bonferroni)

    paired_used = False
    diffs: list[float] = []
    if paired:
        # Match by seed index — require same seed list, and same length
        if (cand_seeds is None or base_seeds is None
                or list(cand_seeds) != list(base_seeds)
                or len(candidate) != len(baseline)):
            paired_used = False
        else:
            diffs = [c - b for c, b in zip(candidate, baseline)]
            paired_used = True

    if paired_used:
        t, p_t = paired_t_test(diffs)
        u, p_u = wilcoxon_signed_rank(diffs)
        mean_d, ci_lo, ci_hi = bootstrap_paired_delta(diffs)
        test_name_t = "paired_t"
        test_name_u = "wilcoxon_signed_rank"
    else:
        t, p_t = welch_t_test(candidate, baseline)
        u, p_u = mann_whitney_u(candidate, baseline)
        mean_d, ci_lo, ci_hi = bootstrap_mean_delta(candidate, baseline)
        test_name_t = "welch_t"
        test_name_u = "mann_whitney"

    ci_excludes_zero = (ci_lo > 0 and ci_hi > 0) or (ci_lo < 0 and ci_hi < 0)
    p_min = min(p_t if p_t == p_t else 1.0,
                p_u if p_u == p_u else 1.0)
    significant = better and p_min <= alpha_eff and ci_excludes_zero

    # Build LaTeX-ready strings
    sign = "+" if delta >= 0 else ""
    p_str = f"{p_min:.3g}"
    delta_decimals = max(0, 3 - 1 - int(math.floor(math.log10(abs(delta)))))\
        if delta != 0 and math.isfinite(delta) else 3
    delta_decimals = min(max(delta_decimals, 1), 6)

    return {
        "decision": "keep" if significant else "discard",
        "direction": direction,
        "alpha": alpha,
        "alpha_eff": alpha_eff,
        "bonferroni_K": bonferroni,
        "paired": paired_used,
        "candidate": {"n": len(candidate), "mean": mean_c,
                       "std": std_c,
                       "median": statistics.median(candidate),
                       "values": candidate},
        "baseline": {"n": len(baseline), "mean": mean_b,
                      "std": std_b,
                      "median": statistics.median(baseline),
                      "values": baseline},
        "delta_mean": delta,
        test_name_t: {"statistic": t, "p_value": p_t},
        test_name_u: {"statistic": u, "p_value": p_u},
        "bootstrap_mean_delta": {
            "mean": mean_d, "ci95_low": ci_lo, "ci95_high": ci_hi,
            "excludes_zero": ci_excludes_zero,
        },
        "latex": {
            "candidate_pm": _fmt_pm(mean_c, std_c),
            "baseline_pm":  _fmt_pm(mean_b, std_b),
            "delta_pm":     f"${sign}{delta:.{delta_decimals}f}$",
            "p_str":        f"$p={p_str}$",
            "table_cell":   f"{_fmt_pm(mean_c, std_c)} ({sign}{delta:.{delta_decimals}f})",
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, records: Iterable[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def cmd_baseline(args):
    workdir = Path(args.workdir).resolve()
    records = run_all(args.command, args.seeds, args.metric, workdir,
                       args.timeout, args.seed_env_var)
    out = Path(args.out)
    _write_jsonl(out, records)
    vals = _values(records)
    if not vals:
        print("[statsig] all seeds failed to emit a metric", file=sys.stderr)
        return 2
    summary = {
        "n": len(vals), "mean": statistics.mean(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "values": vals, "path": str(out),
    }
    print(json.dumps(summary, indent=2))
    return 0


def cmd_compare(args):
    workdir = Path(args.workdir).resolve()
    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        print(f"[statsig] baseline file not found: {baseline_path}", file=sys.stderr)
        return 2
    baseline_records = _read_jsonl(baseline_path)
    # Sort by seed to make paired pairing deterministic
    baseline_records.sort(key=lambda r: r.get("seed", 0))
    baseline_vals = _values(baseline_records)
    base_seeds = [r.get("seed") for r in baseline_records
                   if isinstance(r.get("metric"), (int, float))]

    cand_records = run_all(args.command, args.seeds, args.metric, workdir,
                            args.timeout, args.seed_env_var)
    cand_records.sort(key=lambda r: r.get("seed", 0))
    cand_vals = _values(cand_records)
    cand_seeds = [r.get("seed") for r in cand_records
                   if isinstance(r.get("metric"), (int, float))]
    if not cand_vals:
        print("[statsig] candidate had no successful run", file=sys.stderr)
        return 2
    result = decide(cand_vals, baseline_vals, args.direction, args.alpha,
                     paired=args.paired, bonferroni=args.bonferroni,
                     cand_seeds=cand_seeds, base_seeds=base_seeds)
    if args.paired and not result.get("paired", False):
        print("[statsig] WARNING: --paired requested but seeds do not match "
              "between candidate and baseline; fell back to independent tests",
              file=sys.stderr)
    result["candidate_records"] = cand_records
    result["baseline_path"] = str(baseline_path)

    report = Path(args.report) if args.report else workdir / "runs" / "statsig_report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2))
    # Pull p-values from whichever keys decide() emitted
    p_t = (result.get("welch_t") or result.get("paired_t") or {}).get("p_value")
    p_u = (result.get("mann_whitney") or result.get("wilcoxon_signed_rank") or {}).get("p_value")
    print(json.dumps({
        "decision": result["decision"],
        "delta_mean": result["delta_mean"],
        "paired": result.get("paired", False),
        "alpha_eff": result.get("alpha_eff", args.alpha),
        "p_t": p_t, "p_u": p_u,
        "ci95": [result["bootstrap_mean_delta"]["ci95_low"],
                  result["bootstrap_mean_delta"]["ci95_high"]],
        "latex_cell": result.get("latex", {}).get("table_cell"),
        "report": str(report),
    }, indent=2))
    return 0 if result["decision"] == "keep" else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Multi-seed statistical gate")
    sub = ap.add_subparsers(dest="sub", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--command", "-c", required=True)
    common.add_argument("--seeds", nargs="+", type=int, required=True)
    common.add_argument("--metric", "-m", required=True)
    common.add_argument("--timeout", type=int, default=7200)
    common.add_argument("--workdir", default=".")
    common.add_argument("--seed-env-var", default="SEED",
                        help="Env var exported to the command per seed")

    p = sub.add_parser("baseline", parents=[common])
    p.add_argument("--out", required=True)

    p = sub.add_parser("compare", parents=[common])
    p.add_argument("--baseline", required=True)
    p.add_argument("--direction", default="higher", choices=["higher", "lower"])
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--report", default=None)
    p.add_argument("--paired", action="store_true",
                    help="Use paired tests (matched-by-seed). Requires baseline "
                         "to have the same seeds as --seeds.")
    p.add_argument("--bonferroni", type=int, default=1, metavar="K",
                    help="Bonferroni K (alpha_eff = alpha / K). Use when "
                         "comparing against K baselines simultaneously.")

    args = ap.parse_args(argv)
    if args.sub == "baseline":
        return cmd_baseline(args)
    if args.sub == "compare":
        return cmd_compare(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
