#!/bin/bash
set -euo pipefail

# ============================================================
# Quality Gate — runs after every passing benchmark.
# If this script exits non-zero, the experiment is logged as
# "checks_failed" and code changes are reverted.
#
# Add your correctness constraints here.
# ============================================================

PYTHON="python"

$PYTHON -c "
import json, sys

# Example: check that a critical metric stays above threshold
# Modify this for your project
try:
    with open('results/eval_results.json') as f:
        r = json.load(f)

    # Example constraint: accuracy must stay >= 0.80
    # acc = r.get('accuracy', 0)
    # assert acc >= 0.80, f'accuracy {acc} < 0.80 threshold'

    print('Checks passed')
except Exception as e:
    print(f'FAIL: {e}')
    sys.exit(1)
"
