# LineXRD

Core code for sparse peak-set learning with verifiable scientific-record
retrieval for PXRD space-group prediction.

This source-only repository contains the peak-set encoder, cosine retrieval,
conditional expert fusion, an independent evidence verifier, and four local
command-line entry points. Datasets, experimental results, logs, fitted models,
checkpoints, and paper files are not distributed here.

## Install

Use Python 3.10 or newer. Install the PyTorch build appropriate for your CPU or
CUDA environment, then install this package from the repository root:

```bash
python -m pip install -e .
```

All commands default to CPU; pass `--device cuda` to use an available GPU.

## Local input format

Prepare your own NumPy `.npz` files outside the repository:

- `features`: finite `float32` intensities with shape `(N, 8500)` on a common
  angular grid. Apply the same preprocessing to training, reference, and query
  patterns.
- `labels`: integer space-group indices in `[0, 229]`, required for training,
  validation, reference-bank construction, and calibration. Prediction does
  not read query labels. Returned `space_group` values are one-based, `[1, 230]`.
- Optional reference metadata: `record_id`, `entity_id`, and `source`, each a
  length-`N` NumPy Unicode array. These fields are returned with retrieved rows.

No data are downloaded automatically. Use entity-disjoint training, calibration,
validation, and evaluation membership. Reference records must be selected
according to your evaluation protocol; do not insert evaluation entities.

## Basic workflow

The paths below are placeholders for your own local files. Output directories
are created when needed. Choose new output filenames; existing files are not
overwritten by a new command.

### 1. Train the peak-set encoder

```bash
python -m linexrd train --train /path/to/train.npz \
  --validation /path/to/validation.npz --checkpoint artifacts/pst.pt
```

The default encoder uses 512 selected samples, 70 per-sample features, six
256-dimensional Transformer blocks, and eight attention heads. Selection is a
local-maximum-biased Top-K rule; adjacent samples can still be selected. The
default uses raw intensities for the residual channel; cubic background
subtraction is available with `--subtract-background`.

The compact trainer uses AdamW, class-weighted cross-entropy, label smoothing,
gradient clipping, cosine scheduling, and validation checkpoint selection. It
is a basic training entry point, not an experiment scheduler or an automated
reproduction of all paper evaluation protocols.

### 2. Build a reference bank

```bash
python -m linexrd build-bank --checkpoint artifacts/pst.pt \
  --data /path/to/reference_records.npz --output artifacts/bank.npz
```

This saves normalized CLS embeddings, reference labels, and optional provenance.
Use the same encoder checkpoint for bank construction, calibration, and queries.

### 3. Fit fusion and verification on calibration examples

```bash
python -m linexrd calibrate --checkpoint artifacts/pst.pt \
  --bank artifacts/bank.npz --data /path/to/calibration.npz \
  --output artifacts/calibration.pkl --k 4 --temperature 0.03
```

Fusion uses 13 posterior, agreement, and similarity statistics. Its tree is fit
on clean calibration disagreements for which exactly one expert is correct.
`--mapping hard|soft` controls expert selection or mixing;
`--benefit-threshold off` disables the advantage guard. Tree depth and iteration
count are exposed through `--tree-depth` and `--tree-iterations`.

The separate relevance tree uses 16 statistics and is fit on clean evidence and
query-mismatched, recomputed evidence generated in memory from calibration
queries. Both checks must pass: recomputed similarities must agree within
`1e-4`, and the relevance score must reach `--relevance-threshold`. Otherwise
the original PST posterior is returned. Exactly the same verifier can wrap
the kNN posterior using `linexrd.verification.apply`.

This command fits one configuration. Select k, temperature, tree settings, and
thresholds using development data, then keep them fixed for final evaluation.
Calibration pickle files must be created locally or obtained from a trusted
source. Keep the scikit-learn version consistent when saving and loading them.

### 4. Predict

```bash
python -m linexrd predict --checkpoint artifacts/pst.pt \
  --bank artifacts/bank.npz --calibration artifacts/calibration.pkl \
  --data /path/to/queries.npz --output artifacts/predictions.npz
```

For PST-only prediction, omit both `--bank` and `--calibration`. Retrieval outputs
include class probabilities, acceptance flags, fusion weights, reference-row
indices, cosine scores, and any supplied record metadata. Returned row indices
refer to the bank file used for that invocation.

## Source layout

```text
src/linexrd/
  model.py          # Peak selection, angular/local features, PST encoder
  fusion.py         # kNN voting and conditional expert tree
  verification.py   # Score consistency, relevance, and PST fallback
  runtime.py        # Local array I/O, embeddings, and cosine retrieval
  cli.py            # Train, build-bank, calibrate, predict
```

The repository uses a `.gitignore` allowlist that admits only source code and
the dependency and documentation files. Keep datasets, generated arrays,
training console captures, weights, and evaluation outputs outside Git.
