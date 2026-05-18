# LineXRD

A multi-resolution linear-time sequence-model frontier for
full-resolution powder X-ray diffraction (PXRD) space-group
classification.

This repository accompanies the manuscript
*"LineXRD: A Multi-Resolution Linear-Time Sequence-Model Frontier for
Full-Resolution Powder X-Ray Diffraction"* submitted to the
International Journal of Wavelets, Multiresolution and Information
Processing (IJWMIP, World Scientific).

## Repository layout

```
linexrd/
├── src/                        # training code
│   ├── train_linexrd.py        # supervised training entry point
│   ├── models/backbones.py     # CNN-NoPool / GLA / Mamba-2 / RetNet (chunkwise)
│   └── data/loader.py          # CCDC-SG / INOR / CoREMOF19 NPY loader
├── autoresearch/               # autoresearch pipeline tools (runner, checks)
├── run_matrix_v4.sh            # canonical 40-epoch 3-backbone matrix
├── run_ablations.sh            # downsample + cross-dataset ablations
├── autoresearch.{md,sh,checks.sh}   # project-level autoresearch config
└── paper/
    ├── main.tex                # IJWMIP-formatted manuscript
    ├── ws-ijwmip.{cls,bst}     # official World Scientific class + bibstyle
    ├── figures/                # data figures (fig 2, 6, 7, 8) + uploaded schematic
    ├── figs/                   # data figures (pareto, training_curves, barchart, status)
    └── make_*.py               # figure-generation scripts (offline, from logs/ckpts)
```

## Setup

```bash
# Python env: PyTorch 2.1+, CUDA 12, single A100-40G is enough.
conda activate /opt/data/private/icml2026         # local convention
pip install rdkit matplotlib  # if not present
```

## Training one backbone

```bash
BACKBONE=gla DATASET=ccdc_sg EPOCHS=40 BATCH_SIZE=64 \
  D_MODEL=256 DEPTH=6 LR=2e-4 bash autoresearch.sh
```

`autoresearch.sh` writes:

- `checkpoints/<run>/best.pt`     — best validation checkpoint
- `logs/<run>.log`                — full epoch curves
- METRIC lines parsed by `autoresearch/runner.py`

## Reproducing the v4 main matrix (40 epoch, class-weighted CE)

```bash
bash run_matrix_v4.sh             # CNN → GLA → RetNet, sequential
bash run_ablations.sh             # Abl 1 (downsample) + Abl 3 (cross-dataset)
```

## Regenerating paper figures from logs / checkpoints

```bash
cd paper
python make_figs.py                            # Fig 1, 5, 6 (data scatter, curves)
python make_pxrd_spectrum.py                   # Fig 3 (empirical PXRD spectrum)
python make_confusion_and_filters.py           # Fig 6, 7 (confusion + filter spectra)
```

All scripts read measured metrics from `../logs/` and best checkpoints
from `../checkpoints/`. They emit PDFs into `paper/figs/` and
`paper/figures/`.

## Compiling the paper

```bash
cd paper
pdflatex main.tex && pdflatex main.tex     # two passes for cross-refs
```

`ws-ijwmip.cls` is the official World Scientific IJWMIP class; do not
substitute.

## Headline results (CCDC-SG, 40 epoch, class-weighted CE, single seed)

| Backbone           | SG top-1 | macro-recall | params (M) | s/epoch |
|--------------------|----------|--------------|------------|---------|
| CNN-NoPool         | 33.36 %  |  5.84 %      | 1.44       | 184     |
| **GLA**            | **38.29 %** | 16.19 %   | 4.40       | 478     |
| RetNet (chunkwise) | 33.79 %  | **16.89 %**  | 4.99       | 942     |

GLA wins on top-1 (+4.93 pt over CNN). On the long-tail
macro-recall metric, GLA and RetNet roughly *triple* the CNN
baseline — the most decisive empirical signature of this study.

## Authors

- Di Wu, Mingyang Sun, **Pengkun Wang\***, Yang Wang
- University of Science and Technology of China (Hefei / Suzhou)
- Corresponding author: pengkun@ustc.edu.cn

## Funding

National Natural Science Foundation of China (NSFC) Nos. 62402472,
12227901; Natural Science Foundation of Jiangsu Province No.
BK20240461; Project of Stable Support for Youth Team in Basic Research
Field, CAS No. YSBR-005; Academic Leaders Cultivation Program at USTC.

## License

Code released for academic use. See LICENSE file (to be added on
publication).
