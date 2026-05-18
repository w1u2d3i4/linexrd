# LineXRD

A multi-resolution linear-time sequence-model frontier for
full-resolution powder X-ray diffraction (PXRD) space-group
classification.

Code accompanying the manuscript *"LineXRD: A Multi-Resolution
Linear-Time Sequence-Model Frontier for Full-Resolution Powder X-Ray
Diffraction"* submitted to *International Journal of Wavelets,
Multiresolution and Information Processing* (IJWMIP, World Scientific).

## Layout

```
linexrd/
├── src/
│   ├── train_linexrd.py        # supervised training entry point
│   ├── models/backbones.py     # CNN-NoPool / GLA / Mamba-2 / RetNet (chunkwise)
│   └── data/loader.py          # CCDC-SG / INOR / CoREMOF19 NPY loader
├── train.sh                    # env-var-driven training wrapper
├── run_matrix.sh               # canonical 3-backbone 40-epoch matrix
├── run_ablations.sh            # downsample + cross-dataset ablations
└── paper/
    ├── main.tex                # IJWMIP manuscript
    ├── ws-ijwmip.{cls,bst}     # official World Scientific class + bibstyle
    ├── figures/  figs/         # data + schematic figures
    └── make_*.py               # figure-generation scripts (offline)
```

## Setup

```bash
# PyTorch 2.1+, CUDA 12, a single A100-40G is enough.
pip install torch numpy matplotlib
```

The training data (CCDC-SG / INOR / CoREMOF19 `.npy` files) must be
placed under a directory the loader can see. Adjust the `DATA_ROOT`
constant at the top of `src/data/loader.py` to point at your local
copy.

## Training one backbone

```bash
BACKBONE=gla DATASET=ccdc_sg EPOCHS=40 BATCH_SIZE=64 \
  D_MODEL=256 DEPTH=6 LR=2e-4 bash train.sh
```

Outputs:

- `checkpoints/<run>/best.pt`     — best validation checkpoint
- `logs/<run>.log`                — full epoch curves and `METRIC` lines

## Reproducing the v4 main matrix (40 epoch, class-weighted CE)

```bash
bash run_matrix.sh                 # CNN -> GLA -> RetNet, sequential
bash run_ablations.sh              # Abl 1 (downsample) + Abl 3 (cross-dataset)
```

## Regenerating paper figures

```bash
cd paper
python make_figs.py                            # Pareto / training curves / barchart
python make_pxrd_spectrum.py                   # empirical PXRD power spectrum
python make_confusion_and_filters.py           # confusion + learned filter spectra
```

All scripts read measured metrics from `../logs/` and best checkpoints
from `../checkpoints/` and emit PDFs into `paper/figs/` and
`paper/figures/`.

## Compiling the paper

```bash
cd paper
pdflatex main.tex && pdflatex main.tex     # two passes for cross-refs
```

`ws-ijwmip.cls` is the official World Scientific IJWMIP class.

## Headline results (CCDC-SG, 40 epoch, class-weighted CE, single seed)

| Backbone           | SG top-1     | macro-recall | params (M) | s/epoch |
|--------------------|--------------|--------------|------------|---------|
| CNN-NoPool         | 33.36 %      |  5.84 %      | 1.44       | 184     |
| **GLA**            | **38.29 %**  | 16.19 %      | 4.40       | 478     |
| RetNet (chunkwise) | 33.79 %      | **16.89 %**  | 4.99       | 942     |

GLA wins on top-1 (+4.93 pt over CNN). On the long-tail
macro-recall metric, GLA and RetNet roughly *triple* the CNN
baseline -- the most decisive empirical signature of this study.

## Authors

Di Wu, Mingyang Sun, **Pengkun Wang\***, Yang Wang  
University of Science and Technology of China (Hefei / Suzhou)  
Corresponding author: pengkun@ustc.edu.cn

## Funding

National Natural Science Foundation of China (NSFC) Nos. 62402472,
12227901; Natural Science Foundation of Jiangsu Province No.
BK20240461; Project of Stable Support for Youth Team in Basic Research
Field, CAS No. YSBR-005; Academic Leaders Cultivation Program at USTC.

## License

Code released for academic use. See LICENSE file (to be added on
publication).
