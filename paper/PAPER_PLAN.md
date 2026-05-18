# LineXRD — paper plan (for hero_figure_bundle)

**Title**: LineXRD: A Multi-Resolution Linear-Time Sequence-Model Frontier for Full-Resolution Powder X-Ray Diffraction

**One-sentence contribution**: We benchmark four linear-time ($O(N)$) sequence backbones (CNN-NoPool, Mamba-2, GLA, RetNet) under a shared multi-resolution multi-kernel convolutional stem on full-resolution ($N{=}8500$) PXRD space-group classification, and chart the cost--accuracy frontier across three public CCDC / inorganic / MOF corpora.

**Venue**: International Journal of Wavelets, Multiresolution and Information Processing (IJWMIP), World Scientific.

## Figure list

| Idx | Figure | Description |
|-----|--------|-------------|
| Fig 1 | Cost-accuracy frontier on CCDC-SG | one-panel Pareto scatter, x = train-time-per-epoch (s, A100-40G), y = SG top-1 (%) on the validation set, 4 points (CNN-NoPool, Mamba-2, GLA, RetNet) with one-sigma error bars over 3 seeds; Pareto-front polyline highlights non-dominated backbones; secondary inset plots inference latency vs. SG top-1 |
| Fig 2 | Architecture overview | schematic |
| Fig 3 | Multi-resolution stem | schematic |
| Fig 4 | PXRD signal example with Bragg-peak annotations | data + annotation |
| Fig 5 | Training curves (4 backbones, 3 seeds) | data |
| Fig 6 | Confusion-matrix slice on orthorhombic SGs | data |
| Fig 7 | Learned stem filter spectra | data |
| Fig 8 | Empirical PXRD power spectrum | data |

### Hero Figure spec

The hero figure is a **single-panel cost--accuracy frontier** for the CCDC-SG validation set:

- four labelled points, one per backbone (CNN-NoPool, Mamba-2, GLA, RetNet);
- x-axis = wall-clock training seconds per epoch;
- y-axis = best space-group top-1 accuracy (%);
- one-sigma vertical error bars over 3 seeds (where available);
- Pareto-front polyline through the non-dominated points (expected: CNN, GLA, Mamba-2; RetNet expected to be dominated);
- small inset (lower-right corner) plotting per-pattern inference latency (ms) on x-axis vs. SG top-1 on y-axis with the same four points;
- minimalist publication-grade style (white background, thin strokes, one accent colour for the recommended backbone Mamba-2, neutral grey for the other three);
- in-figure text in Times New Roman 10 pt minimum, deuteranopia-safe colours, greyscale-distinguishable.

## Key quantitative anchors

- CNN-NoPool: 75.4 % SG top-1, 140 s/epoch, 3.0 M params (projected)
- GLA:        78.2 % SG top-1, 165 s/epoch, 3.6 M params (projected)
- RetNet:     77.1 % SG top-1, 185 s/epoch, 3.4 M params (projected)
- Mamba-2:    80.1 % SG top-1, 220 s/epoch, 4.1 M params (projected)
