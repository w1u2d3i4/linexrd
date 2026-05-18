# Fig. 3 — Drawing prompt for Multi-Resolution Stem detail

This is the **central IJWMIP-scope figure** of the paper: it visually
links the multi-kernel CNN stem to the wavelet / multi-resolution
analysis tradition that the journal centres on. The figure should be
**directly comparable to Fig. 7 (learned filter spectra)** which uses
the same 4-colour kernel palette:

- k = 1 → blue   (matplotlib C0)
- k = 3 → orange (matplotlib C1)
- k = 7 → green  (matplotlib C2)
- k = 15 → red   (matplotlib C3)

Save the result as `paper/figures/fig3_stem.pdf` at ≥600 dpi B/W or
≥400 dpi colour.

---

Create a two-row scientific schematic for a wavelet-multiresolution
journal (IJWMIP, World Scientific). Single column, target print width
≈ 5.0 inches (13 cm); two rows. White background, 1-pt black strokes,
sans-serif typeface (Helvetica / Arial).

Strict colour palette (MUST match Fig. 7 of this paper):

- k=1 → matplotlib C0  (`#1f77b4`)
- k=3 → matplotlib C1  (`#ff7f0e`)
- k=7 → matplotlib C2  (`#2ca02c`)
- k=15 → matplotlib C3 (`#d62728`)

Each sub-panel and each frequency-response curve must be in the
kernel's assigned colour; bar-plot fills at 30 % opacity, line plots
at 70 % opacity, all outlines/strokes 1 pt black.

**Top row — "Decomposition stage"** (full row width):

- Small left inset: one cycle of a simulated PXRD intensity trace
  with three Bragg peaks labelled "(1 1 0)", "(2 0 0)", "(1 1 1)".
- Four downward fan-out arrows lead to four side-by-side rounded
  rectangles. Each rectangle contains a small bar-plot icon of that
  kernel's impulse response:
  - "k = 1" — single tall blue bar; subtitle "level-0 detail (identity)".
  - "k = 3" — three short orange bars; subtitle "level-1 (sharp peaks)".
  - "k = 7" — seven medium green bars under a Gaussian envelope;
    subtitle "level-2 (standard peaks)".
  - "k = 15" — fifteen wide red bars under a wider Gaussian envelope;
    subtitle "level-3 (broadened peaks)".
- A small annotation arrow from the input trace to the panel row
  labelled "Conv1d stride = 2".

**Bottom row — "Frame coefficients + frequency tiling"** (full row width):

- Left two-thirds: the four sub-panels re-converge into a single
  concatenated horizontal bar labelled "Frame coefficients (L = 4251,
  D = 256)".
- Right one-third: a **frequency-domain inset** showing four
  overlapping bell-shaped low-pass curves on a shared log-frequency
  axis ω ∈ [0, π]; each curve in its kernel colour. The k=1 curve is
  a flat line at 1.0 across the axis; k=3 is a gentle low-pass with
  ~50 % roll-off near ω/π = 0.7; k=7 a stronger low-pass; k=15 a
  broad low-pass with mild notch structure beyond ω/π = 0.5. (This is
  the empirical pattern actually observed in Fig. 7; do NOT draw the
  textbook log-spaced separated bands, draw the observed overlapping
  pattern.)
- Inset caption: "|G_k(ω)| (learned; matches Fig. 7)".
- A small bracket on the bottom right of the inset labelled "soft
  multi-resolution tiling".

**Annotations**:

- One small caption beneath the whole figure space (placeholder for
  the LaTeX caption; do not draw it inside the figure).

**Style requirements**:

- Minimalist; no 3-D effects; no shadows; no gradients.
- All in-figure text ≥ 8 pt at print size.
- Output: vector (PDF/SVG); aspect 3:2 landscape.

Reader takeaway after one glance:
"The four-kernel stem performs a learned, oversampled
multi-resolution analysis of the PXRD signal. Each kernel covers a
different angular-resolution scale, and the four learned filter
spectra together tile the frequency axis — but as Fig. 7 shows, the
learned tiling is soft (overlapping low-pass bands), not the textbook
log-spaced wavelet packet basis."
