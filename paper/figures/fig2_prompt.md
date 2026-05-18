# Fig. 2 — Drawing prompt for LineXRD architecture overview

This is a **schematic figure** (not data-driven). The visual style must
match the data figures already generated in this paper:

- `figs/training_curves.{pdf,png}` — uses CNN-NoPool=**blue (C0)**,
  GLA=**green (C2)**, RetNet=**red (C3)**, sans-serif Helvetica/Arial,
  thin (1 pt) strokes, white background, clean grid.
- `figs/pareto.{pdf,png}` — same palette, white background, simple
  legend in a corner.
- `figures/fig6_confusion.{pdf,png}` — sequential blue / grey
  colormaps for matrices.
- `figures/fig7_filter_spectra.{pdf,png}` — uses k=1 blue, k=3 orange,
  k=7 green, k=15 red for the four stem-kernel branches.

Paste the block below the `---` line into your preferred image-gen or
vector tool (ChatGPT GPT-image-1 1536×1024 quality=high / DALL·E 3 /
draw.io / Figma / Inkscape). Save the result as
`paper/figures/fig2_arch.pdf` at ≥600 dpi B/W or ≥400 dpi colour.

---

Create a horizontal single-panel pipeline diagram for a deep learning
paper. Target medium: single-column scientific paper figure, World
Scientific journal style (IJWMIP). Print width ≈ 5.0 inches (13 cm),
height ≈ 2.4 inches. White background, 1-pt black strokes, sans-serif
typeface (Helvetica or Arial).

Strict colour palette (MUST MATCH the rest of the paper):

- CNN-NoPool cell → soft blue   (matplotlib C0, `#1f77b4` at 30% opacity fill)
- Mamba-2 cell    → soft orange (matplotlib C1, `#ff7f0e` at 30% opacity fill)
- GLA cell        → soft green  (matplotlib C2, `#2ca02c` at 30% opacity fill)
- RetNet cell     → soft red    (matplotlib C3, `#d62728` at 30% opacity fill)
- Data-flow arrows and bounding boxes → solid black, 1 pt
- Stem-kernel mini-bars in the stem block → k=1 blue, k=3 orange,
  k=7 green, k=15 red (same 4 colours as fig7_filter_spectra)

Left-to-right flow with single-headed arrows. Six labelled blocks:

1. **INPUT** (leftmost): a small inset showing one period of a
   diffraction trace (3 Bragg-peak spikes against a flat baseline).
   Underneath: "PXRD x ∈ ℝ^8500", "2θ ∈ [5°, 90°]".

2. **STEM** (rounded rectangle, title "Multi-resolution Stem"):
   - inside: four parallel **mini-bar-plot icons** of the four kernel
     impulse responses (k=1 single bar, k=3 three bars, k=7 seven bars,
     k=15 fifteen bars under a Gaussian envelope), each in its colour
     above;
   - underneath the four icons: a single horizontal bar labelled
     "Concatenate → BN + GELU + Conv1d(k=3) + BN + GELU";
   - output arrow labelled "(L = 4251, D = 256)".

3. **ENCODER** (a 2×2 grid of 4 rounded rectangles inside a dashed
   parent box titled "4-way switch — only this block varies"):
   - Cell 1 (blue) "CNN11-NoPool" / "O(L·D²)"
   - Cell 2 (orange) "Mamba-2" / "O(L·D·d_state)"  (with a small
     "(not run in v4)" caption below)
   - Cell 3 (green) "GLA" / "O(L·D²)"
   - Cell 4 (red) "RetNet (chunkwise)" / "O(L·D²) per chunk"
   - A single output arrow leaves the parent box.

4. **POOL**: oval "Mean-pool over L".

5. **HEAD**: two boxes from the pool:
   - primary (bold border, black fill) "Linear D → 230 (SG)";
   - secondary (dashed border) "Linear D → 7 (CS), λ = 0.3 (disabled in v4)".

6. **OUTPUT / LOSS**: arrows from each head box converge into a
   bottom-right box "L = CE(p_SG, y_SG) + 0.3 · CE(p_CS, y_CS)" with
   the labels "p_SG ∈ Δ²³⁰" and "p_CS ∈ Δ⁷" on the connecting arrows.

Strict style requirements:

- Minimalist; no shadows; no 3-D effects; no gradients; no glow.
- Cells filled with pastel of the relevant colour at ≤30 % opacity;
  borders solid 1 pt black.
- All in-figure text ≥ 8 pt at print size.
- Colours readable in greyscale (cells must differ by fill texture,
  not just hue).
- No decorative title inside the figure (LaTeX caption will carry it).
- Output: vector PDF/SVG, landscape 3:2.

Reader takeaway after one glance:
"There is a shared multi-resolution CNN stem and a shared
classification head; only the middle encoder is swapped across four
linear-time backbones to chart a cost-accuracy frontier. The blue
(CNN), green (GLA), red (RetNet) coding matches the result figures
later in the paper."
