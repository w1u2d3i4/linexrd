# LineXRD 论文 IJWMIP 投稿 — 图片清单与绘制 Prompt

> 目标期刊: IJWMIP (World Scientific, 5.5×8.5 in 文本区)
> 图片要求: EPS / PDF, 黑白 ≥600 dpi 或 彩色 ≥400 dpi
> 每张图需 `\alttext{...}` 描述（无障碍）

---

## 图片总览（8 张）

| # | 名称 | 类型 | 谁画 | 在论文位置 | 状态 |
|---|------|------|------|----------|------|
| **F1** | Cost-Accuracy Frontier (Pareto plot) | Data | `make_figs.py` 自动 | §6.7 主结果 | ⏳ 等实验数据 |
| **F2** | Architecture overview | Schematic | **用户/AI 画** | §5.1 方法 | ✏️ 见 prompt |
| **F3** | Multi-resolution stem 详细图 | Schematic | **用户/AI 画** | §5.1.1 | ✏️ 见 prompt |
| **F4** | PXRD 信号示例 (含 Bragg 峰标注) | Data + 标注 | `make_figs.py` + 用户标注 | §1.1 引言 | ⏳ 部分 |
| **F5** | Training curves (4 backbones loss + sg_top1) | Data | `make_figs.py` 自动 | §6.1 主结果 | ⏳ 等实验数据 |
| **F6** | Confusion-matrix heatmap (orthorhombic 三联) | Data | `make_figs.py` 自动 | §6.8 失败分析 | ⏳ 等 ckpt |
| **F7** | Learned stem filter spectra | Data | `make_figs.py` 自动 | §7.2 frequency analysis | ⏳ 等 ckpt |
| **F8** | Empirical PXRD power spectrum | Data | `make_figs.py` 自动 | §6.6 multi-resolution | ✅ 现在能画（不需 ckpt） |

**用户需要画的**: F2, F3（schematic 架构图，~2 张）
**用户可选加工**: F4（在自动出的 PXRD 信号上手工标 hkl 峰）
**自动出的**: F1, F5, F6, F7, F8 — 实验跑完后由 `make_figs.py` 一键生成

---

## F2 — Architecture overview（**最重要的 schematic 图**）

### 用途
出现在 §5.1，作为方法节首图。读者看完这张图应该立即理解 LineXRD 整体管线。

### 内容要点（必含 8 个元素）
1. **左侧输入**：一条 1-D PXRD 谱，标注 "x ∈ ℝ^8500, 2θ ∈ [5°, 90°]"
2. **多尺度 stem**（4 个并行 Conv1d 分支）：
   - 标注每个分支的 kernel size `k∈{1, 3, 7, 15}`，stride=2
   - 4 路输出 concatenate → `(L=4251, D=256)` token 序列
3. **Encoder 选择框（4-way switch）**:
   - CNN11-NoPool / Mamba-2 / GLA / RetNet
   - 用 4 个并列圆角矩形，文字标注各自复杂度 `O(N·D²)` / `O(N·D·d_state)` / `O(N·D²)` / `O(N·D²)`
   - 框外标注："Shared stem + shared head; only this block varies"
4. **Mean-pool**：`(L, D) → (D,)`
5. **Classification head**:
   - 主头：`D → 230`（space group）
   - 辅头（虚线）：`D → 7`（crystal system, λ=0.3）
6. **右侧输出**：标注 "p_SG ∈ Δ^230, p_CS ∈ Δ^7"
7. **底部 loss**：`L = CE(p_SG, y_SG) + 0.3·CE(p_CS, y_CS)`
8. **整体配色**：蓝色 = data flow, 灰色 = 共享 modules, 4 种不同颜色区分 4 个 encoder

### 绘制 Prompt（AI 工具用，例如 mermaid/excalidraw/figma/dalle）

```
A clean horizontal pipeline diagram for a deep learning paper, single column, ~5.5 inches wide.
Left to right flow with arrows. Single-column scientific paper figure style (LaTeX, IJWMIP journal).
Black-and-white friendly but 4 distinct soft pastel colors allowed for the 4 encoder choices.

Components left→right:
1. INPUT: A small 1-D waveform icon labelled "PXRD x ∈ R^8500" with axis "2θ ∈ [5°, 90°]". Caption: "L_raw=8500 samples".
2. STEM (rounded rectangle box titled "Multi-resolution Stem"): four parallel Conv1d branches drawn as four small parallel arrows, each labelled with kernel size k=1, k=3, k=7, k=15, all stride=2. Below: "Concatenate → fuse (BN+GELU+Conv1d+BN+GELU)". Output arrow labelled "(L=4251, D=256)".
3. ENCODER (a 2x2 grid of 4 rounded rectangles inside a dashed parent box labelled "4-way switch — only this block varies"). Each cell:
   - CNN11-NoPool [light blue]   complexity O(L·D²)
   - Mamba-2 [light orange]      complexity O(L·D·d_state)
   - GLA [light green]           complexity O(L·D²)
   - RetNet (chunkwise) [light red] complexity O(L·D²) per chunk
4. POOL: a small "Mean-pool over L" oval.
5. HEAD: two parallel boxes — primary "Linear D→230 (SG)" with bold border, secondary "Linear D→7 (CS), λ=0.3" with dashed border.
6. OUTPUT: "p_SG ∈ Δ²³⁰" and "p_CS ∈ Δ⁷" labels with arrows to a final "Loss = CE(p_SG, y_SG) + 0.3·CE(p_CS, y_CS)" box.

Style: minimalist, no shadows, 1pt black strokes, sans-serif typeface (Helvetica or Arial),
rounded rectangles, light pastel fills only where colour is required for the 4 encoders.
Output: clean SVG / EPS at 600 dpi suitable for journal print.
```

### `\alttext` 文本
> *"Horizontal pipeline showing the LineXRD architecture: an 8500-sample PXRD input is processed by a multi-resolution multi-kernel CNN stem (k=1,3,7,15) into a length-4251 token sequence at width 256, then routed to one of four interchangeable linear-time encoders (CNN-NoPool, Mamba-2, GLA, RetNet), mean-pooled, and classified by a 230-way space-group head with an auxiliary 7-way crystal-system head."*

---

## F3 — Multi-resolution stem 详图（核心 IJWMIP 卖点图）

### 用途
出现在 §5.1.1，作为 multi-resolution analysis 的可视化锚点。这是 IJWMIP "wavelet/multiresolution" scope 最直接呼应的一张图。

### 内容要点
1. **顶部**：单条 PXRD 信号
2. **中部**：4 路并行卷积分支，**每路画出该 kernel 的脉冲响应（impulse response）**:
   - k=1: 单点 δ 函数图标 → "level-0 detail (identity)"
   - k=3: 3-点窄锐脉冲 → "level-1 detail (sharp peaks)"
   - k=7: 7-点中宽脉冲 → "level-2 detail (standard peaks)"
   - k=15: 15-点宽脉冲 → "level-3 smooth (broadened peaks)"
3. **频域子图**（小插图）：4 路滤波器在频域的覆盖，类似 wavelet packet sub-band tiling
4. **底部**：concatenate 后的 token 序列 + Bragg 峰位还原说明

### 绘制 Prompt

```
A multi-panel scientific schematic for a wavelet-multiresolution journal (IJWMIP).
Single column, ~5.5 inches wide, two-row layout.

Top row: A 1-D PXRD intensity trace (toy, with three labelled Bragg peaks
at 2θ=20°, 35°, 60°). Below the trace, 4 fan-out arrows go to 4 parallel
sub-panels labelled "k=1", "k=3", "k=7", "k=15", each containing:
  - The convolutional kernel's impulse response drawn as a small bar plot
    (k=1: single bar, k=3: 3 short bars, k=7: 7 medium bars, k=15: 15 wide
    bars with smooth Gaussian envelope).
  - A label: "level-0 detail", "level-1 detail", "level-2 detail",
    "level-3 smooth approximation" respectively.

Bottom row: The 4 branches re-converge into a single concatenated token
sequence labelled "(L=4251, D=256) frame coefficients". On the right of
the bottom row, a small inset showing the four kernels' Fourier-amplitude
responses |G_k(ω)| as 4 overlapping coloured curves on a shared
log-frequency axis, visualising the multi-resolution tiling of [0, π].

Style: minimalist, monochrome with 4 distinct subdued colours for the
branches (blue, green, orange, purple), 1-pt strokes, sans-serif labels.
600 dpi PDF/SVG suitable for journal print.
```

### `\alttext` 文本
> *"Multi-resolution stem schematic: a PXRD trace is convolved by four parallel Conv1d kernels of widths k=1, 3, 7, 15 (level-0 to level-3 wavelet-packet analogues), whose impulse responses are shown as small bar plots; an inset shows the corresponding frequency-domain tiling, an oversampled multi-resolution analysis of the input."*

---

## F4 — PXRD 信号示例（可选 schematic-with-data）

### 内容
- 真实从 CCDC-SG 抽 2 条 PXRD 样本
- 一条为 cubic（高对称），一条为 monoclinic（低对称）
- **用户手工标注**关键 Bragg 峰的 (hkl) 索引
- 在峰附近用箭头标注 "Bragg peak: 2θ = arcsin(λ/2d_{hkl})"

### `make_figs.py` 会自动出**原始信号**；用户在此基础上加 hkl 标注（PowerPoint/Illustrator/Inkscape）。

---

## F5 — Training curves（4 backbones）

### 自动出（matrix v3 跑完后）
4 个子图（CNN / Mamba-2 / GLA / RetNet），每个画：
- 训练 loss
- val SG top-1
- val CS top-1
- val macroR

横轴 epoch (0-40)。三 seed 阴影。

**已写好 `make_figs.py` 框架**，跑完实验直接执行即可。

---

## F6 — Confusion matrix heatmap (orthorhombic 三联误判)

### 自动出
- 抽 best CNN / Mamba-2 checkpoint
- 计算 230×230 confusion matrix
- 高亮 `{Pnma (62), Pnam, Pmcn}` 三个 SG 的 3×3 子块
- 旁边附一个 7×7 crystal-system confusion 子图

---

## F7 — Learned stem filter spectra

### 自动出
- 加载训练好的 stem 模块
- 对每个 kernel `g_k` 做 FFT
- 绘 |G_k(ω)| 在 [0, π] 上的频率响应曲线
- 验证 §5.1.1 lemma 的多尺度覆盖（log-spaced centres at 0.45π, 0.24π, 0.12π, 0.05π）

---

## F8 — Empirical PXRD power spectrum（**现在就能画**）

### 内容
对 CCDC-SG train 集所有 PXRD 做 FFT，取平均 `|X(ω)|²`，绘 ω∈[0, π] 上的归一化谱。
应能看出三波段结构（DC / 10-200 cycle 平台 / >400 cycle 尾噪声）。

### 写代码现在就能产出
用现成数据，不依赖 ckpt。我现在就加进 `make_figs.py`。

---

## 总结：用户需要做的 4 件事

| Idx | 任务 | 工具建议 | 紧急度 |
|-----|------|---------|--------|
| 1 | **画 F2 (architecture overview)** | excalidraw / draw.io / dalle / figma | 高（论文核心） |
| 2 | **画 F3 (multi-resolution stem)** | 同上 | 高（IJWMIP scope 卖点） |
| 3 | F4 PXRD 上手工标 hkl 峰 | PowerPoint / Illustrator | 中（可后期） |
| 4 | 实验跑完后 `cd paper && python make_figs.py` 生成 F1/F5/F6/F7/F8 | 命令行 | 自动 |

**5 张图都靠数据，2 张图需要你手工画。**

---

## 实验状态（决定 F1/F5/F6/F7 何时能出）

- Matrix v3 进度（CCDC-SG, 40 epoch, GPU1）:
  - CNN: **ep 13/40** 跑中（loss 平稳，sg_top1 仍振荡）
  - GLA: 待 CNN 完成（fix 后已 smoke 验证）
  - RetNet: 待 GLA 完成（chunkwise 已 smoke 验证）
- 预计完成时间: ~6 h
- 完成后 `make_figs.py` 一次生成 F1/F5/F6/F7/F8 五张数据图
