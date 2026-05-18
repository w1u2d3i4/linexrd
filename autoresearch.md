# Autoresearch: LineXRD — Linear-Time Backbone Frontier on Full-Resolution PXRD

## Objective
量化 4 种主流 backbone 在 **8500-token 全分辨率 PXRD** 上的空间群分类性能，绘制 **cost-accuracy frontier**：
- CNN11-NoPool (baseline, O(N) 卷积)
- Mamba-2 (selective SSM, O(N))
- GatedLinearAttention / GLA (O(N))
- RetNet (O(N))

目标投稿：**IJWMIP** (multi-resolution + 1D long-signal scope)。

## Metrics
- **Primary**: `sg_top1` — Space Group top-1 accuracy on val (higher is better) on **ccdc_sg** (default) 或 inor / coremof19
- **Secondary**:
  - `cs_top1` — Crystal System top-1 acc
  - `sg_macro_f1` — 长尾 macro-F1
  - `params_M` — backbone 参数量 (M)
  - `train_sec_per_epoch` — 单 epoch 训练秒数
  - `infer_ms_per_sample` — 推理 latency

## Baseline
CNN11-NoPool 在 ccdc_sg 上 SG top-1 (smoke 12 epoch):
- sg_top1 = TBD（D1 跑完填）
- cs_top1 = TBD
- params_M = ~3.0

## How to Run
`./autoresearch.sh` — 训练 N epoch 后输出 `METRIC name=value` 行供 runner.py 解析

## Files in Scope
- `src/models/backbones.py` — Mamba-2 / GLA / RetNet 实现
- `src/models/cnn_baseline.py` — CNN11-NoPool 参考实现 (来自 ICL/models/Resnet1D.py)
- `src/train_linexrd.py` — 训练入口（包装 ICL/train.py，加 backbone 选项）
- `src/eval_linexrd.py` — 评估 + METRIC 输出
- `autoresearch.sh` — HYPERPARAMS section
- `paper/main.tex` — IMRAD 论文

## Off Limits
- `/opt/data/private/xrd2c_data/` — 已处理的数据（只读）
- `/opt/data/private/ICL/` — 原始 ICL 代码（只读，参考）
- `src/data/loader.py` — 数据加载需与 ICL 一致

## Constraints
- 单卡 A100-40G（实际可见 2×A100-40G）
- batch_size ≤ 256
- 不可改 8500 token 全分辨率（核心实验假设）
- 不可破坏 ccdc_sg / inor / coremof19 train/val/test split

## What's Been Tried
（待填）

### Experiment 0: CNN11-NoPool baseline on ccdc_sg (smoke 12 epoch)
- Config: lr=1e-3, batch=128, AMP fp16
- Result: 待跑
- Insight: 待填

## Ideas Backlog
- DiffAttn-hybrid backbone（contingency 如果纯 Mamba 不优于 CNN）
- Hierarchical head (CS+SG multi-task)（v2 ablation）
- Down-resolution 8500 vs 2000 在 Mamba-2 + CNN 上消融
- RRUFF zero-shot transfer（supplement）

## Target Numbers (期望，论文 placeholder 用)
基于 SimXRD-4M CNN11-NoPool 75.8% 和 XRDMamba 在 mp20 上 82.3% 的先验：
- 期望 Mamba-2 / GLA 在 ccdc_sg 上 SG top-1 ≈ 78-82%
- 期望 RetNet ≈ 76-79%
- 期望 CNN11-NoPool baseline ≈ 73-76%
