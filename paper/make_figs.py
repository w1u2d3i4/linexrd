"""LineXRD result figures from v4 (measured) training logs.

Inputs: /opt/data/private/linexrd/logs/v4_{cnn,gla,retnet}_ccdc_sg.out
Outputs: paper/figs/{training_curves, pareto, barchart, status}.{pdf,png}

Mamba-2 is NOT included — its pure-PyTorch sequential scan is too slow to
run within budget on the shared A100. We mark it explicitly as "not run".
"""
from __future__ import annotations
import re
import pathlib
import numpy as np
import matplotlib.pyplot as plt

ROOT = pathlib.Path('/opt/data/private/linexrd')
LOGS = ROOT / 'logs'
OUT = ROOT / 'paper' / 'figs'
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'legend.fontsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'figure.dpi': 150, 'savefig.dpi': 600, 'pdf.fonttype': 42,
})


def parse_curve(p: pathlib.Path | None):
    if p is None or not p.exists():
        return None
    txt = p.read_text()
    rx = re.compile(r"\[ep (\d+)\] loss=(\S+) sg_top1=(\S+) cs_top1=\S+ macroR=(\S+) "
                    r"train_s=(\S+) val_ms=(\S+)")
    rows = rx.findall(txt)
    if not rows:
        return None
    arr = np.array([[float(v) if v != 'nan' else np.nan for v in r] for r in rows])
    return dict(epoch=arr[:, 0], loss=arr[:, 1], sg=arr[:, 2],
                macroR=arr[:, 3], train_s=arr[:, 4], val_ms=arr[:, 5])


def parse_metric(p: pathlib.Path | None):
    if p is None or not p.exists():
        return None
    txt = p.read_text()
    m = dict(re.findall(r"METRIC (\w+)=(\S+)", txt))
    return {k: float(v) for k, v in m.items()}


cnn = parse_curve(LOGS / "v4_cnn_ccdc_sg.out")
gla = parse_curve(LOGS / "v4_gla_ccdc_sg.out")
ret = parse_curve(LOGS / "v4_retnet_ccdc_sg.out")
cnn_m = parse_metric(LOGS / "v4_cnn_ccdc_sg.out")
gla_m = parse_metric(LOGS / "v4_gla_ccdc_sg.out")
ret_m = parse_metric(LOGS / "v4_retnet_ccdc_sg.out")

PALETTE = {'CNN-NoPool': 'C0', 'GLA': 'C2', 'RetNet': 'C3', 'Mamba-2': 'C4'}


# ====================== Fig 1: training curves =========================
fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.0))

# Top-left: loss
ax = axes[0, 0]
for name, d in [('CNN-NoPool', cnn), ('GLA', gla), ('RetNet', ret)]:
    if d is None:
        continue
    ax.plot(d['epoch'], d['loss'], 'o-', color=PALETTE[name], lw=1.2,
            ms=2.5, label=name)
ax.set_xlabel('Epoch'); ax.set_ylabel('Train cross-entropy loss')
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax.set_title('(a) Training loss')

# Top-right: SG top-1
ax = axes[0, 1]
for name, d in [('CNN-NoPool', cnn), ('GLA', gla), ('RetNet', ret)]:
    if d is None:
        continue
    ax.plot(d['epoch'], d['sg'], 'o-', color=PALETTE[name], lw=1.2,
            ms=2.5, label=name)
ax.set_xlabel('Epoch'); ax.set_ylabel('Val SG top-1 (%)')
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax.set_title('(b) Space-group top-1 accuracy')

# Bottom-left: macroR
ax = axes[1, 0]
for name, d in [('CNN-NoPool', cnn), ('GLA', gla), ('RetNet', ret)]:
    if d is None:
        continue
    ax.plot(d['epoch'], d['macroR'], 'o-', color=PALETTE[name], lw=1.2,
            ms=2.5, label=name)
ax.set_xlabel('Epoch'); ax.set_ylabel('Val macro-recall (%)')
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax.set_title('(c) Long-tail macro-recall')

# Bottom-right: train-time
ax = axes[1, 1]
names = []; vals = []; cols = []
for name, m in [('CNN-NoPool', cnn_m), ('GLA', gla_m), ('RetNet', ret_m)]:
    if m is None:
        continue
    names.append(name); vals.append(m['train_sec_per_epoch']); cols.append(PALETTE[name])
ax.bar(names, vals, color=cols, edgecolor='k', lw=0.5)
for n, v in zip(names, vals):
    ax.text(n, v + 20, f'{v:.0f}s', ha='center', fontsize=8)
ax.set_ylabel('Wall-clock s / epoch')
ax.set_title('(d) Training cost')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / 'training_curves.pdf'); plt.savefig(OUT / 'training_curves.png')
plt.close()


# ====================== Fig 2: cost-accuracy Pareto ====================
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4))

# Left: SG top-1 vs train cost
ax = axes[0]
pts = []
for name, m in [('CNN-NoPool', cnn_m), ('GLA', gla_m), ('RetNet', ret_m)]:
    if m is None:
        continue
    x = m['train_sec_per_epoch']
    y = 100 * m['best_sg_top1']
    ax.scatter(x, y, s=140, marker='o', color=PALETTE[name],
               edgecolor='k', zorder=3,
               label=f"{name} ({m['params_M']:.1f} M)")
    ax.annotate(name, (x, y), xytext=(8, 6), textcoords='offset points', fontsize=8)
    pts.append((x, y, name))
# Pareto front: a point is dominated if there's another with lower x AND higher y
pts_sorted = sorted(pts)
front = []
best_y = -1
for x, y, n in pts_sorted:
    if y > best_y:
        front.append((x, y))
        best_y = y
if len(front) >= 2:
    fx = [p[0] for p in front]; fy = [p[1] for p in front]
    ax.plot(fx, fy, '--', color='gray', alpha=0.5, lw=1.2, label='Pareto front', zorder=2)
ax.set_xlabel('Training time / epoch (s, A100-40G)')
ax.set_ylabel('SG top-1 (%)')
ax.set_title('(a) cost vs SG top-1')
ax.legend(fontsize=8, loc='lower right'); ax.grid(alpha=0.3)

# Right: macroR vs inference latency
ax = axes[1]
for name, m in [('CNN-NoPool', cnn_m), ('GLA', gla_m), ('RetNet', ret_m)]:
    if m is None:
        continue
    x = m['infer_ms_per_sample']
    y = 100 * m['sg_macro_recall']
    ax.scatter(x, y, s=140, marker='s', color=PALETTE[name],
               edgecolor='k', zorder=3)
    ax.annotate(name, (x, y), xytext=(8, 6), textcoords='offset points', fontsize=8)
ax.set_xlabel('Inference latency (ms / pattern)')
ax.set_ylabel('Macro-recall (%)')
ax.set_title('(b) inference vs long-tail macroR')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / 'pareto.pdf'); plt.savefig(OUT / 'pareto.png')
plt.close()


# ====================== Fig 3: bar chart ===============================
fig, ax = plt.subplots(figsize=(7, 3.6))
names = ['CNN-NoPool', 'GLA', 'RetNet']
sg = [100 * m['best_sg_top1'] if m else 0 for m in (cnn_m, gla_m, ret_m)]
mr = [100 * m['sg_macro_recall'] if m else 0 for m in (cnn_m, gla_m, ret_m)]
x = np.arange(len(names))
w = 0.38
b1 = ax.bar(x - w / 2, sg, w, color='C0', label='SG top-1 (%)', edgecolor='k', lw=0.4)
b2 = ax.bar(x + w / 2, mr, w, color='C3', label='macro-recall (%)', edgecolor='k', lw=0.4)
for i, v in enumerate(sg):
    ax.text(x[i] - w / 2, v + 0.5, f'{v:.1f}', ha='center', fontsize=8)
for i, v in enumerate(mr):
    ax.text(x[i] + w / 2, v + 0.5, f'{v:.1f}', ha='center', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(names)
ax.set_ylabel('Validation (%)')
ax.set_title('CCDC-SG measured: 40-epoch best (class-weighted CE)')
ax.legend(); ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 50)
plt.tight_layout()
plt.savefig(OUT / 'barchart.pdf'); plt.savefig(OUT / 'barchart.png')
plt.close()


# ====================== Fig 4: status table ============================
fig, ax = plt.subplots(figsize=(8, 2.6))
ax.axis('off')
rows = [['Backbone', 'best SG top-1', 'macroR', '#P (M)', 'train s/ep', 'infer ms/sample']]
for name, m in [('CNN-NoPool', cnn_m), ('GLA', gla_m), ('RetNet', ret_m)]:
    if m is None:
        rows.append([name, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'])
    else:
        rows.append([
            name,
            f"{100*m['best_sg_top1']:.2f}%",
            f"{100*m['sg_macro_recall']:.2f}%",
            f"{m['params_M']:.2f}",
            f"{m['train_sec_per_epoch']:.0f}",
            f"{m['infer_ms_per_sample']:.2f}",
        ])
rows.append(['Mamba-2', 'not run', '—', '—', '—', '—'])
tbl = ax.table(cellText=rows, loc='center', cellLoc='center', colWidths=[0.18] * 6)
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.0, 1.5)
for j in range(6):
    tbl[(0, j)].set_facecolor('#dddddd'); tbl[(0, j)].set_text_props(weight='bold')
for i, r in enumerate(rows[1:], 1):
    c = '#d9efdc' if 'N/A' not in r[1] and 'not run' not in r[1] else '#f0f0f0'
    for j in range(6):
        tbl[(i, j)].set_facecolor(c)
ax.set_title('LineXRD matrix v4 — CCDC-SG, 40 epochs, single seed', pad=12)
plt.tight_layout()
plt.savefig(OUT / 'status.pdf'); plt.savefig(OUT / 'status.png')
plt.close()


print('Wrote figs to', OUT)
for f in sorted(OUT.iterdir()):
    print(' ', f.name, f.stat().st_size, 'bytes')

# Echo back the headline numbers
print('\n=== Measured numbers ===')
for name, m in [('CNN-NoPool', cnn_m), ('GLA', gla_m), ('RetNet', ret_m)]:
    if not m:
        continue
    print(f"  {name:11s} sg_top1={100*m['best_sg_top1']:.2f}% "
          f"macroR={100*m['sg_macro_recall']:.2f}% "
          f"params={m['params_M']:.2f}M "
          f"s/ep={m['train_sec_per_epoch']:.0f} "
          f"ms/sample={m['infer_ms_per_sample']:.2f}")
