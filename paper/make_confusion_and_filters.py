"""F6 confusion matrix + F7 learned filter spectra + CS-population table.

Loads v4 best.pt from /opt/data/private/linexrd/checkpoints/ and produces:
  - paper/figures/fig6_confusion.{pdf,png}
  - paper/figures/fig7_filter_spectra.{pdf,png}
  - paper/figures/cs_population.txt (numbers to splice into Table 1)

Runs on GPU 0 to avoid interfering with the ablation matrix on GPU 1.
"""
from __future__ import annotations
import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT = pathlib.Path('/opt/data/private/linexrd')
CKPT = ROOT / 'checkpoints'
sys.path.insert(0, str(ROOT / 'src'))
from data.loader import PXRDDataset, get_sg_to_cs, N_CLASSES   # noqa: E402
from models.backbones import LineXRDClassifier   # noqa: E402

OUT = ROOT / 'paper' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
DEV = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'figure.dpi': 150, 'savefig.dpi': 600, 'pdf.fonttype': 42,
})

BACKBONES = {
    'CNN-NoPool': 'cnn_ccdc_sg_d256_dp6_ep40_s0',
    'GLA':        'gla_ccdc_sg_d256_dp6_ep40_s0',
    'RetNet':     'retnet_ccdc_sg_d256_dp6_ep40_s0',
}


def load_model(name: str, run: str) -> tuple[LineXRDClassifier, dict]:
    pt = CKPT / run / 'best.pt'
    blob = torch.load(pt, map_location='cpu', weights_only=False)
    args = blob['args']
    m = LineXRDClassifier(args['backbone'], args['d_model'], args['depth'],
                          args.get('use_hier_head', False))
    m.load_state_dict(blob['state_dict'])
    m.to(DEV).eval()
    return m, blob


@torch.no_grad()
def evaluate_confusion(m: LineXRDClassifier, ds: PXRDDataset, name: str):
    print(f'  evaluating {name} ...', flush=True)
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    cs_cm = np.zeros((7, 7), dtype=np.int64)
    sg_to_cs = get_sg_to_cs()
    dl = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False, num_workers=2)
    total, correct = 0, 0
    for x, ysg, ycs in dl:
        x = x.to(DEV, non_blocking=True)
        ysg = ysg.numpy(); ycs = ycs.numpy()
        with torch.amp.autocast('cuda'):
            out = m(x)
        pred = out['sg_logits'].argmax(dim=1).cpu().numpy()
        # 0-indexed; map to crystal system via sg_to_cs (which is 1-indexed input)
        pred_cs = np.array([sg_to_cs.get(int(p) + 1, 0) for p in pred])
        for t, p in zip(ysg, pred):
            cm[t, p] += 1
        for t, p in zip(ycs, pred_cs):
            cs_cm[t, p] += 1
        total += len(pred); correct += (pred == ysg).sum()
    return cm, cs_cm, correct / total


# ---------- Fig 6: confusion matrix heatmap (one per backbone, log scale) ---
val_ds = PXRDDataset('ccdc_sg', 'val')
print(f'val set: {len(val_ds)} patterns')

fig, axes = plt.subplots(2, 3, figsize=(10, 6.6))
ortho_idxs = [62 - 1, 33 - 1, 17 - 1]   # SG ids (0-indexed): Pnma=62, Pna2_1=33, Pmc2_1=26... we'll just pick orthorhombic block
# A simpler choice: pick the 3 most-confused SGs from the CNN result post-hoc.

for col, (name, run) in enumerate(BACKBONES.items()):
    m, _ = load_model(name, run)
    cm, cs_cm, acc = evaluate_confusion(m, val_ds, name)
    # Top row: full 230x230 confusion (log scale)
    ax = axes[0, col]
    im = ax.imshow(cm + 1, cmap='Greys', norm=LogNorm())
    ax.set_title(f'{name}\nSG top-1 = {100*acc:.2f}%')
    ax.set_xlabel('Predicted SG')
    ax.set_ylabel('True SG' if col == 0 else '')
    plt.colorbar(im, ax=ax, shrink=0.7)
    # Bottom row: 7x7 crystal-system confusion (linear)
    ax = axes[1, col]
    cs_row = cs_cm.sum(axis=1, keepdims=True).clip(1)
    im = ax.imshow(cs_cm / cs_row, cmap='Blues', vmin=0, vmax=1)
    ax.set_title(f'{name} — CS (row-normalised)')
    ax.set_xticks(range(7)); ax.set_yticks(range(7))
    ax.set_xticklabels(['Tri', 'Mon', 'Ort', 'Tet', 'Trg', 'Hex', 'Cub'])
    ax.set_yticklabels(['Tri', 'Mon', 'Ort', 'Tet', 'Trg', 'Hex', 'Cub'])
    for i in range(7):
        for j in range(7):
            v = cs_cm[i, j] / max(1, cs_row[i, 0])
            if v > 0.05:
                ax.text(j, i, f'{100*v:.0f}', ha='center', va='center',
                        fontsize=6, color='white' if v > 0.5 else 'black')
    plt.colorbar(im, ax=ax, shrink=0.7)

plt.suptitle('Fig. 6 — confusion on CCDC-SG validation', y=1.01, fontsize=11)
plt.tight_layout()
plt.savefig(OUT / 'fig6_confusion.pdf')
plt.savefig(OUT / 'fig6_confusion.png')
plt.close()
print('wrote', OUT / 'fig6_confusion.pdf')


# ---------- Fig 7: learned stem filter spectra ----------------------------
print('extracting stem kernels ...')
fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.0))
for col, (name, run) in enumerate(BACKBONES.items()):
    m, _ = load_model(name, run)
    stem = m.stem
    branches = {'k=1': stem.k1, 'k=3': stem.k3, 'k=7': stem.k7, 'k=15': stem.k15}
    ax = axes[col]
    for label, conv in branches.items():
        w = conv.weight.detach().cpu().squeeze(1).numpy()   # (out_channels, k)
        # average squared magnitude across output channels
        # FFT zero-pad to 256 for smooth display
        W = np.fft.rfft(w, n=256, axis=1)
        P = (np.abs(W) ** 2).mean(axis=0)
        P = P / P.max()
        omega = np.linspace(0, np.pi, P.shape[0])
        ax.plot(omega / np.pi, P, lw=1.3, label=label)
    ax.set_xlabel(r'normalised frequency $\omega/\pi$')
    ax.set_ylabel('|G(omega)| (normalised)' if col == 0 else '')
    ax.set_title(name)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(alpha=0.3); ax.set_xlim(0, 1)
plt.suptitle('Fig. 7 — learned stem filter spectra', y=1.04, fontsize=11)
plt.tight_layout()
plt.savefig(OUT / 'fig7_filter_spectra.pdf')
plt.savefig(OUT / 'fig7_filter_spectra.png')
plt.close()
print('wrote', OUT / 'fig7_filter_spectra.pdf')


# ---------- crystal-system population table -------------------------------
labels = np.load('/opt/data/private/xrd2c_data/ccdc_sg_train/train_ccdc_sg.npy',
                 allow_pickle=True).item()['labels230']
if labels.min() == 0:
    labels = labels + 1
sg_to_cs = get_sg_to_cs()
cs_counts = np.zeros(7, dtype=np.int64)
n_per_cs = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
# also count #SGs per CS (universal map, not data-dependent):
sgs_per_cs = {c: set() for c in range(7)}
for sg, cs in sg_to_cs.items():
    sgs_per_cs[cs].add(sg)
for sg in labels:
    cs = sg_to_cs.get(int(sg), 0)
    n_per_cs[cs] += 1
print('\nCS population on CCDC-SG train:')
names = ['Tri', 'Mon', 'Ort', 'Tet', 'Trg', 'Hex', 'Cub']
total = sum(n_per_cs.values())
out_lines = ['# crystal-system population on CCDC-SG train']
for c in range(7):
    pct = 100 * n_per_cs[c] / total
    out_lines.append(f'CS={c} ({names[c]}): n_sg={len(sgs_per_cs[c]):2d}  samples={n_per_cs[c]:6d}  ({pct:5.2f}%)')
    print(out_lines[-1])
(OUT / 'cs_population.txt').write_text('\n'.join(out_lines) + '\n')
print('wrote', OUT / 'cs_population.txt')
