"""Fig 8 — empirical PXRD power spectrum across the CCDC-SG training set.

No checkpoints required; uses only the raw NPY data. Produces
paper/figures/fig8_pxrd_spectrum.{pdf,png}.
"""
from __future__ import annotations
import pathlib
import numpy as np
import matplotlib.pyplot as plt

ROOT = pathlib.Path('/opt/data/private/linexrd')
DATA = pathlib.Path('/opt/data/private/xrd2c_data/ccdc_sg_train/train_ccdc_sg.npy')
OUT = ROOT / 'paper' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'legend.fontsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'figure.dpi': 150, 'savefig.dpi': 600, 'pdf.fonttype': 42,
})

d = np.load(DATA, allow_pickle=True).item()
X = d['features']                 # (N, 8500) float32
print('loaded', X.shape, X.dtype)

# Sub-sample to ~5000 patterns for the spectrum (full 108K is overkill).
rng = np.random.default_rng(0)
idx = rng.choice(X.shape[0], size=min(5000, X.shape[0]), replace=False)
Xs = X[idx]
L = Xs.shape[1]

# Compute mean |FFT|^2
F = np.fft.rfft(Xs, axis=1)
P = (np.abs(F) ** 2).mean(axis=0)
P = P / P.sum()
omega = np.linspace(0, np.pi, P.shape[0])

# Smooth lightly for plotting clarity
win = 11
kern = np.ones(win) / win
P_smooth = np.convolve(P, kern, mode='same')

# Three-band annotation
band_dc = (omega < 0.02)
band_plateau = (omega >= 0.02) & (omega < 0.4)
band_tail = (omega >= 0.4)
e_dc = P[band_dc].sum()
e_plateau = P[band_plateau].sum()
e_tail = P[band_tail].sum()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.2))

ax1.semilogy(omega / np.pi, P_smooth, color='C0', lw=1.0)
ax1.fill_between(omega / np.pi, 1e-12, P_smooth,
                 where=band_dc, color='C3', alpha=0.18, label=f'DC band ({100*e_dc:.0f}%)')
ax1.fill_between(omega / np.pi, 1e-12, P_smooth,
                 where=band_plateau, color='C2', alpha=0.18, label=f'Bragg plateau ({100*e_plateau:.0f}%)')
ax1.fill_between(omega / np.pi, 1e-12, P_smooth,
                 where=band_tail, color='C4', alpha=0.18, label=f'noise tail ({100*e_tail:.0f}%)')
ax1.set_xlabel(r'normalised frequency $\omega/\pi$')
ax1.set_ylabel(r'$\hat{S}(\omega)$ (normalised)')
ax1.set_title('Empirical PXRD power spectrum')
ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
ax1.set_xlim(0, 1); ax1.set_ylim(P_smooth.min() * 0.5, P_smooth.max() * 2)

# Right panel: cumulative energy
cdf = np.cumsum(P)
ax2.plot(omega / np.pi, cdf, color='C0', lw=1.3)
ax2.axhline(0.5, color='gray', ls='--', alpha=0.5)
ax2.axhline(0.9, color='gray', ls='--', alpha=0.5)
# find cross point at 0.5 and 0.9
i50 = np.searchsorted(cdf, 0.5)
i90 = np.searchsorted(cdf, 0.9)
ax2.axvline(omega[i50] / np.pi, color='C2', alpha=0.6)
ax2.axvline(omega[i90] / np.pi, color='C4', alpha=0.6)
ax2.annotate(f'50% energy at\n$\\omega/\\pi={omega[i50]/np.pi:.3f}$',
             xy=(omega[i50] / np.pi, 0.5), xytext=(0.3, 0.35),
             arrowprops=dict(arrowstyle='->', alpha=0.6), fontsize=8)
ax2.annotate(f'90% energy at\n$\\omega/\\pi={omega[i90]/np.pi:.3f}$',
             xy=(omega[i90] / np.pi, 0.9), xytext=(0.5, 0.7),
             arrowprops=dict(arrowstyle='->', alpha=0.6), fontsize=8)
ax2.set_xlabel(r'normalised frequency $\omega/\pi$')
ax2.set_ylabel('cumulative normalised energy')
ax2.set_title('Cumulative power')
ax2.set_xlim(0, 1); ax2.set_ylim(0, 1.02); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / 'fig8_pxrd_spectrum.pdf')
plt.savefig(OUT / 'fig8_pxrd_spectrum.png')
plt.close()
print(f'Wrote {OUT}/fig8_pxrd_spectrum.{{pdf,png}}')
print(f'energies — DC={e_dc:.4f}, plateau={e_plateau:.4f}, tail={e_tail:.4f}')
print(f'50% energy at omega/pi = {omega[i50]/np.pi:.4f}, 90% at {omega[i90]/np.pi:.4f}')
