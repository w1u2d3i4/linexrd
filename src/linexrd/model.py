"""Peak-set encoder, using the original selection, features, and attention blocks."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

N_SG = 230
N_CS = 7
L_INPUT = 8500


class SinusoidalPosEnc(nn.Module):
    """Continuous sinusoidal positional encoding for a scalar position."""

    def __init__(self, n_freq: int = 32, max_period: float = 8500.0):
        super().__init__()
        self.n_freq = n_freq
        freqs = torch.exp(torch.linspace(0.0, math.log(max_period), n_freq))
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, p):
        # p: (..., 1) -- normalised to whatever scale; we feed pos in [0,1]
        arg = p * self.freqs * (2.0 * math.pi)
        return torch.cat([arg.sin(), arg.cos()], dim=-1)         # (..., 2*n_freq)


class PSTBlock(nn.Module):
    """Pre-LN Transformer block with FlashAttention via SDPA."""

    def __init__(self, d_model: int, n_heads: int = 8, ffn_mult: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

    def _attn(self, x):
        B, K, D = x.shape
        qkv = self.qkv(x).view(B, K, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, K, D)
        return self.out(y)

    def forward(self, x):
        x = x + self.dropout(self._attn(self.norm1(x)))
        x = x + self.ffn(self.norm2(x))
        return x


class PeakSetTransformer(nn.Module):
    def __init__(self, d_model: int = 256, depth: int = 6, n_heads: int = 8,
                 K: int = 256, n_freq: int = 32, use_hier_head: bool = False,
                 dropout: float = 0.1, multi_scale: bool = False,
                 bg_subtract: bool = True):
        super().__init__()
        self.K = K
        self.d_model = d_model
        self.n_freq = n_freq
        self.use_hier = use_hier_head
        self.multi_scale = multi_scale
        self.bg_subtract = bg_subtract
        self._L = L_INPUT
        # multi-scale window radii (only used when multi_scale=True)
        self._ms_radii = (9, 27) if multi_scale else ()

        # background-subtraction basis filled lazily (device/dtype-aware)
        self.register_buffer("_bg_X", torch.empty(0), persistent=False)
        self.register_buffer("_bg_pinv", torch.empty(0), persistent=False)

        # peak feature dim: pos + raw + residual + lmean + lstd + fwhm_proxy + sin-PE
        # MS adds: for each extra radius (mean_x, std_x, mean_abs_r, max_abs_r) = 4
        # plus prominence (raw / |lmean|+eps) = 1, sign(res) = 1
        ms_extra = 4 * len(self._ms_radii) + 2 if multi_scale else 0
        self._feat_dim = 6 + 2 * n_freq + ms_extra
        self.peak_pos = SinusoidalPosEnc(n_freq=n_freq, max_period=float(L_INPUT))
        self.feat_proj = nn.Sequential(
            nn.LayerNorm(self._feat_dim),
            nn.Linear(self._feat_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.blocks = nn.ModuleList([
            PSTBlock(d_model, n_heads=n_heads, dropout=dropout)
            for _ in range(depth)
        ])
        self.head_norm = nn.LayerNorm(d_model)
        self.cls_sg = nn.Linear(d_model, N_SG)
        if use_hier_head:
            self.cls_cs = nn.Linear(d_model, N_CS)

    def _maybe_build_bg(self, device):
        # always store basis in fp32 -- it is tiny and used in fp32 BG subtract
        if (self._bg_pinv.numel() == 0 or self._bg_pinv.device != device):
            with torch.cuda.amp.autocast(enabled=False):
                t = torch.linspace(-1.0, 1.0, self._L, device=device, dtype=torch.float32)
                X = torch.stack([torch.ones_like(t), t, t * t, t ** 3], dim=1)  # (L, 4)
                XtX = X.T @ X
                pinv = torch.linalg.solve(XtX, X.T)                              # (4, L)
            self._bg_X = X
            self._bg_pinv = pinv

    def _bg_subtract(self, x):
        # Run BG subtract in fp32 regardless of outer autocast for stability.
        self._maybe_build_bg(x.device)
        with torch.cuda.amp.autocast(enabled=False):
            xf = x.float()
            coef = xf @ self._bg_pinv.T                                        # (B, 4)
            bg = coef @ self._bg_X.T                                           # (B, L)
            r = xf - bg
        return r.to(x.dtype), bg.to(x.dtype)

    @torch.no_grad()
    def _select_peaks(self, r):
        """Return (B, K) integer indices, sorted ascending by position."""
        a = r.abs()
        pooled = F.max_pool1d(a.unsqueeze(1), kernel_size=5,
                              stride=1, padding=2).squeeze(1)
        is_local_max = (a >= pooled - 1e-8).float()
        score = a * (0.5 + 0.5 * is_local_max)
        idx = score.topk(self.K, dim=1).indices                                # (B, K)
        idx = idx.sort(dim=1).values
        return idx

    def _gather_features(self, x, r, idx):
        B, L = x.shape
        K = idx.size(1)
        pad = 2
        xp = F.pad(x, (pad, pad), mode='replicate')
        rp = F.pad(r, (pad, pad), mode='replicate')
        idx_p = idx + pad
        offsets = torch.arange(-pad, pad + 1, device=x.device)
        win_idx = idx_p.unsqueeze(-1) + offsets                                # (B, K, 5)
        win_flat = win_idx.reshape(B, -1)
        x_win = torch.gather(xp, 1, win_flat).reshape(B, K, 2 * pad + 1)
        r_win = torch.gather(rp, 1, win_flat).reshape(B, K, 2 * pad + 1)

        raw = x_win[:, :, pad]
        res = r_win[:, :, pad]
        local_mean = x_win.mean(dim=-1)
        local_std = x_win.std(dim=-1)
        half = (res.abs() * 0.5).unsqueeze(-1)
        fwhm_proxy = (r_win.abs() >= half).float().mean(dim=-1)

        pos_norm = idx.float() / float(L)
        pos_pe = self.peak_pos(pos_norm.unsqueeze(-1))                         # (B, K, 2F)

        parts = [
            pos_norm.unsqueeze(-1),
            raw.unsqueeze(-1),
            res.unsqueeze(-1),
            local_mean.unsqueeze(-1),
            local_std.unsqueeze(-1),
            fwhm_proxy.unsqueeze(-1),
            pos_pe,
        ]

        if self.multi_scale:
            # Gather wider windows around each peak; report mean(raw), std(raw),
            # mean(|residual|), max(|residual|) at each scale.
            for radius in self._ms_radii:
                xp_r = F.pad(x, (radius, radius), mode='replicate')
                rp_r = F.pad(r, (radius, radius), mode='replicate')
                idxr = idx + radius
                offs = torch.arange(-radius, radius + 1, device=x.device)
                win = (idxr.unsqueeze(-1) + offs).reshape(B, -1)
                xw = torch.gather(xp_r, 1, win).reshape(B, K, 2 * radius + 1)
                rw = torch.gather(rp_r, 1, win).reshape(B, K, 2 * radius + 1).abs()
                parts.extend([
                    xw.mean(dim=-1).unsqueeze(-1),
                    xw.std(dim=-1).unsqueeze(-1),
                    rw.mean(dim=-1).unsqueeze(-1),
                    rw.max(dim=-1).values.unsqueeze(-1),
                ])
            prominence = raw / (local_mean.abs() + 1e-4)
            parts.append(prominence.unsqueeze(-1))
            parts.append(torch.sign(res).unsqueeze(-1))

        feats = torch.cat(parts, dim=-1)                                        # (B, K, F)
        return feats

    def encode(self, x):
        """Return the normalized CLS representation before classification."""
        if x.ndim != 2 or x.shape[1] != L_INPUT:
            raise ValueError("Expected (N, 8500) PXRD intensities")
        if not 1 <= self.K <= L_INPUT:
            raise ValueError("K must be in [1, 8500]")
        if self.bg_subtract:
            r, _ = self._bg_subtract(x)
        else:
            r = x  # use raw spectrum as the "residual" channel
        idx = self._select_peaks(r)
        feats = self._gather_features(x, r, idx)
        h = self.feat_proj(feats)                                              # (B, K, D)

        cls = self.cls.expand(h.size(0), 1, -1)
        h = torch.cat([cls, h], dim=1)                                         # (B, 1+K, D)
        for blk in self.blocks:
            h = blk(h)

        return self.head_norm(h[:, 0])

    def forward(self, x):
        cls_h = self.encode(x)
        out = {"sg_logits": self.cls_sg(cls_h)}
        if self.use_hier:
            out["cs_logits"] = self.cls_cs(cls_h)
        return out
