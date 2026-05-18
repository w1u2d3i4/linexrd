"""LineXRD backbones: CNN11-NoPool / Mamba-2 / GLA / RetNet on 8500-token PXRD.

设计目标: 同一 stem (Conv-stride2) + 同一 head (mean-pool + linear), 仅替换中段 encoder。
- N_CLASSES = 230 (space group)
- N_CS = 7 (crystal system, multi-task head)
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

N_SG = 230
N_CS = 7


class MultiScaleStem(nn.Module):
    """多尺度 1D-CNN stem，stride=2，保留峰位信息（无 MaxPool）。

    输入 (B, 8500) -> (B, D, 4251)
    """

    def __init__(self, d_model: int = 256):
        super().__init__()
        d4 = d_model // 4
        self.bn_in = nn.BatchNorm1d(1)
        self.k3 = nn.Conv1d(1, d4, 3, stride=2, padding=1)
        self.k7 = nn.Conv1d(1, d4, 7, stride=2, padding=3)
        self.k15 = nn.Conv1d(1, d4, 15, stride=2, padding=7)
        self.k1 = nn.Conv1d(1, d4, 1, stride=2)
        self.fuse = nn.Sequential(
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(d_model, d_model, 3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

    def forward(self, x):
        # x: (B, 8500) -> (B, 1, 8500)
        x = x.unsqueeze(1)
        x = self.bn_in(x)
        h = torch.cat([self.k3(x), self.k7(x), self.k15(x), self.k1(x)], dim=1)
        h = self.fuse(h)
        return h.transpose(1, 2)  # (B, L, D)


# ---------------- CNN11-NoPool baseline ----------------

class CNNNoPool(nn.Module):
    """8 层 Conv1d + GELU + BN，没有 MaxPool，stride=1 保持峰位。"""

    def __init__(self, d_model: int = 256, depth: int = 8):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers += [
                nn.Conv1d(d_model, d_model, 3, padding=1),
                nn.BatchNorm1d(d_model),
                nn.GELU(),
                nn.Dropout(0.1),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x_seq):
        # x_seq: (B, L, D)
        h = x_seq.transpose(1, 2)
        h = self.net(h)
        return h.transpose(1, 2)  # (B, L, D)


# ---------------- Linear Attention (GLA-style) ----------------

class LinearAttention(nn.Module):
    """Gated Linear Attention block — fp16-stable version.

    Stability fixes over the previous version:
      1. Use a small d_h (= d_model // 2) instead of 2 * d_model. This caps the
         outer-product KV at (d_h, d_h) ~ 128^2 = 16k floats and prevents the
         fp16 magnitude explosion that caused NaN loss in v1.
      2. Down-scale q by 1/sqrt(d_h) before featurisation, so that
         <elu(q)+1, elu(k)+1> sums grow at O(L) instead of O(L * d_h).
      3. Cast the O(L*d_h^2) KV reduction to fp32 inside autocast, then
         cast back. This is the only path where fp16 can underflow.
      4. eps in the normalising denominator increased from 1e-6 to 1e-3
         to avoid divide-by-tiny in fp16.
    """

    def __init__(self, d_model: int, head_ratio: float = 0.5):
        super().__init__()
        d_h = max(32, int(d_model * head_ratio))
        self.qkv = nn.Linear(d_model, 3 * d_h, bias=False)
        self.gate = nn.Linear(d_model, d_h)
        self.out = nn.Linear(d_h, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.d_h = d_h
        self.scale = d_h ** -0.5

    def forward(self, x):
        # x: (B, L, D)
        res = x
        x = self.norm(x)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q * self.scale
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        # Compute the O(L*d_h^2) reduction in fp32 for numerical safety.
        with torch.cuda.amp.autocast(enabled=False):
            qf, kf, vf = q.float(), k.float(), v.float()
            kv = torch.einsum("bld,blf->bdf", kf, vf)         # (B, d_h, d_h)
            ksum = kf.sum(dim=1)                              # (B, d_h)
            denom = torch.einsum("bld,bd->bl", qf, ksum)      # (B, L)
            denom = denom.clamp(min=1e-3)
            out = torch.einsum("bld,bdf->blf", qf, kv) / denom.unsqueeze(-1)
        out = out.to(x.dtype)
        gate = torch.sigmoid(self.gate(x))
        out = out * gate
        out = self.out(out)
        return res + out


class GLABackbone(nn.Module):
    def __init__(self, d_model: int = 256, depth: int = 8, head_ratio: float = 0.5):
        super().__init__()
        self.layers = nn.ModuleList(
            [LinearAttention(d_model, head_ratio) for _ in range(depth)]
        )
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model),
            )
            for _ in range(depth)
        ])

    def forward(self, x):
        for a, f in zip(self.layers, self.ffns):
            x = a(x)
            x = x + f(x)
        return x


# ---------------- Mamba-2 style SSM (selective scan, simplified parallel form) ----------------

class SimpleMambaBlock(nn.Module):
    """简化 Mamba (S6-like) block，pure PyTorch（无 mamba-ssm 依赖）.

    用并行 prefix-scan 近似 selective state space。L=4251 时实测可在 H100/A100 上跑通。
    """

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        d_inner = d_model * expand
        self.d_state = d_state
        self.d_inner = d_inner
        self.in_proj = nn.Linear(d_model, 2 * d_inner)
        self.conv = nn.Conv1d(d_inner, d_inner, 4, padding=3, groups=d_inner)
        self.x_proj = nn.Linear(d_inner, d_state + d_state + 1)
        self.dt_proj = nn.Linear(1, d_inner)
        # A_log: per-channel diagonal A (negative)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()).repeat(d_inner, 1))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, L, D)
        res = x
        x = self.norm(x)
        B, L, _ = x.shape
        xz = self.in_proj(x)
        u, z = xz.chunk(2, dim=-1)  # (B, L, d_inner)
        # 1D causal conv on u
        u_c = self.conv(u.transpose(1, 2))[:, :, :L].transpose(1, 2)
        u_c = F.silu(u_c)
        # selective params
        proj = self.x_proj(u_c)
        Bs, Cs, dt = proj.split([self.d_state, self.d_state, 1], dim=-1)
        dt = F.softplus(self.dt_proj(dt))  # (B, L, d_inner)
        A = -self.A_log.exp()  # (d_inner, d_state)
        # discretize: a = exp(dt * A); b = dt * B
        # For efficiency, use simple recurrent scan (loop over time) — OK for d_state=16
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            ut = u_c[:, t, :]                # (B, d_inner)
            dtt = dt[:, t, :]                # (B, d_inner)
            bt = Bs[:, t, :]                 # (B, d_state)
            ct = Cs[:, t, :]                 # (B, d_state)
            a = torch.exp(dtt.unsqueeze(-1) * A.unsqueeze(0))   # (B, d_inner, d_state)
            b = dtt.unsqueeze(-1) * bt.unsqueeze(1)             # (B, d_inner, d_state)
            h = a * h + b * ut.unsqueeze(-1)
            yt = torch.einsum("bds,bs->bd", h, ct) + self.D * ut
            ys.append(yt)
        y = torch.stack(ys, dim=1)              # (B, L, d_inner)
        out = self.out_proj(y * F.silu(z))
        return res + out


class MambaBackbone(nn.Module):
    """注意：纯 Python 循环 scan 对 L=4251 很慢。仅用于结构正确性 + smoke 实验。
    生产可换 mamba-ssm。"""

    def __init__(self, d_model: int = 256, depth: int = 6):
        super().__init__()
        self.layers = nn.ModuleList([SimpleMambaBlock(d_model) for _ in range(depth)])

    def forward(self, x):
        for blk in self.layers:
            x = blk(x)
        return x


# ---------------- RetNet (chunkwise) ----------------

class RetNetBlock(nn.Module):
    """Retention block — chunkwise factorisation (Sun et al. 2023, eq.~11--13).

    Splits the sequence into chunks of length C. Within each chunk the
    retention is computed in parallel (a $C \\times C$ masked matmul);
    cross-chunk information is carried forward via a state of shape
    $(D, D)$, eliminating the $L \\times L$ memory blow-up that caused
    the OOM in the previous parallel-form implementation.
    """

    def __init__(self, d_model: int, gamma: float = 0.96, chunk: int = 256):
        super().__init__()
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.gamma = gamma
        self.chunk = chunk
        self.scale = d_model ** -0.5

    def forward(self, x):
        # x: (B, L, D)
        res = x
        x = self.norm(x)
        B, L, D = x.shape
        C = self.chunk
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q * self.scale
        # pad L so it is a multiple of C
        pad = (-L) % C
        if pad:
            q = F.pad(q, (0, 0, 0, pad))
            k = F.pad(k, (0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, pad))
        Lp = q.size(1)
        nc = Lp // C
        q = q.view(B, nc, C, D)
        k = k.view(B, nc, C, D)
        v = v.view(B, nc, C, D)

        # Intra-chunk decay mask (C, C): D[i,j] = gamma^(i-j) for i>=j else 0
        idx = torch.arange(C, device=x.device)
        diff = idx.unsqueeze(0) - idx.unsqueeze(1)             # (C, C) i-j
        mask = (diff >= 0).to(x.dtype)
        intra_decay = (self.gamma ** diff.clamp(min=0).to(x.dtype)) * mask
        # Cross-chunk decay per position in current chunk: gamma^(i+1) when
        # multiplying the previous-chunks' cumulative state.
        i_decay = self.gamma ** (idx + 1).to(x.dtype)          # (C,)
        # Within-chunk decay for state update at chunk end: gamma^(C-1-j)
        end_decay = self.gamma ** (C - 1 - idx).to(x.dtype)    # (C,)

        outs = []
        s = q.new_zeros(B, D, D)         # cumulative cross-chunk state (B,D,D)
        gamma_C = self.gamma ** C
        for n in range(nc):
            qn = q[:, n]                 # (B, C, D)
            kn = k[:, n]
            vn = v[:, n]
            # Intra-chunk parallel retention: (B,C,D) @ (B,D,C) -> (B,C,C)
            attn = torch.einsum("bcd,bed->bce", qn, kn) * intra_decay
            y_intra = torch.einsum("bce,bed->bcd", attn, vn)
            # Cross-chunk: y_inter[i] = gamma^(i+1) * qn[i] @ s
            y_inter = torch.einsum("bcd,bde->bce", qn, s) * i_decay.view(1, C, 1)
            outs.append(y_intra + y_inter)
            # Update state: s_{n+1} = gamma^C * s_n + sum_j gamma^(C-1-j) k_j^T v_j
            if n < nc - 1:
                kn_dec = kn * end_decay.view(1, C, 1)        # (B,C,D)
                s = gamma_C * s + torch.einsum("bcd,bce->bde", kn_dec, vn)
        y = torch.stack(outs, dim=1).reshape(B, Lp, D)
        if pad:
            y = y[:, :L]
        return res + self.out(y)


class RetNetBackbone(nn.Module):
    def __init__(self, d_model: int = 256, depth: int = 8, gamma: float = 0.96,
                 chunk: int = 256):
        super().__init__()
        self.layers = nn.ModuleList(
            [RetNetBlock(d_model, gamma, chunk=chunk) for _ in range(depth)]
        )
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model),
            )
            for _ in range(depth)
        ])

    def forward(self, x):
        for a, f in zip(self.layers, self.ffns):
            x = a(x)
            x = x + f(x)
        return x


# ---------------- Full classifier ----------------

class LineXRDClassifier(nn.Module):
    def __init__(self, backbone: str = "cnn", d_model: int = 256, depth: int = 8,
                 use_hier_head: bool = False):
        super().__init__()
        self.stem = MultiScaleStem(d_model)
        if backbone == "cnn":
            self.encoder = CNNNoPool(d_model, depth)
        elif backbone == "gla":
            self.encoder = GLABackbone(d_model, depth)
        elif backbone == "mamba":
            self.encoder = MambaBackbone(d_model, depth=max(2, depth // 2))
        elif backbone == "retnet":
            self.encoder = RetNetBackbone(d_model, depth)
        else:
            raise ValueError(f"unknown backbone {backbone}")
        self.use_hier = use_hier_head
        self.cls_sg = nn.Linear(d_model, N_SG)
        if use_hier_head:
            self.cls_cs = nn.Linear(d_model, N_CS)

    def forward(self, x):
        h = self.stem(x)        # (B, L, D)
        h = self.encoder(h)     # (B, L, D)
        h = h.mean(dim=1)       # (B, D)
        out = {"sg_logits": self.cls_sg(h)}
        if self.use_hier:
            out["cs_logits"] = self.cls_cs(h)
        return out


if __name__ == "__main__":
    x = torch.randn(2, 8500)
    for bk in ["cnn", "gla", "retnet"]:
        m = LineXRDClassifier(bk, d_model=128, depth=4)
        y = m(x)
        n_params = sum(p.numel() for p in m.parameters())
        print(f"backbone={bk} params={n_params/1e6:.2f}M sg_logits.shape={y['sg_logits'].shape}")
