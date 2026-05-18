"""LineXRD 训练入口 — 4 种 backbone (cnn / gla / mamba / retnet) 在全分辨率 PXRD 上空间群分类.

用法:
  python -m src.train_linexrd \
      --backbone gla --dataset ccdc_sg --epochs 12 \
      --batch_size 128 --lr 1e-3 --d_model 256 --depth 8 \
      --save_dir checkpoints/gla_ccdc_smoke

输出: 训练日志 + METRIC name=value 行（供 autoresearch.sh 解析）
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from data.loader import make_loaders, N_CLASSES
from models.backbones import LineXRDClassifier


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int | None = None):
    model.eval()
    sg_correct, cs_correct, total = 0, 0, 0
    per_class_correct = torch.zeros(N_CLASSES, dtype=torch.long)
    per_class_total = torch.zeros(N_CLASSES, dtype=torch.long)
    t0 = time.time()
    for i, (x, y_sg, y_cs) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y_sg = y_sg.to(device, non_blocking=True)
        y_cs = y_cs.to(device, non_blocking=True)
        with autocast():
            out = model(x)
        pred = out["sg_logits"].argmax(dim=1)
        sg_correct += (pred == y_sg).sum().item()
        if "cs_logits" in out:
            cs_pred = out["cs_logits"].argmax(dim=1)
            cs_correct += (cs_pred == y_cs).sum().item()
        total += x.size(0)
        # per-class for macro F1
        for c in range(N_CLASSES):
            mask = (y_sg == c)
            if mask.any():
                per_class_total[c] += int(mask.sum().item())
                per_class_correct[c] += int(((pred == y_sg) & mask).sum().item())
    elapsed = time.time() - t0
    sg_top1 = sg_correct / max(1, total)
    cs_top1 = cs_correct / max(1, total)
    # macro-F1 proxy: macro recall (per-class accuracy averaged over classes with samples)
    seen = per_class_total > 0
    if seen.any():
        macro_recall = (per_class_correct[seen].float() / per_class_total[seen].float()).mean().item()
    else:
        macro_recall = 0.0
    ms_per_sample = 1000.0 * elapsed / max(1, total)
    return {
        "sg_top1": sg_top1,
        "cs_top1": cs_top1,
        "sg_macro_recall": macro_recall,
        "infer_ms_per_sample": ms_per_sample,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=["cnn", "gla", "mamba", "retnet"], default="cnn")
    ap.add_argument("--dataset", choices=["ccdc_sg", "inor", "coremof19"], default="ccdc_sg")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--use_hier_head", action="store_true")
    ap.add_argument("--save_dir", type=str, default="checkpoints/run")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_train_batches", type=int, default=None,
                    help="for smoke test, cap batches/epoch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--class_weight", choices=["none", "sqrt_inv", "inv"], default="sqrt_inv",
                    help="re-weight CE loss by class frequency to fight 22.45% majority-class collapse")
    ap.add_argument("--downsample_to", type=int, default=0,
                    help="if > 0, bilinearly resample input from 8500 to this length before stem")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[LineXRD] backbone={args.backbone} dataset={args.dataset} device={device}")
    train_loader, val_loader, train_ds, val_ds = make_loaders(
        args.dataset, args.batch_size, args.num_workers
    )
    print(f"[data] train={len(train_ds)} val={len(val_ds)}")

    # Class re-weighting from training-set frequencies (sqrt-inverse by default).
    class_weight = None
    if args.class_weight != "none":
        counts = np.bincount(train_ds.labels230, minlength=N_CLASSES).astype(np.float64)
        counts = np.maximum(counts, 1.0)
        if args.class_weight == "sqrt_inv":
            w = 1.0 / np.sqrt(counts)
        else:
            w = 1.0 / counts
        w = w * (N_CLASSES / w.sum())
        class_weight = torch.tensor(w, dtype=torch.float32, device=device)
        print(f"[class_weight] {args.class_weight}: min={w.min():.3f} max={w.max():.3f} "
              f"top1_class_w={w[train_ds.labels230[0]]:.3f}")

    model = LineXRDClassifier(
        args.backbone, args.d_model, args.depth, args.use_hier_head
    ).to(device)
    # optional input downsampling for the resolution ablation
    if args.downsample_to and args.downsample_to > 0:
        DS_LEN = int(args.downsample_to)
        orig_forward = model.forward

        def fwd_ds(x):
            x_ds = F.interpolate(x.unsqueeze(1), size=DS_LEN, mode='linear',
                                 align_corners=False).squeeze(1)
            # pad / repeat back to 8500 so the stem matches; simpler than
            # retraining the stem for a smaller L.
            x_up = F.interpolate(x_ds.unsqueeze(1), size=8500, mode='linear',
                                 align_corners=False).squeeze(1)
            return orig_forward(x_up)
        model.forward = fwd_ds
        print(f"[downsample] input bilinearly resampled 8500 -> {DS_LEN} -> 8500")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params/1e6:.2f}M")

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_epochs = max(1, args.epochs // 20)         # 5% warmup
    sched = SequentialLR(
        opt,
        schedulers=[
            LambdaLR(opt, lr_lambda=lambda e: (e + 1) / warmup_epochs),
            CosineAnnealingLR(opt, T_max=max(1, args.epochs - warmup_epochs)),
        ],
        milestones=[warmup_epochs],
    )
    scaler = GradScaler()

    best_top1 = 0.0
    epoch_secs = []
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        loss_sum, n = 0.0, 0
        for i, (x, y_sg, y_cs) in enumerate(train_loader):
            if args.max_train_batches is not None and i >= args.max_train_batches:
                break
            x = x.to(device, non_blocking=True)
            y_sg = y_sg.to(device, non_blocking=True)
            y_cs = y_cs.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast():
                out = model(x)
                loss = F.cross_entropy(
                    out["sg_logits"], y_sg,
                    weight=class_weight,
                    label_smoothing=args.label_smoothing,
                )
                if "cs_logits" in out:
                    loss = loss + 0.3 * F.cross_entropy(
                        out["cs_logits"], y_cs,
                        label_smoothing=args.label_smoothing,
                    )
            if not torch.isfinite(loss):
                print(f"  [warn] non-finite loss at epoch {epoch+1} batch {i}; skipping step")
                opt.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            loss_sum += float(loss.item()) * x.size(0)
            n += x.size(0)
        sched.step()
        t_train = time.time() - t0
        epoch_secs.append(t_train)
        # eval
        metrics = evaluate(model, val_loader, device, max_batches=None)
        avg_loss = loss_sum / max(1, n)
        print(
            f"[ep {epoch+1:02d}] loss={avg_loss:.4f} "
            f"sg_top1={metrics['sg_top1']*100:.2f} "
            f"cs_top1={metrics['cs_top1']*100:.2f} "
            f"macroR={metrics['sg_macro_recall']*100:.2f} "
            f"train_s={t_train:.1f} val_ms={metrics['infer_ms_per_sample']:.2f}"
        )
        if metrics["sg_top1"] > best_top1:
            best_top1 = metrics["sg_top1"]
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "args": vars(args),
                "metrics": metrics,
            }, save_dir / "best.pt")

    final_metrics = evaluate(model, val_loader, device)
    final_metrics["best_sg_top1"] = best_top1
    final_metrics["params_M"] = n_params / 1e6
    final_metrics["train_sec_per_epoch"] = float(np.mean(epoch_secs)) if epoch_secs else 0.0
    final_metrics["backbone"] = args.backbone
    final_metrics["dataset"] = args.dataset
    with open(save_dir / "final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    # autoresearch METRIC lines
    print(f"METRIC sg_top1={final_metrics['sg_top1']:.4f}")
    print(f"METRIC best_sg_top1={best_top1:.4f}")
    print(f"METRIC cs_top1={final_metrics['cs_top1']:.4f}")
    print(f"METRIC sg_macro_recall={final_metrics['sg_macro_recall']:.4f}")
    print(f"METRIC params_M={final_metrics['params_M']:.3f}")
    print(f"METRIC train_sec_per_epoch={final_metrics['train_sec_per_epoch']:.1f}")
    print(f"METRIC infer_ms_per_sample={final_metrics['infer_ms_per_sample']:.3f}")


if __name__ == "__main__":
    main()
