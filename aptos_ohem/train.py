"""
Training script – OHEM classification on APTOS 2019.

Example:
    python train.py \
        --train_csv  /kaggle/input/datasets/mariaherrerot/aptos2019/train_1.csv \
        --img_dir    /kaggle/input/datasets/mariaherrerot/aptos2019/train_images/train_images \
        --epochs     50 \
        --batch_size 32 \
        --ohem_ratio 0.5 \
        --output_dir /kaggle/working/aptos_ckpt
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

# ── local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset   import APTOSDataset
from model     import build_model
from ohem_loss import OHEMLoss


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OHEM APTOS 2019 Training")
    p.add_argument("--train_csv",  required=True)
    p.add_argument("--img_dir",    required=True)
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--ohem_ratio", type=float, default=0.5,
                   help="Fraction of each batch used for backward (OHEM ratio)")
    p.add_argument("--val_frac",   type=float, default=0.2,
                   help="Fraction of train_csv held out for validation")
    p.add_argument("--num_workers",type=int,   default=4)
    p.add_argument("--output_dir", type=str,   default="aptos_ckpt")
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


def seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loaders(args) -> tuple[DataLoader, DataLoader]:
    """Stratified split of train_csv → train / val loaders."""
    full_ds = APTOSDataset(args.train_csv, args.img_dir, split="train")
    labels  = full_ds.df["diagnosis"].values

    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=args.val_frac, random_state=args.seed
    )
    train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))

    # val dataset uses test-time transforms
    val_ds = APTOSDataset(args.train_csv, args.img_dir, split="val")

    train_loader = DataLoader(
        Subset(full_ds, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(val_ds, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"  Train samples: {len(train_idx)} | Val samples: {len(val_idx)}")
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# One epoch
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    criterion:  OHEMLoss,
    optimizer:  torch.optim.Optimizer,
    device:     torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(imgs)                    # (B, 5)
        loss   = criterion(logits, labels)      # OHEM selection inside
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    qwk  = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    cm   = confusion_matrix(y_true, y_pred, labels=list(range(5)))

    return {"acc": acc, "f1": f1, "qwk": qwk, "cm": cm,
            "y_true": y_true, "y_pred": y_pred}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  OHEM APTOS 2019 Training")
    print(f"  Device: {device}  |  Epochs: {args.epochs}  |  "
          f"Batch: {args.batch_size}  |  OHEM ratio: {args.ohem_ratio}")
    print(f"{'='*60}\n")

    # ── data ──────────────────────────────────────────────────────────────
    train_loader, val_loader = make_loaders(args)

    # ── model ─────────────────────────────────────────────────────────────
    model = build_model(pretrained=True).to(device)

    # ── loss / optimizer / scheduler ──────────────────────────────────────
    criterion = OHEMLoss(ohem_ratio=args.ohem_ratio, min_kept=8)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_qwk  = -1.0
    log_rows  = []

    for epoch in range(1, args.epochs + 1):
        t0        = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        metrics    = evaluate(model, val_loader, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch [{epoch:02d}/{args.epochs}] "
            f"loss={train_loss:.4f}  "
            f"acc={metrics['acc']:.4f}  "
            f"f1={metrics['f1']:.4f}  "
            f"qwk={metrics['qwk']:.4f}  "
            f"({elapsed:.1f}s)"
        )

        log_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            **{k: metrics[k] for k in ("acc", "f1", "qwk")},
        })

        # ── save best ─────────────────────────────────────────────────────
        if metrics["qwk"] > best_qwk:
            best_qwk = metrics["qwk"]
            ckpt_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(
                {
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "qwk":         best_qwk,
                    "ohem_ratio":  args.ohem_ratio,
                },
                ckpt_path,
            )
            print(f"  ✔ New best QWK={best_qwk:.4f} → saved to {ckpt_path}")

    # ── save training log ──────────────────────────────────────────────────
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(os.path.join(args.output_dir, "train_log.csv"), index=False)
    print(f"\nTraining complete. Best QWK = {best_qwk:.4f}")


if __name__ == "__main__":
    main()
