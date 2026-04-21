"""
Training script: OHEM classification on APTOS 2019 or HAM10000.

Example (APTOS):
    python train.py \
        --dataset aptos \
        --train_csv /kaggle/input/datasets/mariaherrerot/aptos2019/train_1.csv \
        --img_dir /kaggle/input/datasets/mariaherrerot/aptos2019/train_images/train_images \
        --epochs 50 --batch_size 32 --ohem_ratio 0.5 \
        --output_dir /kaggle/working/aptos_ckpt

Example (HAM10000):
    python train.py \
        --dataset ham10000 \
        --train_csv /kaggle/input/datasets/kmader/skin-cancer-mnist-ham10000/HAM10000_metadata.csv \
        --ham_img_dir_1 /kaggle/input/datasets/kmader/skin-cancer-mnist-ham10000/HAM10000_images_part_1 \
        --ham_img_dir_2 /kaggle/input/datasets/kmader/skin-cancer-mnist-ham10000/HAM10000_images_part_2 \
        --epochs 50 --batch_size 32 --ohem_ratio 0.5 \
        --output_dir /kaggle/working/ham_ckpt
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
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import APTOSDataset, HAM10000Dataset, build_ham_dataframe
from model import build_model
from ohem_loss import OHEMLoss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OHEM Training (APTOS / HAM10000)")
    p.add_argument("--dataset", choices=["aptos", "ham10000"], default="aptos")

    p.add_argument("--train_csv", required=True)
    p.add_argument("--img_dir", default=None, help="APTOS image directory")
    p.add_argument("--ham_img_dir_1", default=None, help="HAM10000 images part 1")
    p.add_argument("--ham_img_dir_2", default=None, help="HAM10000 images part 2")

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument(
        "--ohem_ratio",
        type=float,
        default=0.5,
        help="Fraction of each batch used for backward (OHEM ratio)",
    )
    p.add_argument(
        "--val_frac",
        type=float,
        default=0.2,
        help="Fraction of train_csv held out for validation",
    )
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_dir", type=str, default="ckpt")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def seed_everything(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loaders(args) -> tuple[DataLoader, DataLoader, int]:
    """Build train/val loaders with stratified split for the selected dataset."""
    if args.dataset == "aptos":
        if not args.img_dir:
            raise ValueError("For --dataset aptos, --img_dir is required.")

        full_ds = APTOSDataset(args.train_csv, args.img_dir, split="train")
        labels = full_ds.df["diagnosis"].values

        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=args.val_frac, random_state=args.seed
        )
        train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))

        val_ds = APTOSDataset(args.train_csv, args.img_dir, split="val")
        num_classes = 5

    else:
        if not args.ham_img_dir_1 or not args.ham_img_dir_2:
            raise ValueError(
                "For --dataset ham10000, both --ham_img_dir_1 and --ham_img_dir_2 are required."
            )

        ham_df = build_ham_dataframe(args.train_csv, args.ham_img_dir_1, args.ham_img_dir_2)
        labels = ham_df["label"].values

        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=args.val_frac, random_state=args.seed
        )
        train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))

        full_ds = HAM10000Dataset(ham_df, split="train")
        val_ds = HAM10000Dataset(ham_df, split="val")
        num_classes = 7

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
    return train_loader, val_loader, num_classes


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: OHEMLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> dict:
    model.eval()
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    return {"acc": acc, "f1": f1, "qwk": qwk, "cm": cm, "y_true": y_true, "y_pred": y_pred}


def main():
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  OHEM Training ({args.dataset.upper()})")
    print(
        f"  Device: {device}  |  Epochs: {args.epochs}  |  "
        f"Batch: {args.batch_size}  |  OHEM ratio: {args.ohem_ratio}"
    )
    print(f"{'='*60}\n")

    train_loader, val_loader, num_classes = make_loaders(args)
    model = build_model(num_classes=num_classes, pretrained=True).to(device)

    criterion = OHEMLoss(ohem_ratio=args.ohem_ratio, min_kept=8)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_qwk = -1.0
    log_rows = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        metrics = evaluate(model, val_loader, device, num_classes)
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

        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                **{k: metrics[k] for k in ("acc", "f1", "qwk")},
            }
        )

        if metrics["qwk"] > best_qwk:
            best_qwk = metrics["qwk"]
            ckpt_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "qwk": best_qwk,
                    "ohem_ratio": args.ohem_ratio,
                    "dataset": args.dataset,
                    "num_classes": num_classes,
                },
                ckpt_path,
            )
            print(f"  New best QWK={best_qwk:.4f} -> saved to {ckpt_path}")

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(os.path.join(args.output_dir, "train_log.csv"), index=False)
    print(f"\nTraining complete. Best QWK = {best_qwk:.4f}")


if __name__ == "__main__":
    main()
