"""
Evaluation / inference script for APTOS checkpoints.
Compatible with checkpoints that include num_classes.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import APTOSDataset
from model import build_model

NUM_CLASSES = 5
CLASS_NAMES = ["Grade 0", "Grade 1", "Grade 2", "Grade 3", "Grade 4"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OHEM APTOS Evaluation")
    p.add_argument("--checkpoint", required=True, help="Path to best_model.pth")
    p.add_argument("--csv", required=True, help="CSV file to evaluate on")
    p.add_argument("--img_dir", required=True, help="Image directory")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="/kaggle/working")
    p.add_argument(
        "--trust_checkpoint",
        action="store_true",
        help="Allow unsafe checkpoint loading (weights_only=False).",
    )
    return p.parse_args()


def load_checkpoint(path: str, device: torch.device, trust_checkpoint: bool):
    load_kwargs = {"map_location": device}

    if trust_checkpoint:
        try:
            return torch.load(path, weights_only=False, **load_kwargs)
        except TypeError:
            return torch.load(path, **load_kwargs)

    try:
        try:
            return torch.load(path, weights_only=True, **load_kwargs)
        except TypeError:
            return torch.load(path, **load_kwargs)
    except pickle.UnpicklingError as e:
        raise RuntimeError(
            "Checkpoint could not be loaded with safe mode (weights_only=True). "
            "If trusted, rerun with --trust_checkpoint."
        ) from e


@torch.no_grad()
def run_inference(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    return np.concatenate(all_labels), np.concatenate(all_preds)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = load_checkpoint(args.checkpoint, device, args.trust_checkpoint)
    num_classes = int(ckpt.get("num_classes", NUM_CLASSES))
    model = build_model(num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])

    if args.split == "val":
        full_ds = APTOSDataset(args.csv, args.img_dir, split="val")
        labels = full_ds.df["diagnosis"].values
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_frac, random_state=args.seed)
        _, val_idx = next(sss.split(np.zeros(len(labels)), labels))
        dataset = Subset(full_ds, val_idx)
    else:
        dataset = APTOSDataset(args.csv, args.img_dir, split="test")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    y_true, y_pred = run_inference(model, loader, device)

    os.makedirs(args.output_dir, exist_ok=True)
    if args.split == "test":
        df_out = APTOSDataset(args.csv, args.img_dir, split="test").df.copy()
        df_out["diagnosis"] = y_pred
        pred_path = os.path.join(args.output_dir, "test_predictions.csv")
        df_out[["id_code", "diagnosis"]].to_csv(pred_path, index=False)
        print(f"Predictions saved -> {pred_path}")
        return

    acc = accuracy_score(y_true, y_pred)
    f1_m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    class_names = CLASS_NAMES if num_classes == len(CLASS_NAMES) else [f"Class {i}" for i in range(num_classes)]
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1_m:.4f}")
    print(f"Weighted F1: {f1_w:.4f}")
    print(f"Macro Precision: {prec:.4f}")
    print(f"Macro Recall: {rec:.4f}")
    print(f"QWK: {qwk:.4f}")
    print("\nConfusion matrix:")
    print(pd.DataFrame(cm, index=class_names, columns=class_names).to_string())
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))


if __name__ == "__main__":
    main()
