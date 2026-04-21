"""
Evaluation / Inference script – APTOS 2019 with OHEM-trained model.

Outputs per-class TP, TN, FP, FN and overall Accuracy & F1 score.

Example (validation split – labelled):
    python evaluate.py \
        --checkpoint /kaggle/working/aptos_ckpt/best_model.pth \
        --csv        /kaggle/input/datasets/mariaherrerot/aptos2019/train_1.csv \
        --img_dir    /kaggle/input/datasets/mariaherrerot/aptos2019/train_images/train_images \
        --split      val

Example (test set – no labels, only predictions saved):
    python evaluate.py \
        --checkpoint /kaggle/working/aptos_ckpt/best_model.pth \
        --csv        /kaggle/input/datasets/mariaherrerot/aptos2019/test.csv \
        --img_dir    /kaggle/input/datasets/mariaherrerot/aptos2019/test_images/test_images \
        --split      test \
        --output_dir /kaggle/working
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
from model   import build_model

NUM_CLASSES  = 5
CLASS_NAMES  = ["Grade 0", "Grade 1", "Grade 2", "Grade 3", "Grade 4"]

# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OHEM APTOS 2019 Evaluation")
    p.add_argument("--checkpoint",  required=True, help="Path to best_model.pth")
    p.add_argument("--csv",         required=True, help="CSV file to evaluate on")
    p.add_argument("--img_dir",     required=True, help="Image directory")
    p.add_argument("--split",       default="val",
                   choices=["val", "test"],
                   help="'val' uses 20%% hold-out; 'test' runs on full CSV (no labels)")
    p.add_argument("--val_frac",    type=float, default=0.2)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--output_dir",  type=str,   default="/kaggle/working",
                   help="Where to save predictions CSV")
    p.add_argument(
        "--trust_checkpoint",
        action="store_true",
        help=(
            "Allow unsafe checkpoint loading (weights_only=False). "
            "Use only for trusted checkpoints."
        ),
    )
    return p.parse_args()


def load_checkpoint(path: str, device: torch.device, trust_checkpoint: bool):
    """
    PyTorch 2.6 changed torch.load default to weights_only=True.
    Keep the safe path by default, and require explicit opt-in for unsafe load.
    """
    load_kwargs = {"map_location": device}

    if trust_checkpoint:
        try:
            return torch.load(path, weights_only=False, **load_kwargs)
        except TypeError:
            # For older torch versions that do not expose weights_only.
            return torch.load(path, **load_kwargs)

    try:
        try:
            return torch.load(path, weights_only=True, **load_kwargs)
        except TypeError:
            return torch.load(path, **load_kwargs)
    except pickle.UnpicklingError as e:
        raise RuntimeError(
            "Checkpoint could not be loaded with safe mode (weights_only=True). "
            "If this checkpoint is trusted, rerun with --trust_checkpoint."
        ) from e


# ─────────────────────────────────────────────────────────────────────────────
# Confusion-matrix → per-class TP / TN / FP / FN
# ─────────────────────────────────────────────────────────────────────────────

def cm_to_per_class_stats(cm: np.ndarray) -> pd.DataFrame:
    """
    For each class i:
        TP_i = cm[i, i]
        FP_i = sum(cm[:, i]) - TP_i          (predicted i but not i)
        FN_i = sum(cm[i, :]) - TP_i          (actually i but predicted else)
        TN_i = total - TP_i - FP_i - FN_i
    """
    total  = cm.sum()
    rows   = []
    for i, name in enumerate(CLASS_NAMES):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum()) - tp
        fn = int(cm[i, :].sum()) - tp
        tn = int(total) - tp - fp - fn
        rows.append({"Class": name, "TP": tp, "TN": tn, "FP": fp, "FN": fn})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    return np.concatenate(all_labels), np.concatenate(all_preds)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── load model ────────────────────────────────────────────────────────
    ckpt  = load_checkpoint(args.checkpoint, device, args.trust_checkpoint)
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"\nLoaded checkpoint (epoch {ckpt['epoch']}, QWK={ckpt['qwk']:.4f})")

    # ── build dataset / loader ────────────────────────────────────────────
    if args.split == "val":
        full_ds = APTOSDataset(args.csv, args.img_dir, split="val")
        labels  = full_ds.df["diagnosis"].values
        sss     = StratifiedShuffleSplit(
            n_splits=1, test_size=args.val_frac, random_state=args.seed
        )
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

    # ── save predictions ──────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    if args.split == "test":
        df_out = APTOSDataset(args.csv, args.img_dir, split="test").df.copy()
        df_out["diagnosis"] = y_pred
        pred_path = os.path.join(args.output_dir, "test_predictions.csv")
        df_out[["id_code", "diagnosis"]].to_csv(pred_path, index=False)
        print(f"\nPredictions saved → {pred_path}")
        return

    # ── metrics (only when labels available) ─────────────────────────────
    acc  = accuracy_score(y_true, y_pred)
    f1_m = f1_score(y_true, y_pred, average="macro",  zero_division=0)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred,    average="macro", zero_division=0)
    qwk  = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    cm   = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    # ── per-class TP/TN/FP/FN ─────────────────────────────────────────────
    stats_df = cm_to_per_class_stats(cm)

    # ─────────────────────────────────────────────────────────────────────
    # Print report
    # ─────────────────────────────────────────────────────────────────────
    sep = "=" * 60
    print(f"\n{sep}")
    print("  EVALUATION RESULTS – APTOS 2019  (OHEM model)")
    print(sep)

    print(f"\n{'Metric':<30}{'Value':>10}")
    print("-" * 40)
    print(f"{'Accuracy':<30}{acc:>10.4f}")
    print(f"{'Macro F1':<30}{f1_m:>10.4f}")
    print(f"{'Weighted F1':<30}{f1_w:>10.4f}")
    print(f"{'Macro Precision':<30}{prec:>10.4f}")
    print(f"{'Macro Recall':<30}{rec:>10.4f}")
    print(f"{'Quadratic Weighted Kappa':<30}{qwk:>10.4f}")

    print(f"\n── Per-class TP / TN / FP / FN {'─'*28}")
    print(stats_df.to_string(index=False))

    print(f"\n── Confusion Matrix (rows=true, cols=pred) {'─'*15}")
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    print(cm_df.to_string())

    print(f"\n── Per-class Classification Report {'─'*22}")
    print(classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        zero_division=0,
    ))
    print(sep)

    # ── save stats to CSV ─────────────────────────────────────────────────
    stats_df["Accuracy"]  = acc
    stats_df["Macro_F1"]  = f1_m
    stats_df["QWK"]       = qwk
    out_csv = os.path.join(args.output_dir, "eval_results.csv")
    stats_df.to_csv(out_csv, index=False)
    print(f"\nDetailed results saved → {out_csv}")


if __name__ == "__main__":
    main()
