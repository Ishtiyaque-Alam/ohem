"""
Online Hard Example Mining (OHEM) for classification.

Concept (from Shrivastava et al., CVPR 2016):
  1. Forward pass the full batch WITHOUT gradient tracking.
  2. Compute per-sample loss.
  3. Select the top-K samples with the HIGHEST loss  ← "hard examples"
  4. Re-forward only those K samples WITH gradient, then backward.

This forces the network to focus on examples it currently handles worst,
mirroring the original paper's ROI-level hard-example selection.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OHEMLoss(nn.Module):
    """
    Drop-in replacement for CrossEntropyLoss that applies OHEM.

    Args:
        ohem_ratio  : fraction of each batch to keep (default 0.5 → top-50%)
        min_kept    : minimum number of samples to keep regardless of ratio
        ignore_index: passed through to F.cross_entropy
    """

    def __init__(
        self,
        ohem_ratio: float = 0.5,
        min_kept: int = 1,
        ignore_index: int = -100,
    ):
        super().__init__()
        assert 0.0 < ohem_ratio <= 1.0, "ohem_ratio must be in (0, 1]"
        self.ohem_ratio   = ohem_ratio
        self.min_kept     = min_kept
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (B, C)  — raw model outputs
            targets : (B,)    — integer class labels
        Returns:
            scalar loss (mean over hard examples)
        """
        B = logits.size(0)

        # ── Step 1: per-sample loss, no gradient ──────────────────────────
        with torch.no_grad():
            per_loss = F.cross_entropy(
                logits, targets,
                reduction="none",
                ignore_index=self.ignore_index,
            )                                   # shape: (B,)

        # ── Step 2: select top-K hard examples ────────────────────────────
        k = max(self.min_kept, int(B * self.ohem_ratio))
        k = min(k, B)                           # safety clamp

        _, hard_idx = per_loss.topk(k, largest=True, sorted=False)

        # ── Step 3: compute loss ONLY on hard examples (with gradient) ────
        hard_loss = F.cross_entropy(
            logits[hard_idx],
            targets[hard_idx],
            reduction="mean",
            ignore_index=self.ignore_index,
        )
        return hard_loss

    def extra_repr(self) -> str:
        return f"ohem_ratio={self.ohem_ratio}, min_kept={self.min_kept}"
