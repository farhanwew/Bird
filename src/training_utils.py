"""
training_utils.py — Shared training utilities for the SSM pipeline.
"""

import numpy as np
import torch
import torch.nn.functional as F


def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    pos_weight: torch.Tensor = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Focal BCE loss for multi-label classification.
    Down-weights easy (well-classified) examples so training focuses
    on hard, rare species.

    Args:
        logits:     raw model outputs (any shape)
        targets:    same shape as logits, values in [0, 1]
        gamma:      focusing parameter (0 = standard BCE, 2 = typical focal)
        pos_weight: per-class weight tensor for class imbalance
        reduction:  'mean' or 'none'
    """
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none"
    )
    p = torch.sigmoid(logits)
    pt = targets * p + (1 - targets) * (1 - p)
    focal_weight = (1 - pt) ** gamma
    loss = focal_weight * bce
    return loss.mean() if reduction == "mean" else loss


def build_pos_weights(y: np.ndarray, cap: float = 25.0) -> torch.Tensor:
    """
    Inverse-frequency positive weights for BCEWithLogitsLoss.
    Rare species get higher weight.

    Args:
        y:   (N, n_classes) binary label matrix
        cap: maximum weight to avoid extreme values
    Returns:
        pos_weight: (n_classes,) tensor
    """
    n = y.shape[0]
    pos_count = y.sum(axis=0).clip(min=1)
    neg_count = n - pos_count
    weights = (neg_count / pos_count).clip(max=cap)
    return torch.tensor(weights, dtype=torch.float32)


def mixup_files(
    emb: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    site_ids: torch.Tensor,
    hours: torch.Tensor,
    alpha: float = 0.4,
):
    """
    File-level Mixup augmentation for ProtoSSM training.
    Interpolates entire file sequences (12 windows) between two files.

    Args:
        emb:      (B, T, 1536)
        logits:   (B, T, n_classes)
        labels:   (B, T, n_classes)
        site_ids: (B,)
        hours:    (B,)
        alpha:    Beta distribution parameter (0 = no mixup)
    Returns:
        Mixed versions of all inputs.
    """
    if alpha <= 0:
        return emb, logits, labels, site_ids, hours

    lam = float(np.random.beta(alpha, alpha))
    B = emb.size(0)
    idx = torch.randperm(B, device=emb.device)

    emb_mix = lam * emb + (1 - lam) * emb[idx]
    logits_mix = lam * logits + (1 - lam) * logits[idx]
    labels_mix = lam * labels + (1 - lam) * labels[idx]

    # For metadata, use the dominant file's site/hour
    site_mix = site_ids if lam >= 0.5 else site_ids[idx]
    hour_mix = hours if lam >= 0.5 else hours[idx]

    return emb_mix, logits_mix, labels_mix, site_mix, hour_mix


def build_class_freq_weights(y: np.ndarray, cap: float = 10.0) -> np.ndarray:
    """
    Per-sample inverse-frequency weights for WeightedRandomSampler.

    Args:
        y:   (N, n_classes) label matrix (can be multi-label)
        cap: max weight ratio
    Returns:
        sample_weights: (N,) float array
    """
    pos_count = y.sum(axis=0).clip(min=1)
    class_weights = 1.0 / pos_count
    class_weights = class_weights / class_weights.max()
    # Each sample weight = max class weight among its active classes
    sample_weights = (y * class_weights).max(axis=1)
    sample_weights = sample_weights.clip(min=sample_weights[sample_weights > 0].min())
    return sample_weights.astype(np.float32)
