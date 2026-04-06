"""
model_perch.py — Simple MLP classifier on Perch embeddings.

Used as a baseline sanity check before ProtoSSM.
Expected ROC-AUC: ~0.80+ just from Perch embeddings + MLP head.
"""

import torch
import torch.nn as nn
from typing import Tuple


class PerchMLPClassifier(nn.Module):
    """
    Simple per-window MLP head on top of Perch 1536-dim embeddings.

    Accepts both per-window (B, 1536) and per-file (B, T, 1536) input.
    Output shape matches input: (B, n_classes) or (B, T, n_classes).
    """
    def __init__(
        self,
        d_input: int = 1536,
        n_classes: int = 234,
        hidden: Tuple[int, ...] = (512, 256),
        dropout: float = 0.3,
    ):
        super().__init__()
        layers = []
        in_dim = d_input
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (B, 1536) or (B, T, 1536)
        Returns:
            logits: (B, n_classes) or (B, T, n_classes)
        """
        return self.net(embeddings)
