"""
dataset_perch.py — Dataset for cached Perch embeddings.

Unlike TrainAudioDataset (per-window), PerchEmbeddingDataset returns
one sample per FILE (12 windows), enabling temporal sequence modeling.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional


class PerchEmbeddingDataset(Dataset):
    """
    Loads pre-cached Perch embeddings from perch_arrays.npz.
    Each sample is one file = 12 windows of 5 seconds.

    Args:
        cache_dir    : directory containing perch_arrays.npz + perch_meta.parquet
        label_matrix : (N_windows, n_classes) float32 ground truth labels
        meta_df      : DataFrame with columns row_id, filename, site, hour_utc
        n_windows    : windows per file (default 12)
        site_to_idx  : dict mapping site string → int index
    """

    def __init__(
        self,
        cache_dir: str,
        label_matrix: Optional[np.ndarray],
        meta_df: pd.DataFrame,
        n_windows: int = 12,
        site_to_idx: Optional[dict] = None,
    ):
        arrays = np.load(os.path.join(cache_dir, 'perch_arrays.npz'))
        self.emb_full = arrays['emb_full'].astype(np.float32)        # (N, 1536)
        self.scores_full = arrays['scores_full_raw'].astype(np.float32)  # (N, n_classes)

        self.meta_df = meta_df.reset_index(drop=True)
        self.label_matrix = label_matrix                              # (N, n_classes) or None
        self.n_windows = n_windows

        # Build site index mapping
        if site_to_idx is None:
            unique_sites = sorted(meta_df['site'].unique().tolist())
            site_to_idx = {s: i for i, s in enumerate(unique_sites)}
        self.site_to_idx = site_to_idx

        # Group windows by file to build file-level index
        self.meta_df['_site_idx'] = meta_df['site'].map(
            lambda s: site_to_idx.get(s, len(site_to_idx))
        )

        # Get unique files and their window start indices in emb_full
        filenames = meta_df['filename'].tolist()
        self.files = []
        i = 0
        while i < len(filenames):
            fname = filenames[i]
            start = i
            while i < len(filenames) and filenames[i] == fname:
                i += 1
            self.files.append({'filename': fname, 'start': start, 'end': i})

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        info = self.files[idx]
        s, e = info['start'], info['end']

        # Slice windows for this file
        emb = self.emb_full[s:e]        # (T, 1536)
        scores = self.scores_full[s:e]  # (T, n_classes)

        # Pad or trim to n_windows
        T = emb.shape[0]
        if T < self.n_windows:
            pad = self.n_windows - T
            emb = np.pad(emb, ((0, pad), (0, 0)))
            scores = np.pad(scores, ((0, pad), (0, 0)))
        elif T > self.n_windows:
            emb = emb[:self.n_windows]
            scores = scores[:self.n_windows]

        # Labels
        if self.label_matrix is not None:
            labels = self.label_matrix[s:e]
            if labels.shape[0] < self.n_windows:
                labels = np.pad(labels, ((0, self.n_windows - labels.shape[0]), (0, 0)))
            elif labels.shape[0] > self.n_windows:
                labels = labels[:self.n_windows]
        else:
            labels = np.zeros((self.n_windows, scores.shape[1]), dtype=np.float32)

        # Metadata (use first window's metadata)
        row = self.meta_df.iloc[s]
        site_idx = int(row['_site_idx'])
        hour = int(row.get('hour_utc', 0))

        return {
            'embeddings': torch.from_numpy(emb),        # (12, 1536)
            'logits': torch.from_numpy(scores),          # (12, n_classes)
            'labels': torch.from_numpy(labels.astype(np.float32)),  # (12, n_classes)
            'site_id': torch.tensor(site_idx, dtype=torch.long),
            'hour': torch.tensor(hour, dtype=torch.long),
            'filename': info['filename'],
        }
