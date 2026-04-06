"""
prior.py — Site × hour prior tables for BirdCLEF predictions.

Prior tables capture the empirical distribution of species by recording
location and time of day. Fusing these with model predictions reduces
false positives from species that simply cannot be present at a given
site or time.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


class PriorAndProbeManager:
    """
    Fits and applies site/hour prior tables to model predictions.

    Usage:
        mgr = PriorAndProbeManager(label_list)
        tables = mgr.fit_prior_tables(meta_df, label_matrix)
        prior_logits = mgr.prior_logits_from_tables(sites, hours, tables)
        fused = mgr.fuse_scores(base_logits, prior_logits, weight=0.45)
    """

    def __init__(self, label_list: List[str], eps: float = 1e-4):
        self.label_list = label_list
        self.n_classes = len(label_list)
        self.eps = eps

    # ------------------------------------------------------------------
    # Fit prior tables
    # ------------------------------------------------------------------

    def fit_prior_tables(
        self,
        meta_df: pd.DataFrame,
        label_matrix: np.ndarray,
    ) -> Dict:
        """
        Compute empirical prior probabilities of each species given
        global, site, hour, and joint site×hour context.

        Args:
            meta_df:      DataFrame with row_id, site, hour_utc columns
            label_matrix: (N_windows, n_classes) binary ground truth
        Returns:
            tables: dict with keys 'global', 'site', 'hour', 'site_hour'
        """
        y = (label_matrix > 0.5).astype(np.float32)
        sites = meta_df['site'].tolist()
        hours = meta_df['hour_utc'].tolist()

        unique_sites = sorted(set(sites))
        site_to_idx = {s: i for i, s in enumerate(unique_sites)}
        n_sites = len(unique_sites)
        n_hours = 24

        # Global prior: P(species)
        global_prior = (y.sum(0) + self.eps) / (len(y) + self.eps)

        # Site prior: P(species | site)
        site_prior = np.zeros((n_sites, self.n_classes), dtype=np.float32)
        site_counts = np.zeros(n_sites, dtype=np.float32)
        for i, site in enumerate(sites):
            si = site_to_idx.get(site, 0)
            site_prior[si] += y[i]
            site_counts[si] += 1
        site_prior = (site_prior + self.eps) / (site_counts[:, None] + self.eps)

        # Hour prior: P(species | hour)
        hour_prior = np.zeros((n_hours, self.n_classes), dtype=np.float32)
        hour_counts = np.zeros(n_hours, dtype=np.float32)
        for i, hour in enumerate(hours):
            h = int(hour) % 24
            hour_prior[h] += y[i]
            hour_counts[h] += 1
        hour_prior = (hour_prior + self.eps) / (hour_counts[:, None] + self.eps)

        # Site × hour joint prior
        site_hour_prior = np.zeros((n_sites, n_hours, self.n_classes), dtype=np.float32)
        site_hour_counts = np.zeros((n_sites, n_hours), dtype=np.float32)
        for i, (site, hour) in enumerate(zip(sites, hours)):
            si = site_to_idx.get(site, 0)
            h = int(hour) % 24
            site_hour_prior[si, h] += y[i]
            site_hour_counts[si, h] += 1
        # Avoid division by zero
        denom = site_hour_counts[:, :, None] + self.eps
        site_hour_prior = (site_hour_prior + self.eps) / denom

        return {
            'global': global_prior,           # (n_classes,)
            'site': site_prior,               # (n_sites, n_classes)
            'hour': hour_prior,               # (24, n_classes)
            'site_hour': site_hour_prior,     # (n_sites, 24, n_classes)
            'site_to_idx': site_to_idx,
            'n_sites': n_sites,
        }

    # ------------------------------------------------------------------
    # Apply prior tables
    # ------------------------------------------------------------------

    def prior_logits_from_tables(
        self,
        sites: List[str],
        hours: List[int],
        tables: Dict,
        blend: float = 0.6,
    ) -> np.ndarray:
        """
        Compute log-odds prior for each window from fitted tables.

        Args:
            sites:  list of site strings per window
            hours:  list of hour ints per window
            tables: output of fit_prior_tables()
            blend:  weight for site×hour joint vs. marginals
        Returns:
            prior_logits: (N, n_classes) log-odds
        """
        site_to_idx = tables['site_to_idx']
        N = len(sites)
        prior_probs = np.zeros((N, self.n_classes), dtype=np.float32)

        for i, (site, hour) in enumerate(zip(sites, hours)):
            si = site_to_idx.get(site, 0)
            h = int(hour) % 24

            # Blend global, site, hour, and joint priors
            p = (
                0.1 * tables['global']
                + 0.2 * tables['site'][si]
                + 0.2 * tables['hour'][h]
                + 0.5 * tables['site_hour'][si, h]
            )
            prior_probs[i] = p

        prior_probs = np.clip(prior_probs, self.eps, 1 - self.eps)
        return np.log(prior_probs / (1 - prior_probs))  # log-odds

    # ------------------------------------------------------------------
    # Score fusion
    # ------------------------------------------------------------------

    @staticmethod
    def fuse_scores(
        base_logits: np.ndarray,
        prior_logits: np.ndarray,
        weight: float = 0.45,
    ) -> np.ndarray:
        """
        Fuse base model logits with prior log-odds.

        Args:
            base_logits:  (N, n_classes)
            prior_logits: (N, n_classes) from prior_logits_from_tables()
            weight:       blending weight for prior (0 = no prior)
        Returns:
            fused_logits: (N, n_classes)
        """
        return (1 - weight) * base_logits + weight * prior_logits

    # ------------------------------------------------------------------
    # Temporal smoothing
    # ------------------------------------------------------------------

    @staticmethod
    def smooth_temporal(
        scores: np.ndarray,
        n_windows: int = 12,
        alpha: float = 0.20,
        mode: str = 'mean',
    ) -> np.ndarray:
        """
        Temporal smoothing within each file.

        Args:
            scores:    (N_total, n_classes) — windows are file-contiguous
            n_windows: windows per file
            alpha:     smoothing strength
            mode:      'mean' or 'max' (how to combine with neighbors)
        Returns:
            smoothed scores
        """
        if alpha <= 0:
            return scores

        N_total = scores.shape[0]
        n_files = N_total // n_windows
        out = scores.copy()

        for fi in range(n_files):
            s = fi * n_windows
            e = s + n_windows
            p = scores[s:e].copy()
            T = p.shape[0]
            smooth = p.copy()

            for t in range(T):
                prev = p[t - 1] if t > 0 else p[t]
                nxt = p[t + 1] if t < T - 1 else p[t]
                if mode == 'max':
                    neighbors = np.maximum(prev, nxt)
                else:
                    neighbors = 0.5 * (prev + nxt)
                smooth[t] = (1 - alpha) * p[t] + alpha * neighbors

            out[s:e] = smooth

        return out

    # ------------------------------------------------------------------
    # Handcrafted features for MLP probe
    # ------------------------------------------------------------------

    @staticmethod
    def build_class_features(
        emb_pca: np.ndarray,
        raw_scores: np.ndarray,
        prior_logits: np.ndarray,
        base_scores: np.ndarray,
        n_windows: int = 12,
        class_idx: int = 0,
    ) -> np.ndarray:
        """
        Build 15-dim feature vector per window per class for MLP probe.

        Features:
          0:     raw Perch score
          1:     prior logit
          2:     fused base score
          3:     previous window score
          4:     next window score
          5:     file-level mean score
          6:     file-level max score
          7:     file-level std
          8:     delta: current - previous
          9:     delta: current - mean
          10:    delta: current - next
          11:    interaction: raw * prior
          12:    interaction: raw * base
          13:    interaction: prior * base
          14:    PCA component 0 (most informative)

        Args:
            emb_pca:    (N, n_pca) PCA-compressed embeddings
            raw_scores: (N,) raw Perch scores for this class
            prior_logits: (N,) prior log-odds for this class
            base_scores: (N,) fused base scores for this class
            n_windows:  windows per file
            class_idx:  unused (for reference)
        Returns:
            features: (N, 15)
        """
        N = len(raw_scores)
        n_files = N // n_windows
        feats = np.zeros((N, 15), dtype=np.float32)

        for fi in range(n_files):
            s = fi * n_windows
            e = s + n_windows
            raw = raw_scores[s:e]
            prior = prior_logits[s:e]
            base = base_scores[s:e]
            T = e - s

            file_mean = raw.mean()
            file_max = raw.max()
            file_std = raw.std() + 1e-6

            for t in range(T):
                i = s + t
                prev = raw[t - 1] if t > 0 else raw[t]
                nxt = raw[t + 1] if t < T - 1 else raw[t]

                feats[i, 0] = raw[t]
                feats[i, 1] = prior[t]
                feats[i, 2] = base[t]
                feats[i, 3] = prev
                feats[i, 4] = nxt
                feats[i, 5] = file_mean
                feats[i, 6] = file_max
                feats[i, 7] = file_std
                feats[i, 8] = raw[t] - prev
                feats[i, 9] = raw[t] - file_mean
                feats[i, 10] = raw[t] - nxt
                feats[i, 11] = raw[t] * prior[t]
                feats[i, 12] = raw[t] * base[t]
                feats[i, 13] = prior[t] * base[t]
                feats[i, 14] = emb_pca[i, 0] if emb_pca.shape[1] > 0 else 0.0

        return feats
