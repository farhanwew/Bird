"""
postprocessing.py — Post-processing for BirdCLEF predictions.

Applied after the main ensemble to boost confidence and reduce noise.
All methods operate on (N_files * n_windows, n_classes) probability arrays
organized in file-contiguous order.
"""

import numpy as np
import pandas as pd
from typing import Optional, List


class PostProcessor:
    """
    Chain of post-processing operations applied to submission probabilities.

    Typical usage:
        pp = PostProcessor(n_windows=12)
        probs = pp.temperature_scale(logits, class_temps)
        probs = pp.file_level_confidence_scale(probs)
        probs = pp.rank_aware_scaling(probs, power=0.4)
        probs = pp.adaptive_delta_smooth(probs, alpha=0.20)
        probs = np.clip(probs, 0, 1)
    """

    def __init__(self, n_windows: int = 12):
        self.n_windows = n_windows

    # ------------------------------------------------------------------
    # Temperature scaling
    # ------------------------------------------------------------------

    @staticmethod
    def temperature_scale(
        logits: np.ndarray,
        temperatures: Optional[np.ndarray] = None,
        default_temp: float = 1.0,
    ) -> np.ndarray:
        """
        Per-class temperature scaling before sigmoid.

        Args:
            logits:       (N, n_classes) raw logits
            temperatures: (n_classes,) per-class temperature (higher = softer)
            default_temp: fallback temperature if temperatures is None
        Returns:
            probs: (N, n_classes) probabilities
        """
        if temperatures is not None:
            temps = np.array(temperatures, dtype=np.float32)
            scaled = logits / np.clip(temps, 0.1, 10.0)
        else:
            scaled = logits / default_temp
        return 1.0 / (1.0 + np.exp(-scaled))

    # ------------------------------------------------------------------
    # File-level confidence scaling
    # ------------------------------------------------------------------

    def file_level_confidence_scale(
        self,
        probs: np.ndarray,
        top_k: int = 2,
    ) -> np.ndarray:
        """
        Boost predictions for files where the model is confident.
        For each file, find the top-K predictions and use their
        mean confidence as a scaling factor.

        Args:
            probs:  (N_files * n_windows, n_classes)
            top_k:  number of top predictions to average
        Returns:
            scaled probs
        """
        N_total = probs.shape[0]
        n_files = N_total // self.n_windows
        out = probs.copy()

        for fi in range(n_files):
            s = fi * self.n_windows
            e = s + self.n_windows
            file_probs = probs[s:e]  # (n_windows, n_classes)

            # Top-K max probabilities across the entire file
            file_max = file_probs.max(axis=0)  # (n_classes,)
            top_conf = np.sort(file_max)[-top_k:].mean()
            scale = float(np.clip(top_conf * 2.0, 0.5, 2.0))
            out[s:e] = np.clip(file_probs * scale, 0, 1)

        return out

    # ------------------------------------------------------------------
    # Rank-aware scaling
    # ------------------------------------------------------------------

    def rank_aware_scaling(
        self,
        probs: np.ndarray,
        power: float = 0.4,
    ) -> np.ndarray:
        """
        Scale window predictions exponentially by the file's max confidence.
        Files where the model is very confident get boosted more.

        Args:
            probs: (N_files * n_windows, n_classes)
            power: exponent (0 = no scaling, 1 = linear)
        Returns:
            scaled probs
        """
        N_total = probs.shape[0]
        n_files = N_total // self.n_windows
        out = probs.copy()

        for fi in range(n_files):
            s = fi * self.n_windows
            e = s + self.n_windows
            file_probs = probs[s:e]
            file_max = file_probs.max()
            scale = float(file_max ** power) if file_max > 0 else 1.0
            out[s:e] = np.clip(file_probs * scale, 0, 1)

        return out

    # ------------------------------------------------------------------
    # Event vs texture class alpha builder
    # ------------------------------------------------------------------

    @staticmethod
    def build_class_alphas(
        taxonomy_csv: str,
        label_list: List[str],
        alpha_event: float = 0.15,
        alpha_texture: float = 0.35,
    ) -> np.ndarray:
        """
        Build per-class smoothing alpha based on taxonomic class.

        Event species (Aves, Mammalia, Reptilia) use alpha_event — shorter,
        discrete calls benefit from less temporal blending.
        Texture species (Amphibia, Insecta) use alpha_texture — continuous
        calls benefit from stronger temporal smoothing.

        Args:
            taxonomy_csv:  path to taxonomy.csv with primary_label + class_name
            label_list:    ordered list of competition species labels
            alpha_event:   smoothing for event species (default 0.15)
            alpha_texture: smoothing for texture species (default 0.35)
        Returns:
            alphas: (n_classes,) float32 array
        """
        alphas = np.full(len(label_list), alpha_event, dtype=np.float32)
        texture_classes = {'Amphibia', 'Insecta'}
        label_to_idx = {lbl: i for i, lbl in enumerate(label_list)}

        try:
            tax_df = pd.read_csv(taxonomy_csv)
            if 'class_name' in tax_df.columns and 'primary_label' in tax_df.columns:
                for _, row in tax_df.iterrows():
                    lbl = str(row['primary_label'])
                    cls = str(row['class_name'])
                    if lbl in label_to_idx and cls in texture_classes:
                        alphas[label_to_idx[lbl]] = alpha_texture
        except Exception:
            pass

        return alphas

    # ------------------------------------------------------------------
    # Adaptive delta smoothing
    # ------------------------------------------------------------------

    def adaptive_delta_smooth(
        self,
        probs: np.ndarray,
        alpha: float = 0.20,
        class_alphas: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Temporal smoothing across windows within each file.
        x'[t] = (1 - a) * x[t] + 0.5 * a * (x[t-1] + x[t+1])

        Args:
            probs:        (N_files * n_windows, n_classes)
            alpha:        scalar smoothing strength (overridden by class_alphas)
            class_alphas: (n_classes,) per-class alpha. When provided, event
                          species (Aves/Mammalia) get low alpha and texture
                          species (Amphibia/Insecta) get high alpha.
        Returns:
            smoothed probs
        """
        if alpha <= 0 and class_alphas is None:
            return probs

        N_total = probs.shape[0]
        n_files = N_total // self.n_windows
        out = probs.copy()

        # Per-class alpha broadcast shape: (1, n_classes)
        a = class_alphas[np.newaxis, :] if class_alphas is not None else alpha

        for fi in range(n_files):
            s = fi * self.n_windows
            e = s + self.n_windows
            p = probs[s:e].copy()  # (T, n_classes)
            T = p.shape[0]

            smooth = p.copy()
            for t in range(T):
                prev = p[t - 1] if t > 0 else p[t]
                nxt = p[t + 1] if t < T - 1 else p[t]
                smooth[t] = (1 - a) * p[t] + 0.5 * a * (prev + nxt)

            out[s:e] = smooth

        return out

    # ------------------------------------------------------------------
    # Per-class threshold application
    # ------------------------------------------------------------------

    @staticmethod
    def apply_thresholds(
        probs: np.ndarray,
        thresholds: np.ndarray,
        scale_below: float = 0.5,
    ) -> np.ndarray:
        """
        Soft threshold: scale down predictions below class-specific threshold.

        Args:
            probs:       (N, n_classes)
            thresholds:  (n_classes,) per-class thresholds
            scale_below: multiply predictions below threshold by this factor
        Returns:
            adjusted probs
        """
        out = probs.copy()
        below = probs < thresholds[np.newaxis, :]
        out[below] = probs[below] * scale_below
        return out

    # ------------------------------------------------------------------
    # Convenience: full post-processing chain
    # ------------------------------------------------------------------

    def process(
        self,
        logits: np.ndarray,
        class_temperatures: Optional[np.ndarray] = None,
        thresholds: Optional[np.ndarray] = None,
        do_confidence_scale: bool = True,
        do_rank_scale: bool = True,
        do_smooth: bool = True,
        smooth_alpha: float = 0.20,
        class_alphas: Optional[np.ndarray] = None,
        rank_power: float = 0.4,
    ) -> np.ndarray:
        """
        Full post-processing pipeline.

        Args:
            logits:             (N, n_classes) raw logits from ensemble
            class_temperatures: (n_classes,) per-class temperatures
            thresholds:         (n_classes,) per-class thresholds
            class_alphas:       (n_classes,) per-class smoothing alpha. Build
                                with PostProcessor.build_class_alphas() using
                                taxonomy.csv for event vs texture distinction.
        Returns:
            probs: (N, n_classes) final probabilities in [0, 1]
        """
        probs = self.temperature_scale(logits, class_temperatures)

        if do_confidence_scale:
            probs = self.file_level_confidence_scale(probs)

        if do_rank_scale:
            probs = self.rank_aware_scaling(probs, power=rank_power)

        if do_smooth:
            probs = self.adaptive_delta_smooth(probs, alpha=smooth_alpha, class_alphas=class_alphas)

        if thresholds is not None:
            probs = self.apply_thresholds(probs, thresholds)

        return np.clip(probs, 0.0, 1.0)
