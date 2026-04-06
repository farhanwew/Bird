"""
taxonomy.py — TaxonomyManager for BirdCLEF 2026.

Handles:
  - Parsing soundscape filenames for site/hour metadata
  - Building label matrices from train_soundscapes_labels.csv
  - Mapping Perch class indices to competition classes
  - Building taxonomy family groups for auxiliary head
"""

import re
import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple


_FNAME_RE = re.compile(
    r'BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})',
    re.IGNORECASE,
)


def parse_filename_metadata(fname: str) -> Dict:
    """
    Parse site and hour_utc from BirdCLEF soundscape filename.
    Example: BC2026_Train_0001_S05_20250101_060000.ogg → site='S05', hour=6
    """
    m = _FNAME_RE.search(os.path.basename(fname))
    if m:
        time_str = m.group(4)
        return {'site': m.group(2), 'hour_utc': int(time_str[:2])}
    return {'site': 'UNKNOWN', 'hour_utc': 0}


class TaxonomyManager:
    """
    Central manager for species labels, taxonomy, and metadata.

    Usage:
        tax = TaxonomyManager(base_dir, taxonomy_csv, label_list)
        tax.build_label_matrices(soundscape_labels_csv, meta_df)
        tax.build_taxonomy_groups()
        perch_map = tax.map_perch_indices(perch_label_csv)
    """

    def __init__(
        self,
        base_dir: str,
        taxonomy_csv: Optional[str] = None,
        label_list: Optional[List[str]] = None,
    ):
        self.base_dir = base_dir

        # Load label list
        if label_list is not None:
            self.primary_labels = label_list
        else:
            label_json = os.path.join(base_dir, 'label_list.json')
            if os.path.exists(label_json):
                with open(label_json) as f:
                    self.primary_labels = json.load(f)
            else:
                raise FileNotFoundError(
                    "Provide label_list or ensure output/label_list.json exists."
                )

        self.label_to_idx = {lbl: i for i, lbl in enumerate(self.primary_labels)}
        self.n_classes = len(self.primary_labels)

        # Load taxonomy
        self.taxonomy_df = None
        self.family_to_idx = {}
        self.class_to_family = None
        if taxonomy_csv and os.path.exists(taxonomy_csv):
            self.taxonomy_df = pd.read_csv(taxonomy_csv)

        # Site/hour mappings (built after processing data)
        self.site_to_idx: Dict[str, int] = {}
        self.n_sites = 0

    # ------------------------------------------------------------------
    # Label matrix building
    # ------------------------------------------------------------------

    def build_label_matrix(
        self,
        soundscape_labels_csv: str,
        meta_df: pd.DataFrame,
        window_sec: int = 5,
    ) -> np.ndarray:
        """
        Build (N_windows, n_classes) binary label matrix from
        train_soundscapes_labels.csv aligned to meta_df row order.

        Args:
            soundscape_labels_csv: path to CSV with filename, start, end, primary_label
            meta_df: DataFrame with row_id, filename columns (from perch_meta.parquet)
            window_sec: window duration in seconds (5)
        Returns:
            y: (N, n_classes) float32 label matrix
        """
        labels_df = pd.read_csv(soundscape_labels_csv)

        # Build lookup: (filename_stem, end_sec) → set of species
        label_lookup: Dict[Tuple, np.ndarray] = {}
        for _, row in labels_df.iterrows():
            fname = os.path.splitext(os.path.basename(str(row['filename'])))[0]
            end_sec = int(row['end'])
            vec = np.zeros(self.n_classes, dtype=np.float32)
            species_list = str(row['primary_label']).split(';')
            for sp in species_list:
                sp = sp.strip()
                if sp in self.label_to_idx:
                    vec[self.label_to_idx[sp]] = 1.0
            key = (fname, end_sec)
            if key in label_lookup:
                label_lookup[key] = np.maximum(label_lookup[key], vec)
            else:
                label_lookup[key] = vec

        # Align to meta_df
        y = np.zeros((len(meta_df), self.n_classes), dtype=np.float32)
        for i, row in meta_df.iterrows():
            # row_id format: "{stem}_{end_sec}"
            row_id = str(row['row_id'])
            parts = row_id.rsplit('_', 1)
            if len(parts) == 2:
                stem, end_sec_str = parts[0], parts[1]
                try:
                    end_sec = int(end_sec_str)
                    key = (stem, end_sec)
                    if key in label_lookup:
                        y[i] = label_lookup[key]
                except ValueError:
                    pass

        return y

    # ------------------------------------------------------------------
    # Taxonomy groups for auxiliary head
    # ------------------------------------------------------------------

    def build_taxonomy_groups(self) -> Tuple[np.ndarray, int]:
        """
        Build family-level groupings for auxiliary prediction head.
        Returns:
            class_to_family: (n_classes,) int array mapping species → family index
            n_families: number of unique families
        """
        if self.taxonomy_df is None:
            self.class_to_family = np.zeros(self.n_classes, dtype=np.int64)
            return self.class_to_family, 1

        family_col = None
        for col in ('family', 'family_en', 'order'):
            if col in self.taxonomy_df.columns:
                family_col = col
                break

        if family_col is None:
            self.class_to_family = np.zeros(self.n_classes, dtype=np.int64)
            return self.class_to_family, 1

        families = sorted(self.taxonomy_df[family_col].dropna().unique().tolist())
        self.family_to_idx = {f: i for i, f in enumerate(families)}

        label_col = 'primary_label' if 'primary_label' in self.taxonomy_df.columns else self.taxonomy_df.columns[0]
        tax_lookup = dict(zip(
            self.taxonomy_df[label_col].astype(str),
            self.taxonomy_df[family_col].astype(str),
        ))

        self.class_to_family = np.zeros(self.n_classes, dtype=np.int64)
        for lbl, idx in self.label_to_idx.items():
            fam = tax_lookup.get(lbl, families[0])
            self.class_to_family[idx] = self.family_to_idx.get(fam, 0)

        return self.class_to_family, len(families)

    # ------------------------------------------------------------------
    # Perch → competition class mapping
    # ------------------------------------------------------------------

    def map_perch_indices(self, perch_label_csv: str) -> Dict[int, int]:
        """
        Map Perch class indices (981 classes) to competition class indices (234).

        Tries in order:
          1. Exact match on ebird_code / primary_label column
          2. Genus-level proxy (max over same-genus competition classes)

        Args:
            perch_label_csv: path to Perch's labels.csv file
        Returns:
            dict: {perch_idx: comp_idx}
        """
        if not os.path.exists(perch_label_csv):
            return {}

        perch_df = pd.read_csv(perch_label_csv)
        mapping: Dict[int, int] = {}

        for perch_idx, row in perch_df.iterrows():
            matched = False
            for col in ('ebird_code', 'primary_label', 'species_code', 'label'):
                if col in row.index:
                    val = str(row[col])
                    if val in self.label_to_idx:
                        mapping[perch_idx] = self.label_to_idx[val]
                        matched = True
                        break
            # Genus proxy: if no direct match, try matching first 6 chars (ebird genus)
            if not matched:
                for col in ('ebird_code', 'primary_label', 'species_code', 'label'):
                    if col in row.index:
                        prefix = str(row[col])[:6].lower()
                        candidates = [
                            (lbl, idx) for lbl, idx in self.label_to_idx.items()
                            if lbl.lower().startswith(prefix)
                        ]
                        if len(candidates) == 1:
                            mapping[perch_idx] = candidates[0][1]
                        break

        return mapping

    # ------------------------------------------------------------------
    # Site/hour metadata helpers
    # ------------------------------------------------------------------

    def build_site_index(self, meta_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert site strings and hours to integer indices.
        Builds self.site_to_idx mapping.

        Returns:
            site_ids: (N,) int array
            hours:    (N,) int array
        """
        unique_sites = sorted(meta_df['site'].unique().tolist())
        self.site_to_idx = {s: i for i, s in enumerate(unique_sites)}
        self.n_sites = len(unique_sites)

        site_ids = meta_df['site'].map(
            lambda s: self.site_to_idx.get(s, self.n_sites)
        ).to_numpy().astype(np.int64)

        hours = meta_df['hour_utc'].fillna(0).to_numpy().astype(np.int64)
        return site_ids, hours
