"""
extract_perch.py — Extract and cache Google Perch embeddings from audio files.

Perch is Google's pretrained bird vocalization classifier (1536-dim embeddings).
This script runs Perch once offline and caches the results so that ProtoSSM
training does not require reloading the TF model.

Output files in CACHE_DIR:
  - perch_meta.parquet   : row_id, filename, site, hour_utc per window
  - perch_arrays.npz     : emb_full (N, 1536), scores_full_raw (N, n_classes)

Usage:
    python extract_perch.py --source train_soundscapes
    python extract_perch.py --source train_audio
    python extract_perch.py --source test_soundscapes --config config.yaml
"""

import argparse
import os

# Disable XLA JIT to avoid StableHLO version mismatch with Perch v2 SavedModel
os.environ.setdefault('TF_XLA_FLAGS', '--tf_xla_auto_jit=0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')   # suppress TF C++ warnings

import re
import json
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Filename metadata parsing
# ---------------------------------------------------------------------------

_FNAME_RE = re.compile(
    r'BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg',
    re.IGNORECASE,
)


def parse_filename_metadata(fname: str) -> dict:
    """
    Parse site and hour from BirdCLEF soundscape filename.
    Example: BC2026_Test_0001_S05_20250227_010002.ogg
    Returns: {'site': 'S05', 'hour_utc': 1}
    """
    m = _FNAME_RE.search(os.path.basename(fname))
    if m:
        time_str = m.group(4)          # '010002' → HH MM SS
        hour = int(time_str[:2])
        return {'site': m.group(2), 'hour_utc': hour}
    return {'site': 'UNKNOWN', 'hour_utc': 0}


# ---------------------------------------------------------------------------
# PerchExtractor
# ---------------------------------------------------------------------------

class PerchExtractor:
    """
    Loads Google Perch TF SavedModel and extracts per-window embeddings.

    Each audio file is split into N_WINDOWS × WINDOW_SEC segments.
    Output per file:
      embeddings : (N_WINDOWS, 1536) float32
      logits     : (N_WINDOWS, perch_n_classes) float32
    """

    def __init__(
        self,
        model_dir: str,
        sample_rate: int = 32000,
        window_sec: int = 5,
        n_windows: int = 12,
    ):
        try:
            import tensorflow as tf
            self._tf = tf
        except ImportError:
            raise ImportError(
                "tensorflow is required for Perch extraction. "
                "Install with: pip install tensorflow"
            )

        print(f"Loading Perch from {model_dir} ...")
        self._model = tf.saved_model.load(model_dir)
        self._infer = self._model.signatures["serving_default"]
        self.sample_rate = sample_rate
        self.window_sec = window_sec
        self.n_windows = n_windows
        self.window_samples = sample_rate * window_sec
        self.total_samples = self.window_samples * n_windows
        print("Perch loaded.")

    def _load_audio(self, path: str) -> np.ndarray:
        """Load audio, resample to target SR, pad/trim to total_samples."""
        try:
            import soundfile as sf
            audio, sr = sf.read(path, dtype='float32')
        except ImportError:
            import librosa
            audio, sr = librosa.load(path, sr=self.sample_rate, mono=True)
            return _pad_trim(audio, self.total_samples)

        # Convert to mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample if needed
        if sr != self.sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

        return _pad_trim(audio, self.total_samples)

    def extract_file(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract embeddings and raw logits from a single audio file.
        Returns:
            embeddings : (n_windows, 1536)
            logits     : (n_windows, perch_n_classes)
        """
        tf = self._tf
        audio = self._load_audio(path)
        windows = audio.reshape(self.n_windows, self.window_samples)  # (T, window_samples)

        batch = tf.convert_to_tensor(windows, dtype=tf.float32)
        out = self._infer(inputs=batch)

        embeddings = out['embedding'].numpy().astype(np.float32)   # (T, 1536)
        logits = out['label'].numpy().astype(np.float32)           # (T, perch_classes)
        return embeddings, logits

    def extract_and_cache(
        self,
        audio_paths: List[str],
        cache_dir: str,
        label_list: List[str],
        taxonomy_csv: Optional[str] = None,
        perch_label_csv: Optional[str] = None,
    ) -> None:
        """
        Extract all files, map Perch logits to competition classes, save cache.

        Args:
            audio_paths   : list of .ogg file paths
            cache_dir     : output directory
            label_list    : competition label list (234 species)
            taxonomy_csv  : path to taxonomy.csv for class mapping
            perch_label_csv : path to Perch's labels.csv (981 classes)
        """
        os.makedirs(cache_dir, exist_ok=True)

        # Build Perch → competition class mapping
        perch_to_comp = _build_class_mapping(label_list, taxonomy_csv, perch_label_csv)
        n_comp = len(label_list)

        all_emb = []
        all_scores = []
        all_row_ids = []
        all_filenames = []
        all_sites = []
        all_hours = []

        print(f"Extracting {len(audio_paths)} files...")
        for i, path in enumerate(audio_paths):
            fname = os.path.basename(path)
            stem = os.path.splitext(fname)[0]
            meta = parse_filename_metadata(fname)

            try:
                emb, raw_logits = self.extract_file(path)         # (T, 1536), (T, perch_classes)
            except Exception as e:
                print(f"  SKIP {fname}: {e}")
                continue

            # Map Perch logits to competition classes
            comp_scores = _map_logits(raw_logits, perch_to_comp, n_comp)  # (T, n_comp)

            T = emb.shape[0]
            for t in range(T):
                end_sec = (t + 1) * self.window_sec
                row_id = f"{stem}_{end_sec}"
                all_row_ids.append(row_id)
                all_filenames.append(fname)
                all_sites.append(meta['site'])
                all_hours.append(meta['hour_utc'])

            all_emb.append(emb)
            all_scores.append(comp_scores)

            if (i + 1) % 50 == 0 or (i + 1) == len(audio_paths):
                print(f"  [{i+1}/{len(audio_paths)}] {fname}")

        emb_full = np.vstack(all_emb).astype(np.float32)          # (N, 1536)
        scores_full = np.vstack(all_scores).astype(np.float32)    # (N, n_comp)

        # Save arrays
        arrays_path = os.path.join(cache_dir, 'perch_arrays.npz')
        np.savez_compressed(arrays_path, emb_full=emb_full, scores_full_raw=scores_full)
        print(f"Saved arrays → {arrays_path}  shape: emb={emb_full.shape}, scores={scores_full.shape}")

        # Save metadata
        meta_df = pd.DataFrame({
            'row_id': all_row_ids,
            'filename': all_filenames,
            'site': all_sites,
            'hour_utc': all_hours,
        })
        meta_path = os.path.join(cache_dir, 'perch_meta.parquet')
        meta_df.to_parquet(meta_path, index=False)
        print(f"Saved metadata → {meta_path}  rows: {len(meta_df)}")


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _pad_trim(audio: np.ndarray, target_len: int) -> np.ndarray:
    if len(audio) >= target_len:
        return audio[:target_len]
    return np.pad(audio, (0, target_len - len(audio)))


def _build_class_mapping(
    label_list: List[str],
    taxonomy_csv: Optional[str],
    perch_label_csv: Optional[str],
) -> dict:
    """
    Build mapping from Perch class index → competition class index.
    Returns dict: {perch_idx: comp_idx}
    Falls back to empty mapping if Perch label CSV is not provided.
    """
    if perch_label_csv is None or not os.path.exists(perch_label_csv):
        return {}

    perch_df = pd.read_csv(perch_label_csv)
    label_to_idx = {lbl: i for i, lbl in enumerate(label_list)}
    mapping = {}

    for perch_idx, row in perch_df.iterrows():
        # Try direct match on ebird_code or scientific_name columns
        for col in ('ebird_code', 'primary_label', 'species_code', 'inat2024_fsd50k'):
            if col in row and row[col] in label_to_idx:
                mapping[perch_idx] = label_to_idx[row[col]]
                break

    return mapping


def _map_logits(
    raw_logits: np.ndarray,
    perch_to_comp: dict,
    n_comp: int,
) -> np.ndarray:
    """
    Map Perch logits (T, perch_classes) → competition logits (T, n_comp).
    Unmapped classes get -8.0 (effectively zero probability after sigmoid).
    """
    T = raw_logits.shape[0]
    comp = np.full((T, n_comp), -8.0, dtype=np.float32)
    for perch_idx, comp_idx in perch_to_comp.items():
        if perch_idx < raw_logits.shape[1]:
            comp[:, comp_idx] = raw_logits[:, perch_idx]
    return comp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract Perch embeddings from audio files.")
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument(
        '--source',
        choices=['train_soundscapes', 'train_audio', 'test_soundscapes'],
        default='train_soundscapes',
        help="Which audio source to process.",
    )
    parser.add_argument('--limit', type=int, default=None, help="Process only first N files (for testing).")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    perch_cfg = config.get('PERCH', {})
    model_dir = perch_cfg.get('MODEL_DIR', '')
    if not model_dir:
        raise ValueError("Set PERCH.MODEL_DIR in config.yaml to the Perch TF SavedModel path.")

    cache_dir = perch_cfg.get('CACHE_DIR', 'output/perch_cache')
    cache_dir = os.path.join(cache_dir, args.source)

    # Find audio files
    source_key_map = {
        'train_soundscapes': 'TRAIN_SOUNDSCAPES_DIR',
        'train_audio': 'TRAIN_AUDIO_DIR',
        'test_soundscapes': 'TEST_SOUNDSCAPES_DIR',
    }
    audio_dir = config.get(source_key_map[args.source], '')
    if not audio_dir or not os.path.isdir(audio_dir):
        raise FileNotFoundError(f"Directory not found: {audio_dir}. Check config.yaml.")

    audio_paths = sorted(Path(audio_dir).glob('**/*.ogg'))
    audio_paths = [str(p) for p in audio_paths]
    if args.limit:
        audio_paths = audio_paths[:args.limit]
    print(f"Found {len(audio_paths)} .ogg files in {audio_dir}")

    # Load label list
    label_list_path = os.path.join(config.get('OUTPUT_DIR', 'output'), 'label_list.json')
    if os.path.exists(label_list_path):
        with open(label_list_path) as f:
            label_list = json.load(f)
    elif os.path.exists(config.get('TRAIN_CSV', '')):
        train_df = pd.read_csv(config['TRAIN_CSV'])
        label_list = sorted(train_df['primary_label'].unique().tolist())
    else:
        sub_df = pd.read_csv(config['SAMPLE_SUBMISSION_CSV'])
        label_list = [c for c in sub_df.columns if c != 'row_id']
    print(f"Label list: {len(label_list)} classes")

    extractor = PerchExtractor(
        model_dir=model_dir,
        sample_rate=config.get('SAMPLE_RATE', 32000),
        window_sec=perch_cfg.get('WINDOW_SEC', 5),
        n_windows=perch_cfg.get('N_WINDOWS', 12),
    )

    extractor.extract_and_cache(
        audio_paths=audio_paths,
        cache_dir=cache_dir,
        label_list=label_list,
    )
    print("Done.")


if __name__ == '__main__':
    main()
