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
from tqdm import tqdm


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
    Loads Google Perch v2 and extracts per-window embeddings.

    Supports two backends (tried in order):
      1. perch-hoplite  — official high-level API (pip install perch-hoplite)
      2. raw TF SavedModel — requires tensorflow >= 2.20.0

    Each audio file is split into N_WINDOWS × WINDOW_SEC segments.
    Output per file:
      embeddings : (N_WINDOWS, 1536) float32
      logits     : (N_WINDOWS, perch_n_classes) float32  [14795 for Perch v2]
    """

    def __init__(
        self,
        model_dir: str,
        sample_rate: int = 32000,
        window_sec: int = 5,
        n_windows: int = 12,
    ):
        self.sample_rate = sample_rate
        self.window_sec = window_sec
        self.n_windows = n_windows
        self.window_samples = sample_rate * window_sec
        self.total_samples = self.window_samples * n_windows
        self._backend = None

        # Try perch-hoplite first (official API, no StableHLO issues)
        try:
            from perch_hoplite.zoo import model_configs as _mc
            print("Loading Perch via perch-hoplite API...")
            # Correct API: get_model_class + from_config (load_model_by_name does not exist)
            model_class = _mc.get_model_class('taxonomy_model_tf')
            self._hoplite_model = model_class.from_config({
                'model_path': model_dir,
                'window_size_s': float(window_sec),
                'hop_size_s': float(window_sec),
                'sample_rate': sample_rate,
            })
            self._backend = 'hoplite'
            print("Perch loaded (hoplite backend).")
            return
        except Exception as e:
            print(f"perch-hoplite not available ({e}), falling back to raw SavedModel...")

        # Fallback: raw TF SavedModel (needs TF >= 2.20.0)
        try:
            import tensorflow as tf
            self._tf = tf
        except ImportError:
            raise ImportError(
                "Neither perch-hoplite nor tensorflow>=2.20.0 is available.\n"
                "Fix: pip install perch-hoplite  OR  pip install tensorflow-cpu==2.20.0"
            )

        # Use all available CPU cores for TF ops — critical for CPU-only inference speed
        tf.config.threading.set_inter_op_parallelism_threads(0)
        tf.config.threading.set_intra_op_parallelism_threads(0)

        if not model_dir:
            raise ValueError(
                "Set PERCH.MODEL_DIR in config.yaml to the Perch TF SavedModel path.\n"
                "On Kaggle, e.g.: /kaggle/input/birdclef2026-perch-model/perch_v2_cpu/1"
            )

        print(f"Loading Perch from {model_dir} ...")
        self._model = tf.saved_model.load(model_dir)
        self._infer = self._model.signatures["serving_default"]
        self._backend = 'tf_savedmodel'
        # Detect output key names from the signature
        self._emb_key = 'embedding' if 'embedding' in self._infer.structured_outputs else \
                        next(k for k in self._infer.structured_outputs if 'emb' in k.lower())
        self._logit_key = 'label' if 'label' in self._infer.structured_outputs else \
                          next(k for k in self._infer.structured_outputs if 'label' in k.lower()
                               or 'logit' in k.lower())
        print(f"Perch loaded (SavedModel backend). Keys: emb='{self._emb_key}', logit='{self._logit_key}'")

    def _load_audio(self, path: str) -> np.ndarray:
        """Load audio, resample to 32 kHz mono, pad/trim to total_samples."""
        try:
            import soundfile as sf
            audio, sr = sf.read(path, dtype='float32')
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
        except Exception:
            import librosa
            audio, sr = librosa.load(path, sr=self.sample_rate, mono=True)
            return _pad_trim(audio, self.total_samples)

        if sr != self.sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

        return _pad_trim(audio, self.total_samples)

    def extract_file(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract embeddings and raw logits from a single audio file.
        Returns:
            embeddings : (n_windows, 1536)
            logits     : (n_windows, perch_n_classes)  — 14795 for Perch v2
        """
        audio = self._load_audio(path)

        if self._backend == 'hoplite':
            # perch-hoplite processes one 5s window at a time.
            # embed() returns ModelOutput with .embeddings shape (frames, 1536)
            # and .logits['label'] shape (frames, n_classes) — frames=1 for a 5s input.
            windows = audio.reshape(self.n_windows, self.window_samples)
            emb_list, logit_list = [], []
            for w in windows:
                out = self._hoplite_model.embed(w)
                emb = np.array(out.embeddings)          # (1, 1536) or (1536,)
                logit = np.array(out.logits['label'])   # (1, n_classes) or (n_classes,)
                # Squeeze leading frames dimension if present
                emb_list.append(emb[0] if emb.ndim > 1 else emb)
                logit_list.append(logit[0] if logit.ndim > 1 else logit)
            return np.stack(emb_list), np.stack(logit_list)

        # Raw SavedModel backend
        tf = self._tf
        windows = audio.reshape(self.n_windows, self.window_samples)
        batch = tf.convert_to_tensor(windows, dtype=tf.float32)
        out = self._infer(inputs=batch)

        embeddings = out[self._emb_key].numpy().astype(np.float32)  # (T, 1536)
        logits = out[self._logit_key].numpy().astype(np.float32)    # (T, n_classes)
        return embeddings, logits

    def _infer_batch(self, windows_batch: np.ndarray):
        """
        Run SavedModel inference on a stacked batch of windows.
        Args:
            windows_batch: (N_total_windows, window_samples)
        Returns:
            embeddings: (N_total_windows, 1536)
            logits:     (N_total_windows, perch_n_classes)
        """
        tf = self._tf
        batch = tf.convert_to_tensor(windows_batch, dtype=tf.float32)
        out = self._infer(inputs=batch)
        return (
            out[self._emb_key].numpy().astype(np.float32),
            out[self._logit_key].numpy().astype(np.float32),
        )

    def extract_and_cache(
        self,
        audio_paths: List[str],
        cache_dir: str,
        label_list: List[str],
        taxonomy_csv: Optional[str] = None,
        perch_label_csv: Optional[str] = None,
        batch_files: int = 16,
        resume: bool = False,
        checkpoint_every: int = 500,
    ) -> None:
        """
        Extract all files, map Perch logits to competition classes, save cache.

        Args:
            audio_paths      : list of .ogg file paths
            cache_dir        : output directory
            label_list       : competition label list (234 species)
            taxonomy_csv     : path to taxonomy.csv for class mapping
            perch_label_csv  : path to Perch's labels.csv (981 classes)
            batch_files      : files per SavedModel call (ignored for hoplite backend)
            resume           : if True, skip files already in checkpoint and append
            checkpoint_every : save partial results every N processed files
        """
        os.makedirs(cache_dir, exist_ok=True)

        # Build Perch → competition class mapping
        perch_to_comp = _build_class_mapping(label_list, taxonomy_csv, perch_label_csv)
        n_comp = len(label_list)

        all_emb: list = []
        all_scores: list = []
        all_row_ids: list = []
        all_filenames: list = []
        all_sites: list = []
        all_hours: list = []

        # Resume: load existing checkpoint and skip already-done files
        done_files: set = set()
        arrays_path = os.path.join(cache_dir, 'perch_arrays.npz')
        meta_path   = os.path.join(cache_dir, 'perch_meta.parquet')
        if resume and os.path.exists(arrays_path) and os.path.exists(meta_path):
            print("Resuming from checkpoint...")
            ckpt = np.load(arrays_path)
            ckpt_meta = pd.read_parquet(meta_path)
            all_emb    = list(ckpt['emb_full'].reshape(-1, self.n_windows, 1536))
            all_scores = list(ckpt['scores_full_raw'].reshape(-1, self.n_windows, n_comp))
            all_row_ids   = ckpt_meta['row_id'].tolist()
            all_filenames = ckpt_meta['filename'].tolist()
            all_sites     = ckpt_meta['site'].tolist()
            all_hours     = ckpt_meta['hour_utc'].tolist()
            done_files = set(ckpt_meta['filename'].unique())
            print(f"  Loaded {len(done_files)} files from checkpoint, resuming...")

        audio_paths = [p for p in audio_paths if os.path.basename(p) not in done_files]

        # hoplite processes windows one at a time internally; no cross-file batching needed
        use_batching = (self._backend == 'tf_savedmodel') and (batch_files > 1)

        skipped = 0
        pbar = tqdm(total=len(audio_paths), desc="Extracting", unit="file")

        i = 0
        while i < len(audio_paths):
            # ----------------------------------------------------------------
            # Batched SavedModel path: load N files, run one _infer() call
            # ----------------------------------------------------------------
            if use_batching:
                chunk = audio_paths[i: i + batch_files]
                windows_list, metas, valid_paths = [], [], []

                for path in chunk:
                    try:
                        audio = self._load_audio(path)
                        windows_list.append(audio.reshape(self.n_windows, self.window_samples))
                        metas.append((path, parse_filename_metadata(os.path.basename(path))))
                        valid_paths.append(path)
                    except Exception as e:
                        skipped += 1
                        tqdm.write(f"  SKIP (load) {os.path.basename(path)}: {e}")

                if windows_list:
                    try:
                        stacked = np.vstack(windows_list)          # (N_files*12, samples)
                        emb_all, logit_all = self._infer_batch(stacked)
                    except Exception as e:
                        skipped += len(windows_list)
                        tqdm.write(f"  SKIP batch of {len(windows_list)} files: {e}")
                        i += len(chunk)
                        pbar.update(len(chunk))
                        pbar.set_postfix(skipped=skipped, windows=len(all_row_ids))
                        continue

                    for j, (path, meta) in enumerate(metas):
                        fname = os.path.basename(path)
                        stem = os.path.splitext(fname)[0]
                        emb = emb_all[j * self.n_windows: (j + 1) * self.n_windows]
                        raw_logits = logit_all[j * self.n_windows: (j + 1) * self.n_windows]
                        comp_scores = _map_logits(raw_logits, perch_to_comp, n_comp)
                        for t in range(self.n_windows):
                            all_row_ids.append(f"{stem}_{(t + 1) * self.window_sec}")
                            all_filenames.append(fname)
                            all_sites.append(meta['site'])
                            all_hours.append(meta['hour_utc'])
                        all_emb.append(emb)
                        all_scores.append(comp_scores)

                i += len(chunk)
                pbar.update(len(chunk))
                pbar.set_postfix(skipped=skipped, windows=len(all_row_ids))
                _maybe_checkpoint(all_emb, all_scores, all_row_ids, all_filenames,
                                  all_sites, all_hours, arrays_path, meta_path,
                                  checkpoint_every, self.n_windows, n_comp)
                continue

            # ----------------------------------------------------------------
            # Per-file path (hoplite backend or batch_files=1)
            # ----------------------------------------------------------------
            path = audio_paths[i]
            fname = os.path.basename(path)
            stem = os.path.splitext(fname)[0]
            meta = parse_filename_metadata(fname)

            try:
                emb, raw_logits = self.extract_file(path)
            except Exception as e:
                skipped += 1
                tqdm.write(f"  SKIP {fname}: {e}")
                i += 1
                pbar.update(1)
                pbar.set_postfix(skipped=skipped, windows=len(all_row_ids))
                continue

            comp_scores = _map_logits(raw_logits, perch_to_comp, n_comp)
            for t in range(self.n_windows):
                all_row_ids.append(f"{stem}_{(t + 1) * self.window_sec}")
                all_filenames.append(fname)
                all_sites.append(meta['site'])
                all_hours.append(meta['hour_utc'])
            all_emb.append(emb)
            all_scores.append(comp_scores)

            i += 1
            pbar.update(1)
            pbar.set_postfix(skipped=skipped, windows=len(all_row_ids))
            _maybe_checkpoint(all_emb, all_scores, all_row_ids, all_filenames,
                              all_sites, all_hours, arrays_path, meta_path,
                              checkpoint_every, self.n_windows, n_comp)

        pbar.close()

        if not all_emb:
            raise RuntimeError(
                "All files failed to process. "
                "Most likely cause: TF version too old for this Perch model.\n"
                "Fix: pip install tensorflow-cpu==2.20.0  (then restart kernel)"
            )

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
# Checkpoint helper
# ---------------------------------------------------------------------------

def _maybe_checkpoint(all_emb, all_scores, all_row_ids, all_filenames,
                      all_sites, all_hours, arrays_path, meta_path,
                      checkpoint_every, n_windows, n_comp):
    """Save partial results every checkpoint_every files."""
    n_files = len(all_emb)
    if n_files == 0 or n_files % checkpoint_every != 0:
        return
    emb_full    = np.vstack(all_emb).astype(np.float32)
    scores_full = np.vstack(all_scores).astype(np.float32)
    np.savez_compressed(arrays_path, emb_full=emb_full, scores_full_raw=scores_full)
    pd.DataFrame({
        'row_id': all_row_ids, 'filename': all_filenames,
        'site': all_sites, 'hour_utc': all_hours,
    }).to_parquet(meta_path, index=False)
    tqdm.write(f"  [checkpoint] saved {n_files} files → {arrays_path}")


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
    parser.add_argument('--shard', type=str, default=None,
                        help="Process a fraction of files, e.g. '0/3' = first third, '1/3' = second third.")
    parser.add_argument('--resume', action='store_true',
                        help="Skip files already saved in checkpoint; append new results.")
    parser.add_argument('--checkpoint-every', type=int, default=500,
                        help="Save partial results every N files (default 500).")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    perch_cfg = config.get('PERCH', {})
    model_dir = perch_cfg.get('MODEL_DIR', '')
    # Note: model_dir is needed for both backends (hoplite uses it as model_path;
    # SavedModel backend loads it directly). Only raise after construction fails.

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

    # Shard: split work across multiple notebook instances
    # e.g. --shard 0/3 --shard 1/3 --shard 2/3 in three separate notebooks
    if args.shard:
        shard_idx, n_shards = (int(x) for x in args.shard.split('/'))
        audio_paths = audio_paths[shard_idx::n_shards]
        print(f"Shard {shard_idx}/{n_shards}: processing {len(audio_paths)} files")
    else:
        print(f"Found {len(audio_paths)} .ogg files in {audio_dir}")

    # Load label list — always prefer sample_submission.csv (234 classes).
    # train.csv only has 206 classes; the remaining 28 appear only in soundscapes.
    label_list_path = os.path.join(config.get('OUTPUT_DIR', 'output'), 'label_list.json')
    if os.path.exists(config.get('SAMPLE_SUBMISSION_CSV', '')):
        sub_df = pd.read_csv(config['SAMPLE_SUBMISSION_CSV'])
        label_list = [c for c in sub_df.columns if c != 'row_id']
    elif os.path.exists(label_list_path):
        with open(label_list_path) as f:
            label_list = json.load(f)
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
        batch_files=perch_cfg.get('BATCH_FILES', 16),
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
    )
    print("Done.")


if __name__ == '__main__':
    main()
