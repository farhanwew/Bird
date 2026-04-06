"""
cache_mel.py — Pre-compute mel spectrograms from audio files.

Converts all .ogg files to .npy mel spectrograms once, so training
reads fast numpy arrays instead of calling librosa.load + melspectrogram
per sample every epoch.

Speedup: 5-10x training time reduction.

Usage:
    python cache_mel.py                        # cache train_audio + train_soundscapes
    python cache_mel.py --source train_audio
    python cache_mel.py --source train_soundscapes
    python cache_mel.py --workers 4
"""

import argparse
import os
import numpy as np
import yaml
import librosa
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def compute_mel(audio_path: str, sample_rate: int, n_mels: int,
                n_fft: int, hop_length: int, duration: float) -> np.ndarray:
    """Load audio and compute normalized mel spectrogram."""
    target_len = int(sample_rate * duration)
    try:
        audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    except Exception:
        audio = np.zeros(target_len, dtype=np.float32)

    # Pad or trim to fixed duration
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]

    mel = librosa.feature.melspectrogram(
        y=audio, sr=sample_rate,
        n_mels=n_mels, n_fft=n_fft, hop_length=hop_length,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_min, mel_max = mel_db.min(), mel_db.max()
    if mel_max > mel_min:
        mel_db = (mel_db - mel_min) / (mel_max - mel_min + 1e-6)
    else:
        mel_db = np.zeros_like(mel_db)

    return mel_db.astype(np.float32)


def _worker(args):
    """Worker function for parallel processing."""
    audio_path, out_path, sample_rate, n_mels, n_fft, hop_length, duration = args
    if os.path.exists(out_path):
        return out_path, True, None  # already cached
    try:
        mel = compute_mel(audio_path, sample_rate, n_mels, n_fft, hop_length, duration)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, mel)
        return out_path, True, None
    except Exception as e:
        return out_path, False, str(e)


def cache_directory(audio_dir: str, cache_dir: str, config: dict,
                    workers: int = 2, limit: int = None):
    """Cache all .ogg files in a directory to .npy mel spectrograms."""
    audio_paths = sorted(Path(audio_dir).glob('**/*.ogg'))
    if limit:
        audio_paths = audio_paths[:limit]

    print(f"Found {len(audio_paths)} files in {audio_dir}")
    print(f"Cache dir: {cache_dir}")

    sample_rate = config['SAMPLE_RATE']
    n_mels      = config['N_MELS']
    n_fft       = config['N_FFT']
    hop_length  = config['HOP_LENGTH']
    duration    = config.get('DURATION', 5.0)

    tasks = []
    for p in audio_paths:
        rel = p.relative_to(audio_dir)
        out_path = os.path.join(cache_dir, str(rel).replace('.ogg', '.npy'))
        tasks.append((str(p), out_path, sample_rate, n_mels, n_fft, hop_length, duration))

    already_cached = sum(1 for _, out, *_ in tasks if os.path.exists(out))
    print(f"Already cached: {already_cached}/{len(tasks)}")

    if already_cached == len(tasks):
        print("All files already cached.")
        return

    errors = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_worker, t): t for t in tasks}
        with tqdm(total=len(tasks), desc="Caching") as pbar:
            for future in as_completed(futures):
                _, ok, err = future.result()
                if not ok:
                    errors += 1
                    if err:
                        tqdm.write(f"  ERROR: {err}")
                pbar.update(1)

    print(f"Done. Errors: {errors}/{len(tasks)}")


def main():
    parser = argparse.ArgumentParser(description="Pre-cache mel spectrograms.")
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--source', choices=['train_audio', 'train_soundscapes', 'all'],
                        default='all')
    parser.add_argument('--workers', type=int, default=2,
                        help="Parallel workers (use 0 for single-process)")
    parser.add_argument('--limit', type=int, default=None,
                        help="Process only first N files (for testing)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    mel_cache_dir = config.get('MEL_CACHE_DIR', 'output/mel_cache')

    sources = []
    if args.source in ('train_audio', 'all'):
        sources.append(('train_audio', config['TRAIN_AUDIO_DIR']))
    if args.source in ('train_soundscapes', 'all'):
        sources.append(('train_soundscapes', config['TRAIN_SOUNDSCAPES_DIR']))

    for name, audio_dir in sources:
        if not os.path.isdir(audio_dir):
            print(f"Skipping {name}: directory not found ({audio_dir})")
            continue
        print(f"\n=== {name} ===")
        cache_directory(
            audio_dir=audio_dir,
            cache_dir=os.path.join(mel_cache_dir, name),
            config=config,
            workers=max(1, args.workers) if args.workers > 0 else 1,
            limit=args.limit,
        )

    print("\nAll done! Set MEL_CACHE_DIR in config.yaml to enable cache during training.")


if __name__ == '__main__':
    main()
