"""
pseudo_label.py — Generate pseudo-labels for unlabeled train_soundscapes.

Workflow:
  1. Load trained model (best_model.pt / final_model.pt)
  2. Find soundscape files NOT already in train_soundscapes_labels.csv
  3. Run inference on every 5-second window
  4. Keep windows with max(prob) >= PSEUDO_LABEL_MIN_CONF
  5. Write output/pseudo_labels.csv  (columns: filename, start, end, primary_label)

Usage:
    python pseudo_label.py
    python pseudo_label.py --threshold 0.25
    python pseudo_label.py --model_path output/checkpoint_epoch_5.pt --output output/pl_v2.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import AudioTransform, TestSoundscapeDataset
from src.models.model import get_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_unlabeled_files(soundscape_dir: str, labeled_csv: str) -> list:
    """Return .ogg filenames in soundscape_dir that are NOT in labeled_csv."""
    labeled_files = set(pd.read_csv(labeled_csv)['filename'].tolist())
    all_files = [f for f in os.listdir(soundscape_dir) if f.endswith('.ogg')]
    unlabeled = [f for f in all_files if f not in labeled_files]
    print(f"Soundscape files total: {len(all_files)} | labeled: {len(labeled_files)} | unlabeled: {len(unlabeled)}")
    return unlabeled


def build_row_ids(filenames: list, soundscape_dir: str, sample_rate: int = 32000) -> list:
    """
    Generate row_ids ('{stem}_{end_time}') for each 5-second window.
    Reads actual file duration via soundfile to avoid hardcoding 60 s.
    Falls back to 60 s if soundfile is unavailable.
    """
    try:
        import soundfile as sf
        def get_duration(path):
            info = sf.info(path)
            return info.duration
    except ImportError:
        def get_duration(path):
            return 60.0

    row_ids = []
    for fname in filenames:
        path = os.path.join(soundscape_dir, fname)
        duration = get_duration(path)
        stem = fname.replace('.ogg', '')
        for end_t in range(5, int(duration) + 1, 5):
            row_ids.append(f"{stem}_{end_t}")
    return row_ids


def run_inference(model, dataloader, device) -> tuple:
    """Returns (probs ndarray [N, C], row_ids list)."""
    model.eval()
    all_probs, all_ids = [], []
    with torch.no_grad():
        for inputs, row_ids in tqdm(dataloader, desc="Pseudo-labeling"):
            inputs = inputs.to(device)
            probs = torch.sigmoid(model(inputs)).cpu().numpy()
            all_probs.append(probs)
            all_ids.extend(row_ids)
    return np.vstack(all_probs), all_ids


def probs_to_csv(
    probs: np.ndarray,
    row_ids: list,
    label_list: list,
    threshold: float,
    min_conf: float,
    use_soft: bool,
    output_path: str,
) -> pd.DataFrame:
    """
    Convert probability matrix → pseudo-label CSV.
    Rows where max(prob) < min_conf are discarded.
    primary_label: ';'-separated species codes (same format as TrainSoundscapesDataset).
    """
    records = []
    for i, row_id in enumerate(row_ids):
        p = probs[i]
        if p.max() < min_conf:
            continue

        parts = row_id.rsplit('_', 1)
        filename = parts[0] + '.ogg'
        end_time = int(parts[1])
        start_time = end_time - 5

        if use_soft:
            active = [label_list[j] for j in range(len(label_list)) if p[j] >= threshold]
        else:
            active = [label_list[j] for j in np.where(p >= threshold)[0]]

        if not active:
            continue

        records.append({
            'filename': filename,
            'start': start_time,
            'end': end_time,
            'primary_label': ';'.join(active),
        })

    df = pd.DataFrame(records, columns=['filename', 'start', 'end', 'primary_label'])
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Pseudo-labels saved: {len(df)} segments → {output_path}")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Override PSEUDO_LABEL_THRESHOLD from config')
    parser.add_argument('--model_path', default=None,
                        help='Override model checkpoint path')
    parser.add_argument('--output', default=None,
                        help='Override output CSV path')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    threshold = args.threshold if args.threshold is not None else config.get('PSEUDO_LABEL_THRESHOLD', 0.3)
    min_conf = config.get('PSEUDO_LABEL_MIN_CONF', 0.3)
    use_soft = config.get('PSEUDO_LABEL_USE_SOFT', True)
    output_path = args.output or os.path.join(config['OUTPUT_DIR'], 'pseudo_labels.csv')

    # Resolve model path
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = None
        for fname in ('best_model.pt', 'final_model.pt'):
            candidate = os.path.join(config['OUTPUT_DIR'], fname)
            if os.path.exists(candidate):
                model_path = candidate
                break
    if model_path is None:
        raise FileNotFoundError(f"No trained model found in {config['OUTPUT_DIR']}. Run train.py first.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} | Model: {model_path}")
    print(f"Threshold: {threshold} | Min confidence: {min_conf} | Soft labels: {use_soft}")

    # Label list from sample_submission.csv (same as inference.py)
    submission_df = pd.read_csv(config['SAMPLE_SUBMISSION_CSV'])
    label_list = [col for col in submission_df.columns if col != 'row_id']
    num_classes = len(label_list)
    print(f"Classes: {num_classes}")

    # Load model
    model = get_model(
        num_classes=num_classes,
        backbone=config.get('BACKBONE', 'efficientnet_b0'),
        device=device,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Build row_ids for unlabeled soundscapes
    unlabeled_files = find_unlabeled_files(
        config['TRAIN_SOUNDSCAPES_DIR'],
        config['TRAIN_SOUNDSCAPES_LABELS'],
    )
    if not unlabeled_files:
        print("No unlabeled soundscape files found. Exiting.")
        return

    row_ids = build_row_ids(unlabeled_files, config['TRAIN_SOUNDSCAPES_DIR'], config['SAMPLE_RATE'])
    print(f"Total windows to process: {len(row_ids)}")

    # Write a temporary submission-format CSV so TestSoundscapeDataset can read it
    temp_sub_path = os.path.join(config['OUTPUT_DIR'], '_pseudo_temp_sub.csv')
    temp_sub = pd.DataFrame({'row_id': row_ids})
    for lbl in label_list:
        temp_sub[lbl] = 0.0
    temp_sub.to_csv(temp_sub_path, index=False)

    transform = AudioTransform(
        sample_rate=config['SAMPLE_RATE'],
        n_mels=config['N_MELS'],
        n_fft=config['N_FFT'],
        hop_length=config['HOP_LENGTH'],
        augment=False,
    )

    dataset = TestSoundscapeDataset(
        soundscape_dir=config['TRAIN_SOUNDSCAPES_DIR'],
        sample_submission_path=temp_sub_path,
        transform=transform,
        label_list=label_list,
    )
    loader = DataLoader(
        dataset,
        batch_size=config['BATCH_SIZE'],
        shuffle=False,
        num_workers=config.get('NUM_WORKERS', 2),
        pin_memory=True,
    )

    probs, result_row_ids = run_inference(model, loader, device)

    # Cleanup temp file
    os.remove(temp_sub_path)

    probs_to_csv(
        probs=probs,
        row_ids=result_row_ids,
        label_list=label_list,
        threshold=threshold,
        min_conf=min_conf,
        use_soft=use_soft,
        output_path=output_path,
    )

    print("\nDone! Next steps:")
    print(f"  1. Set PSEUDO_LABEL_CSV: \"{output_path}\" in config.yaml")
    print("  2. Run python train.py for the next iteration")


if __name__ == '__main__':
    main()
