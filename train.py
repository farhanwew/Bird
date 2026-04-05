import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import pandas as pd
import numpy as np
from tqdm import tqdm
import yaml

from src.data.dataset import TrainAudioDataset, AudioTransform
from src.data.dataset_soundscapes import TrainSoundscapesDataset
from src.models.model import get_model


def mixup_batch(inputs, labels, alpha):
    """Mixup augmentation at batch level. alpha=0 disables."""
    if alpha <= 0:
        return inputs, labels
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(inputs.size(0), device=inputs.device)
    return (
        lam * inputs + (1 - lam) * inputs[idx],
        lam * labels + (1 - lam) * labels[idx],
    )


def train_one_epoch(model, dataloader, criterion, optimizer, device, mixup_alpha=0.0):
    model.train()
    running_loss = 0.0

    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        inputs, labels = mixup_batch(inputs, labels, mixup_alpha)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            all_probs.append(torch.sigmoid(outputs).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)

    # Binarize labels (secondary labels may have value 0.5)
    binary_labels = (all_labels > 0).astype(np.float32)

    # Only score classes that have at least one positive sample in val set
    valid_cols = binary_labels.sum(axis=0) > 0
    if valid_cols.sum() == 0:
        roc_auc = 0.0
    else:
        roc_auc = roc_auc_score(
            binary_labels[:, valid_cols],
            all_probs[:, valid_cols],
            average='macro',
        )

    return running_loss / len(dataloader), roc_auc


def build_optimizer(model, config):
    lr = config['LEARNING_RATE']
    wd = config.get('WEIGHT_DECAY', 0.0)
    opt_name = config.get('OPTIMIZER', 'adam').lower()
    if opt_name == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


def build_scheduler(optimizer, config):
    sched_name = config.get('SCHEDULER', 'step').lower()
    epochs = config['MAX_EPOCHS']
    if sched_name == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    return optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)


def build_combined_dataset(config, train_csv_path, label_list, train_transform):
    """Combines TrainAudioDataset + optionally TrainSoundscapesDataset.
    Returns (dataset, per_sample_weights).
    """
    shared_kwargs = dict(
        audio_dir=config['TRAIN_AUDIO_DIR'],
        label_list=label_list,
        use_secondary_labels=config.get('USE_SECONDARY_LABELS', False),
        secondary_label_weight=config.get('SECONDARY_LABEL_WEIGHT', 0.5),
        duration=config.get('DURATION', 5.0),
        sample_rate=config.get('SAMPLE_RATE', 32000),
    )

    audio_dataset = TrainAudioDataset(
        csv_path=train_csv_path, transform=train_transform, **shared_kwargs
    )

    datasets = [audio_dataset]
    weights = [1.0] * len(audio_dataset)

    if config.get('USE_TRAIN_SOUNDSCAPES', False):
        soundscape_dataset = TrainSoundscapesDataset(
            soundscape_dir=config['TRAIN_SOUNDSCAPES_DIR'],
            labels_csv=config['TRAIN_SOUNDSCAPES_LABELS'],
            transform=train_transform,
            label_list=label_list,
        )
        datasets.append(soundscape_dataset)
        oversample = config.get('SOUNDSCAPE_OVERSAMPLE', 1.0)
        weights += [oversample] * len(soundscape_dataset)
        print(f"  + TrainSoundscapesDataset: {len(soundscape_dataset)} samples (oversample x{oversample})")

    pseudo_csv = config.get('PSEUDO_LABEL_CSV', '')
    if pseudo_csv and os.path.exists(pseudo_csv):
        pseudo_dataset = TrainSoundscapesDataset(
            soundscape_dir=config['TRAIN_SOUNDSCAPES_DIR'],
            labels_csv=pseudo_csv,
            transform=train_transform,
            label_list=label_list,
        )
        datasets.append(pseudo_dataset)
        pseudo_weight = config.get('PSEUDO_LABEL_WEIGHT', 0.5)
        weights += [pseudo_weight] * len(pseudo_dataset)
        print(f"  + Pseudo-labels: {len(pseudo_dataset)} samples (weight x{pseudo_weight}) from {pseudo_csv}")
    elif pseudo_csv:
        print(f"  ! PSEUDO_LABEL_CSV set but not found: {pseudo_csv}")

    combined = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    return combined, weights


def _extract_labels_from_dataset(dataset, label_to_idx, use_secondary, secondary_weight):
    """Extract label vectors from dataset metadata (no audio loading)."""
    from torch.utils.data import ConcatDataset as _ConcatDataset

    if isinstance(dataset, _ConcatDataset):
        parts = []
        for ds in dataset.datasets:
            parts.append(_extract_labels_from_dataset(ds, label_to_idx, use_secondary, secondary_weight))
        return np.vstack(parts)

    n = len(label_to_idx)

    # TrainAudioDataset
    if hasattr(dataset, 'df'):
        df = dataset.df
        mat = np.zeros((len(df), n), dtype=np.float32)
        for i, row in enumerate(df.itertuples(index=False)):
            idx = label_to_idx.get(row.primary_label, None)
            if idx is not None:
                mat[i, idx] = 1.0
            if use_secondary:
                raw = str(getattr(row, 'secondary_labels', '') or '')
                raw = raw.strip("[]").replace("'", "").replace('"', '')
                for sec in raw.split(','):
                    sec = sec.strip()
                    if sec and sec in label_to_idx:
                        mat[i, label_to_idx[sec]] = max(mat[i, label_to_idx[sec]], secondary_weight)
        return mat

    # TrainSoundscapesDataset
    if hasattr(dataset, 'labels_df'):
        df = dataset.labels_df
        mat = np.zeros((len(df), n), dtype=np.float32)
        for i, row in enumerate(df.itertuples(index=False)):
            raw = str(getattr(row, 'primary_label', '') or '')
            for lbl in raw.split(';'):
                lbl = lbl.strip()
                if lbl and lbl in label_to_idx:
                    mat[i, label_to_idx[lbl]] = 1.0
        return mat

    # Fallback: unknown dataset type, return uniform weights
    return np.ones((len(dataset), n), dtype=np.float32)


def compute_sample_weights(dataset, label_list, use_secondary=False, secondary_weight=0.5):
    """Fast class-frequency inversion using DataFrame metadata (no audio loading).
    Each sample gets weight = max(1/class_count) over its active classes.
    """
    print("Computing sample weights for class balancing (fast, no audio loading)...")
    label_to_idx = {lbl: i for i, lbl in enumerate(label_list)}
    label_matrix = _extract_labels_from_dataset(dataset, label_to_idx, use_secondary, secondary_weight)

    class_counts = label_matrix.sum(axis=0) + 1e-6
    class_weights = 1.0 / class_counts
    sample_weights = (label_matrix * class_weights).max(axis=1)
    print(f"  Sample weights computed for {len(sample_weights)} samples across {len(label_list)} classes")
    return sample_weights


def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cpu')
    if torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()
            device = torch.device('cuda')
        except Exception as e:
            print(f"Warning: CUDA available but not usable ({e}), falling back to CPU.")
    print(f"Using device: {device}")

    os.makedirs(config['OUTPUT_DIR'], exist_ok=True)

    augment_cfg = {
        'freq_mask_max': config.get('FREQ_MASK_MAX', 27),
        'freq_mask_num': config.get('FREQ_MASK_NUM', 2),
        'time_mask_max': config.get('TIME_MASK_MAX', 40),
        'time_mask_num': config.get('TIME_MASK_NUM', 2),
        'brightness_factor': config.get('BRIGHTNESS_FACTOR', 0.2),
        'prob': config.get('AUGMENT_PROB', 0.5),
    }
    mel_kwargs = dict(
        sample_rate=config['SAMPLE_RATE'],
        n_mels=config['N_MELS'],
        n_fft=config['N_FFT'],
        hop_length=config['HOP_LENGTH'],
    )
    do_augment = config.get('AUGMENT', False)
    train_transform = AudioTransform(
        **mel_kwargs,
        augment=do_augment,
        spec_augment_cfg=augment_cfg if do_augment else None,
    )
    val_transform = AudioTransform(**mel_kwargs, augment=False)

    train_df = pd.read_csv(config['TRAIN_CSV'])
    label_list = sorted(train_df['primary_label'].unique().tolist())
    num_classes = len(label_list)
    print(f"Number of classes: {num_classes}")

    # Save label_list so inference.py uses the exact same classes
    label_list_path = os.path.join(config['OUTPUT_DIR'], 'label_list.json')
    import json
    with open(label_list_path, 'w') as f:
        json.dump(label_list, f)
    print(f"Label list saved → {label_list_path}")

    # Train / val split
    val_frac = config.get('VAL_SPLIT', 0.0)
    seed = config.get('VAL_SEED', 42)

    if val_frac > 0:
        # Stratify by primary_label, but classes with <2 samples can't be stratified.
        # Replace singleton classes with "__rare__" so sklearn doesn't crash.
        counts = train_df['primary_label'].value_counts()
        stratify_col = train_df['primary_label'].where(
            train_df['primary_label'].map(counts) >= 2, other='__rare__'
        )
        train_idx, val_idx = train_test_split(
            range(len(train_df)),
            test_size=val_frac,
            stratify=stratify_col,
            random_state=seed,
        )
        train_df_split = train_df.iloc[train_idx].reset_index(drop=True)
        val_df_split = train_df.iloc[val_idx].reset_index(drop=True)

        train_split_path = os.path.join(config['OUTPUT_DIR'], 'train_split.csv')
        val_split_path = os.path.join(config['OUTPUT_DIR'], 'val_split.csv')
        train_df_split.to_csv(train_split_path, index=False)
        val_df_split.to_csv(val_split_path, index=False)
        print(f"Train: {len(train_df_split)} | Val: {len(val_df_split)}")
    else:
        train_split_path = config['TRAIN_CSV']
        val_split_path = None
        print(f"Train: {len(train_df)} | No validation split")

    print("Building training dataset...")
    train_dataset, sample_weights = build_combined_dataset(
        config, train_split_path, label_list, train_transform
    )
    print(f"  Total train samples: {len(train_dataset)}")

    if config.get('USE_WEIGHTED_SAMPLER', False):
        weights_arr = compute_sample_weights(
            train_dataset,
            label_list=label_list,
            use_secondary=config.get('USE_SECONDARY_LABELS', False),
            secondary_weight=config.get('SECONDARY_LABEL_WEIGHT', 0.5),
        )
        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(weights_arr),
            num_samples=len(train_dataset),
            replacement=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['BATCH_SIZE'],
            sampler=sampler,
            num_workers=config['NUM_WORKERS'],
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['BATCH_SIZE'],
            shuffle=True,
            num_workers=config['NUM_WORKERS'],
            pin_memory=True,
        )

    val_loader = None
    if val_split_path:
        val_dataset = TrainAudioDataset(
            csv_path=val_split_path,
            transform=val_transform,
            audio_dir=config['TRAIN_AUDIO_DIR'],
            label_list=label_list,
            use_secondary_labels=config.get('USE_SECONDARY_LABELS', False),
            secondary_label_weight=config.get('SECONDARY_LABEL_WEIGHT', 0.5),
            duration=config.get('DURATION', 5.0),
            sample_rate=config.get('SAMPLE_RATE', 32000),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['BATCH_SIZE'],
            shuffle=False,
            num_workers=config['NUM_WORKERS'],
            pin_memory=True,
        )
        print(f"  Total val samples: {len(val_dataset)}")

    model = get_model(
        num_classes=num_classes,
        backbone=config.get('BACKBONE', 'simple_cnn'),
        device=device,
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    best_auc = 0.0

    for epoch in range(config['MAX_EPOCHS']):
        print(f"\nEpoch {epoch + 1}/{config['MAX_EPOCHS']}")

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            mixup_alpha=config.get('MIXUP_ALPHA', 0.0),
        )
        print(f"Train Loss: {train_loss:.4f}")

        if val_loader is not None:
            val_loss, val_auc = validate(model, val_loader, criterion, device)
            print(f"Val   Loss: {val_loss:.4f} | Val ROC-AUC: {val_auc:.4f}")
        else:
            val_auc = 0.0

        scheduler.step()

        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
            },
            os.path.join(config['OUTPUT_DIR'], f'checkpoint_epoch_{epoch + 1}.pt'),
        )

        if config.get('SAVE_BEST_MODEL', True) and val_auc > best_auc:
            best_auc = val_auc
            torch.save(
                model.state_dict(),
                os.path.join(config['OUTPUT_DIR'], 'best_model.pt'),
            )
            print(f"  -> Saved best model (ROC-AUC: {best_auc:.4f})")

    torch.save(model.state_dict(), os.path.join(config['OUTPUT_DIR'], 'final_model.pt'))
    print(f"\nTraining complete. Best Val ROC-AUC: {best_auc:.4f}")


if __name__ == '__main__':
    main()
