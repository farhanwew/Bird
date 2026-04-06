import torch
import numpy as np
import pandas as pd
import librosa
import os
from torch.utils.data import Dataset


def _to_seconds(value) -> float:
    """Convert a time value to seconds. Handles float, int, and 'HH:MM:SS' / 'MM:SS' strings."""
    try:
        return float(value)
    except (ValueError, TypeError):
        parts = str(value).strip().split(':')
        parts = [float(p) for p in parts]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        return 0.0


class TrainSoundscapesDataset(Dataset):
    def __init__(self, soundscape_dir, labels_csv, transform=None, label_list=None,
                 mel_cache_dir=None):
        self.soundscape_dir = soundscape_dir
        self.transform = transform
        self.mel_cache_dir = mel_cache_dir

        self.labels_df = pd.read_csv(labels_csv)

        if label_list is None:
            all_labels = set()
            for labels in self.labels_df['primary_label'].dropna():
                for label in labels.split(';'):
                    all_labels.add(label.strip())
            self.label_list = sorted(list(all_labels))
        else:
            self.label_list = label_list

        self.label_to_idx = {label: idx for idx, label in enumerate(self.label_list)}

    def __len__(self):
        return len(self.labels_df)

    def _load_mel(self, filename, start, duration):
        """Load mel from cache if available, else compute from audio.
        Cache key uses filename + start time to distinguish windows.
        """
        if self.mel_cache_dir:
            # Cache key: filename_start.npy (e.g. BC2026_Train_0001_0.npy)
            stem = os.path.splitext(filename)[0]
            npy_name = f"{stem}_{int(start)}.npy"
            npy_path = os.path.join(self.mel_cache_dir, npy_name)
            if os.path.exists(npy_path):
                mel = np.load(npy_path)
                if self.transform is not None and hasattr(self.transform, 'aug') \
                        and self.transform.aug is not None:
                    mel = self.transform.aug(mel)
                return mel

        # Fallback: load from audio
        filepath = os.path.join(self.soundscape_dir, filename)
        try:
            audio, sr = librosa.load(filepath, sr=32000, offset=start, duration=duration)
            target = int(duration * 32000)
            if len(audio) < target:
                audio = np.pad(audio, (0, target - len(audio)), mode='constant')
        except Exception:
            audio = np.zeros(int(duration * 32000))

        if self.transform:
            return self.transform(audio)
        return np.zeros((128, 313))

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        filename = row['filename']
        start    = _to_seconds(row.get('start', 0))
        end      = _to_seconds(row.get('end', 5))
        duration = end - start

        mel = self._load_mel(filename, start, duration)

        labels = torch.zeros(len(self.label_list))
        if pd.notna(row['primary_label']):
            for label in row['primary_label'].split(';'):
                label = label.strip()
                if label in self.label_to_idx:
                    labels[self.label_to_idx[label]] = 1.0

        return torch.FloatTensor(mel).unsqueeze(0), labels