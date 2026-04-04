import torch
import numpy as np
import pandas as pd
import librosa
import os
from torch.utils.data import Dataset


class MelTransform:
    """Converts raw audio to a normalized mel spectrogram. No augmentation."""
    def __init__(self, sample_rate=32000, n_mels=128, n_fft=2048, hop_length=512):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __call__(self, audio):
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-6)
        return mel_spec_db


class SpecAugment:
    """SpecAugment: frequency masking, time masking, brightness jitter on mel spectrograms."""
    def __init__(
        self,
        freq_mask_max=27,
        freq_mask_num=2,
        time_mask_max=40,
        time_mask_num=2,
        brightness_factor=0.2,
        prob=0.5,
    ):
        self.freq_mask_max = freq_mask_max
        self.freq_mask_num = freq_mask_num
        self.time_mask_max = time_mask_max
        self.time_mask_num = time_mask_num
        self.brightness_factor = brightness_factor
        self.prob = prob

    def __call__(self, mel):
        """mel: numpy array (n_mels, time), values in [0, 1]."""
        if np.random.random() > self.prob:
            return mel
        mel = mel.copy()
        n_mels, time_steps = mel.shape

        for _ in range(self.freq_mask_num):
            f = np.random.randint(0, self.freq_mask_max + 1)
            f0 = np.random.randint(0, max(1, n_mels - f))
            mel[f0:f0 + f, :] = 0.0

        for _ in range(self.time_mask_num):
            t = np.random.randint(0, self.time_mask_max + 1)
            t0 = np.random.randint(0, max(1, time_steps - t))
            mel[:, t0:t0 + t] = 0.0

        if self.brightness_factor > 0:
            shift = np.random.uniform(-self.brightness_factor, self.brightness_factor)
            mel = np.clip(mel + shift, 0.0, 1.0)

        return mel


class AudioTransform:
    """Chains MelTransform and optionally SpecAugment. Drop-in replacement for old AudioTransform."""
    def __init__(
        self,
        sample_rate=32000,
        n_mels=128,
        n_fft=2048,
        hop_length=512,
        augment=False,
        spec_augment_cfg=None,
    ):
        self.mel = MelTransform(sample_rate, n_mels, n_fft, hop_length)
        self.aug = SpecAugment(**(spec_augment_cfg or {})) if augment else None

    def __call__(self, audio):
        mel = self.mel(audio)
        if self.aug is not None:
            mel = self.aug(mel)
        return mel


class TrainAudioDataset(Dataset):
    def __init__(
        self,
        csv_path,
        audio_dir,
        transform=None,
        label_list=None,
        use_secondary_labels=False,
        secondary_label_weight=0.5,
        duration=5.0,
        sample_rate=32000,
    ):
        self.df = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.transform = transform
        self.use_secondary_labels = use_secondary_labels
        self.secondary_label_weight = secondary_label_weight
        self.duration = duration
        self.sample_rate = sample_rate
        self.n_samples = int(duration * sample_rate)

        if label_list is None:
            self.label_list = sorted(self.df['primary_label'].unique().tolist())
        else:
            self.label_list = label_list
        self.label_to_idx = {label: idx for idx, label in enumerate(self.label_list)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['filename']
        filepath = os.path.join(self.audio_dir, filename)

        try:
            audio, _ = librosa.load(filepath, sr=self.sample_rate, duration=self.duration)
            if len(audio) < self.n_samples:
                audio = np.pad(audio, (0, self.n_samples - len(audio)), mode='constant')
            else:
                audio = audio[:self.n_samples]
        except Exception:
            audio = np.zeros(self.n_samples)

        if self.transform:
            mel = self.transform(audio)
        else:
            mel = np.zeros((128, 313))

        label_idx = self.label_to_idx.get(row['primary_label'], 0)
        label = torch.zeros(len(self.label_list))
        label[label_idx] = 1.0

        if self.use_secondary_labels:
            raw = str(row.get('secondary_labels', '') or '')
            # Format in train.csv: "['abc', 'xyz']" or empty
            raw = raw.strip("[]").replace("'", "").replace('"', '')
            for sec in raw.split(','):
                sec = sec.strip()
                if sec and sec in self.label_to_idx:
                    label[self.label_to_idx[sec]] = max(
                        label[self.label_to_idx[sec]].item(),
                        self.secondary_label_weight,
                    )

        return torch.FloatTensor(mel).unsqueeze(0), label


class TestSoundscapeDataset(Dataset):
    def __init__(self, soundscape_dir, sample_submission_path, transform=None, label_list=None):
        self.soundscape_dir = soundscape_dir
        self.transform = transform
        
        self.submission_df = pd.read_csv(sample_submission_path)
        
        if label_list is None:
            self.label_list = [col for col in self.submission_df.columns if col != 'row_id']
        else:
            self.label_list = label_list
        
        self.row_ids = self.submission_df['row_id'].tolist()
        
    def __len__(self):
        return len(self.row_ids)
    
    def __getitem__(self, idx):
        row_id = self.row_ids[idx]
        parts = row_id.rsplit('_', 1)
        filename = parts[0] + '.ogg'
        end_time = int(parts[1])
        start_time = end_time - 5
        
        filepath = os.path.join(self.soundscape_dir, filename)
        
        try:
            audio, sr = librosa.load(filepath, sr=32000, offset=start_time, duration=5.0)
            if len(audio) < 5 * 32000:
                audio = np.pad(audio, (0, 5 * 32000 - len(audio)), mode='constant')
        except Exception:
            audio = np.zeros(5 * 32000)
        
        if self.transform:
            mel = self.transform(audio)
        else:
            mel = np.zeros((128, 313))
        
        return torch.FloatTensor(mel).unsqueeze(0), row_id