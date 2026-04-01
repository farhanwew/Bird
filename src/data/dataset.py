import torch
import numpy as np
import pandas as pd
import librosa
import os
from torch.utils.data import Dataset


class AudioTransform:
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
            hop_length=self.hop_length
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-6)
        return mel_spec_db


class TrainAudioDataset(Dataset):
    def __init__(self, csv_path, audio_dir, transform=None, label_list=None):
        self.df = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.transform = transform
        
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
            audio, _ = librosa.load(filepath, sr=32000, duration=5.0)
            if len(audio) < 5 * 32000:
                audio = np.pad(audio, (0, 5 * 32000 - len(audio)), mode='constant')
            else:
                audio = audio[:5 * 32000]
        except Exception as e:
            audio = np.zeros(5 * 32000)
        
        if self.transform:
            mel = self.transform(audio)
        else:
            mel = np.zeros((128, 313))
        
        label_idx = self.label_to_idx.get(row['primary_label'], 0)
        label = torch.zeros(len(self.label_list))
        label[label_idx] = 1.0
        
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