# ── Cell 1: Install & path ──────────────────────────────────
import subprocess
subprocess.run(["pip", "install", "onnxruntime", "-q"], capture_output=True)

import os, sys, json
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── GANTI INI SESUAI DATASET KAMU ───────────────────────────
WEIGHTS_DIR = '/kaggle/input/birdclef2026-weights'   # tempat best_model.pt
CODE_DIR    = '/kaggle/input/birdclef2026-code'      # tempat src/ & inference.py
COMP_DIR    = '/kaggle/input/birdclef-2026'          # dataset kompetisi
# ────────────────────────────────────────────────────────────

sys.path.insert(0, CODE_DIR)
os.makedirs('/kaggle/working/output', exist_ok=True)


# ── Cell 2: Load label list ──────────────────────────────────
label_list_path = f'{WEIGHTS_DIR}/label_list.json'
if os.path.exists(label_list_path):
    with open(label_list_path) as f:
        label_list = json.load(f)
else:
    train_df = pd.read_csv(f'{COMP_DIR}/train.csv')
    label_list = sorted(train_df['primary_label'].unique().tolist())
print(f"Classes: {len(label_list)}")


# ── Cell 3: Load model ───────────────────────────────────────
from src.models.model import get_model

checkpoint_path = f'{WEIGHTS_DIR}/best_model.pt'
state_dict = torch.load(checkpoint_path, map_location='cpu')
num_classes = state_dict['classifier.4.weight'].shape[0]
print(f"Checkpoint classes: {num_classes}")

model = get_model(num_classes=num_classes, backbone='efficientnet_b0', device=torch.device('cpu'), pretrained=False)
model.load_state_dict(state_dict)
model.eval()
print("Model loaded OK")


# ── Cell 4: Build dataloader ─────────────────────────────────
from src.data.dataset import TestSoundscapeDataset, AudioTransform

transform = AudioTransform(sample_rate=32000, n_mels=128, n_fft=2048, hop_length=512, augment=False)

test_dataset = TestSoundscapeDataset(
    soundscape_dir=f'{COMP_DIR}/test_soundscapes',
    sample_submission_path=f'{COMP_DIR}/sample_submission.csv',
    transform=transform,
    label_list=label_list,
)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
print(f"Test windows: {len(test_dataset)}")


# ── Cell 5: Inference ────────────────────────────────────────
all_probs, all_row_ids = [], []

with torch.no_grad():
    for inputs, row_ids in tqdm(test_loader, desc="Predicting"):
        probs = torch.sigmoid(model(inputs)).numpy()
        all_probs.append(probs)
        all_row_ids.extend(row_ids)

all_probs = np.vstack(all_probs)
print(f"Predictions shape: {all_probs.shape}")


# ── Cell 6: Save submission ──────────────────────────────────
result_df = pd.DataFrame(all_probs, columns=label_list)
result_df.insert(0, 'row_id', all_row_ids)

# Align ke format sample_submission (234 kolom)
sub_template = pd.read_csv(f'{COMP_DIR}/sample_submission.csv')
sub_cols = [c for c in sub_template.columns if c != 'row_id']
for col in sub_cols:
    if col not in result_df.columns:
        result_df[col] = 0.0
result_df = result_df[['row_id'] + sub_cols]

out_path = '/kaggle/working/submission.csv'
result_df.to_csv(out_path, index=False)
print(f"Saved: {out_path}  shape: {result_df.shape}")
result_df.head(3)
