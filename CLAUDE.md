# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

BirdCLEF+ 2026 Kaggle competition — multi-label audio classification of 234 wildlife species from passive acoustic monitoring in Brazil's Pantanal. Metric: macro-averaged ROC-AUC. Submission is a Kaggle notebook with CPU-only inference (≤90 min runtime, no internet).

## Commands

```bash
pip install -r requirements.txt   # install dependencies

python train.py                    # train model (reads config.yaml)
python inference.py                # run inference, outputs output/submission.csv

# Verify imports
python -c "from src.data.dataset import TrainAudioDataset, AudioTransform"
python -c "from src.models.model import get_model"
```

All hyperparameters and paths are in `config.yaml`. Data paths default to `/content/drive/MyDrive/bird-data` (Google Colab). Update these for local runs.

## Architecture

**Data flow**: `.ogg` audio → `librosa.load` (32kHz, 5s) → `AudioTransform` (mel spectrogram, 128 mel bands) → normalized float tensor `(1, 128, T)` → model → raw logits.

**Three dataset classes** (`src/data/`):
- `TrainAudioDataset` — per-clip training data from `train.csv`, one-hot labels from `primary_label`
- `TestSoundscapeDataset` — test inference; parses `row_id` as `{filename}_{end_time}` to extract 5s windows
- `TrainSoundscapesDataset` — labeled soundscape training data; supports multi-label `primary_label` with `;`-separated values

**Two model variants** (`src/models/model.py`):
- `BirdClassifier` — pretrained backbone (EfficientNet-B0 or ResNet18) with a 1→3 channel adapter conv and custom classifier head
- `SimpleCNN` — lightweight 4-block CNN with global average pooling; used when `BACKBONE: simple_cnn`

`get_model(num_classes, backbone, device)` dispatches between them.

**Training** (`train.py`): `BCEWithLogitsLoss`, Adam optimizer, StepLR scheduler (step=3, gamma=0.5). Saves checkpoints per epoch to `output/`. No validation split currently — the validate function exists but is not called in `main()`.

**Inference** (`inference.py`): Loads `output/final_model.pt`, applies `torch.sigmoid()` on logits, writes `output/submission.csv` with one column per species.

## Key Notes

- Models output raw logits — use `BCEWithLogitsLoss` for training, `torch.sigmoid()` for inference probabilities.
- `train.py` uses `outputs.max(1)` for accuracy tracking, which is single-label accuracy — not the actual competition metric (ROC-AUC). This is a known discrepancy.
- `NUM_CLASSES: 234` in config, but actual class count is derived at runtime from `train.csv` unique labels (training) or `sample_submission.csv` columns (inference).
- Kaggle submission constraints: CPU only, ≤90 min, no internet. Pre-trained model weights must be bundled or loaded from `/kaggle/input`.
