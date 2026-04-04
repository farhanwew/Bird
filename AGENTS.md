# AGENTS.md - BirdCLEF+ 2026

This document provides guidance for AI coding agents working in this repository.

## Project Overview

This is a **BirdCLEF+ 2026 Kaggle competition** project for multi-label audio classification
of wildlife species (birds, amphibians, mammals, reptiles, insects) from passive acoustic
monitoring recordings in Brazil's Pantanal wetlands.

- **Framework**: PyTorch
- **Task**: Multi-label classification (234 classes)
- **Input**: Audio files (.ogg) converted to mel spectrograms
- **Metric**: Macro-averaged ROC-AUC

## Project Structure

```
Bird/
├── train.py              # Training script
├── inference.py          # Inference/submission script
├── config.yaml           # Hyperparameters and paths
├── requirements.txt      # Python dependencies
├── src/
│   ├── data/
│   │   ├── dataset.py           # TrainAudioDataset, TestSoundscapeDataset
│   │   └── dataset_soundscapes.py  # TrainSoundscapesDataset
│   └── models/
│       └── model.py             # BirdClassifier, SimpleCNN, get_model()
├── data/                 # Dataset directory (not in repo)
└── output/               # Model checkpoints and submissions
```

## Build/Run Commands

### Environment Setup
```bash
pip install -r requirements.txt
```

### Training
```bash
python train.py
```

### Inference
```bash
python inference.py
```

### Running with Custom Config
Edit `config.yaml` to adjust hyperparameters before running.

## Testing

This project does not have a formal test suite. To verify functionality:

```bash
# Verify imports work
python -c "from src.data.dataset import TrainAudioDataset, AudioTransform"
python -c "from src.models.model import get_model"

# Run training for 1 epoch (modify config.yaml: MAX_EPOCHS: 1)
python train.py
```

### Adding Tests
If adding pytest tests, place them in `tests/` and run:
```bash
pytest tests/                    # Run all tests
pytest tests/test_model.py       # Run single test file
pytest tests/test_model.py::test_forward -v  # Run single test function
```

## Linting

No linter is configured. Recommended setup:
```bash
pip install ruff
ruff check .                     # Check all files
ruff check src/models/model.py   # Check single file
ruff check . --fix               # Auto-fix issues
```

## Code Style Guidelines

### Imports

Order imports in this sequence, separated by blank lines:
1. Standard library (`os`, `torch`)
2. Third-party packages (`numpy`, `pandas`, `librosa`, `tqdm`)
3. Local modules (`from src.data.dataset import ...`)

```python
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
from tqdm import tqdm
import yaml

from src.data.dataset import TrainAudioDataset, AudioTransform
from src.models.model import get_model
```

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Classes | PascalCase | `BirdClassifier`, `AudioTransform` |
| Functions | snake_case | `train_one_epoch`, `get_model` |
| Variables | snake_case | `train_loader`, `mel_spec_db` |
| Constants | UPPER_SNAKE_CASE | `SAMPLE_RATE`, `NUM_CLASSES` |
| Config keys | UPPER_SNAKE_CASE | `config['LEARNING_RATE']` |

### Type Hints

Type hints are not currently used but are encouraged for new code:
```python
def get_model(num_classes: int = 234, backbone: str = 'simple_cnn', device: str = 'cpu') -> nn.Module:
    ...
```

### Error Handling

Use try/except for file I/O operations, especially audio loading:
```python
try:
    audio, _ = librosa.load(filepath, sr=32000, duration=5.0)
except Exception as e:
    audio = np.zeros(5 * 32000)  # Fallback to silence
```

### Class Structure

PyTorch modules follow this pattern:
```python
class ModelName(nn.Module):
    def __init__(self, num_classes=234, ...):
        super(ModelName, self).__init__()
        # Initialize layers
        
    def forward(self, x):
        # Forward pass
        return logits
```

Dataset classes follow this pattern:
```python
class DatasetName(Dataset):
    def __init__(self, ...):
        # Load metadata, set up transforms
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        # Load and transform single sample
        return tensor, label
```

### Formatting

- **Indentation**: 4 spaces
- **Line length**: ~100 characters (soft limit)
- **Blank lines**: 2 between top-level definitions, 1 within classes
- **Trailing commas**: Use in multi-line structures

### Audio Processing Constants

Always use these values from `config.yaml`:
- `SAMPLE_RATE`: 32000 Hz
- `DURATION`: 5 seconds per segment
- `N_MELS`: 128 mel bands
- `N_FFT`: 2048
- `HOP_LENGTH`: 512

### Model Output

- Models output raw logits (not probabilities)
- Use `BCEWithLogitsLoss` for training
- Apply `torch.sigmoid()` for inference probabilities

## Configuration

All hyperparameters are in `config.yaml`. Key settings:
- `BATCH_SIZE`: 32
- `LEARNING_RATE`: 0.001
- `MAX_EPOCHS`: 10
- `BACKBONE`: "efficientnet_b0" (or "resnet18", "simple_cnn")
- `NUM_CLASSES`: 234

## Competition Constraints

When modifying code, remember Kaggle notebook constraints:
- **CPU runtime**: max 90 minutes
- **No internet access** during inference
- **External data**: Only public datasets allowed
- **Pre-trained models**: Allowed (e.g., EfficientNet, ResNet)

## Common Tasks

### Adding a New Model Backbone

1. Edit `src/models/model.py`
2. Add new backbone option in `BirdClassifier.__init__`
3. Update `get_model()` if needed
4. Set `BACKBONE` in `config.yaml`

### Adding Data Augmentation

1. Create augmentation class in `src/data/dataset.py`
2. Apply in `__getitem__` method
3. Consider: mixup, time shifting, frequency masking, noise injection

### Debugging Data Loading

```python
from src.data.dataset import TrainAudioDataset, AudioTransform

transform = AudioTransform()
dataset = TrainAudioDataset('data/train.csv', 'data/train_audio', transform=transform)
mel, label = dataset[0]
print(f"Mel shape: {mel.shape}, Label shape: {label.shape}")
```
