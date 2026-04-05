import os
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from src.data.dataset import TestSoundscapeDataset, AudioTransform
from src.models.model import get_model


# ---------------------------------------------------------------------------
# Predictor backends
# ---------------------------------------------------------------------------

class TorchPredictor:
    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def predict_batch(self, inputs_np: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(inputs_np).to(self.device)
        with torch.no_grad():
            logits = self.model(t)
        return torch.sigmoid(logits).cpu().numpy()


class OnnxPredictor:
    def __init__(self, model_path: str, num_threads: int = 4, providers: list = None):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=providers or ['CPUExecutionProvider'],
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict_batch(self, inputs_np: np.ndarray) -> np.ndarray:
        logits = self.session.run(None, {self.input_name: inputs_np.astype(np.float32)})[0]
        return 1.0 / (1.0 + np.exp(-logits))  # sigmoid


# ---------------------------------------------------------------------------
# Unified inference loop
# ---------------------------------------------------------------------------

def predict(predictor, dataloader) -> tuple:
    all_probs, all_row_ids = [], []
    for inputs, row_ids in tqdm(dataloader, desc="Predicting"):
        probs = predictor.predict_batch(inputs.numpy())
        all_probs.append(probs)
        all_row_ids.extend(row_ids)
    return np.vstack(all_probs), all_row_ids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    transform = AudioTransform(
        sample_rate=config['SAMPLE_RATE'],
        n_mels=config['N_MELS'],
        n_fft=config['N_FFT'],
        hop_length=config['HOP_LENGTH'],
        augment=False,
    )

    # Load label_list — priority: label_list.json > TRAIN_CSV > sample_submission
    label_list_path = os.path.join(config['OUTPUT_DIR'], 'label_list.json')
    if os.path.exists(label_list_path):
        with open(label_list_path) as f:
            label_list = json.load(f)
        print(f"Loaded label list from {label_list_path} ({len(label_list)} classes)")
    elif config.get('TRAIN_CSV') and os.path.exists(config['TRAIN_CSV']):
        train_df = pd.read_csv(config['TRAIN_CSV'])
        label_list = sorted(train_df['primary_label'].unique().tolist())
        print(f"Loaded label list from TRAIN_CSV ({len(label_list)} classes)")
    else:
        submission_df = pd.read_csv(config['SAMPLE_SUBMISSION_CSV'])
        label_list = [col for col in submission_df.columns if col != 'row_id']
        print(f"Warning: using sample_submission columns ({len(label_list)} classes)")
    num_classes = len(label_list)

    test_dataset = TestSoundscapeDataset(
        soundscape_dir=config['TEST_SOUNDSCAPES_DIR'],
        sample_submission_path=config['SAMPLE_SUBMISSION_CSV'],
        transform=transform,
        label_list=label_list,
    )

    batch_size = config.get('INFERENCE_BATCH_SIZE', config['BATCH_SIZE'])
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,      # 0 workers is safest on Kaggle CPU
        pin_memory=False,
    )
    print(f"Test windows: {len(test_dataset)} | Batch size: {batch_size}")

    # Choose backend: ONNX if available and configured, else PyTorch
    onnx_path = config.get('ONNX_MODEL_PATH', 'output/model.onnx')
    use_onnx = config.get('USE_ONNX', False) and os.path.exists(onnx_path)

    if use_onnx:
        print(f"Backend: ONNX  ({onnx_path})")
        predictor = OnnxPredictor(
            model_path=onnx_path,
            num_threads=config.get('ONNX_NUM_THREADS', 4),
        )
    else:
        if config.get('USE_ONNX', False):
            print(f"Warning: USE_ONNX=true but {onnx_path} not found. Falling back to PyTorch.")
        print("Backend: PyTorch CPU")
        device = torch.device('cpu')
        model = get_model(
            num_classes=num_classes,
            backbone=config.get('BACKBONE', 'simple_cnn'),
            device=device,
        )
        checkpoint_path = None
        for fname in ('best_model.pt', 'final_model.pt'):
            candidate = os.path.join(config['OUTPUT_DIR'], fname)
            if os.path.exists(candidate):
                checkpoint_path = candidate
                break
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            # Detect actual num_classes from checkpoint to avoid size mismatch
            ckpt_classes = state_dict['classifier.4.weight'].shape[0]
            if ckpt_classes != num_classes:
                print(f"Note: checkpoint has {ckpt_classes} classes, rebuilding model.")
                model = get_model(
                    num_classes=ckpt_classes,
                    backbone=config.get('BACKBONE', 'simple_cnn'),
                    device=device,
                )
                label_list = label_list[:ckpt_classes]  # safe: TRAIN_CSV is already sorted same way
            model.load_state_dict(state_dict)
            print(f"Loaded weights from {checkpoint_path} ({ckpt_classes} classes)")
        else:
            print("Warning: No trained model found, using random weights")
        predictor = TorchPredictor(model, device)

    predictions, row_ids = predict(predictor, test_loader)

    result_df = pd.DataFrame(predictions, columns=label_list)
    result_df.insert(0, 'row_id', row_ids)

    # Align to sample_submission format (may have more columns than trained classes)
    submission_df = pd.read_csv(config['SAMPLE_SUBMISSION_CSV'])
    submission_cols = [col for col in submission_df.columns if col != 'row_id']
    for col in submission_cols:
        if col not in result_df.columns:
            result_df[col] = 0.0
    result_df = result_df[['row_id'] + submission_cols]

    os.makedirs(config['OUTPUT_DIR'], exist_ok=True)
    output_path = os.path.join(config['OUTPUT_DIR'], 'submission.csv')
    result_df.to_csv(output_path, index=False)
    print(f"Submission saved → {output_path}  shape: {result_df.shape}")


if __name__ == '__main__':
    main()
