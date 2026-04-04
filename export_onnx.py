"""
export_onnx.py — Export trained PyTorch model to ONNX format.

The exported model:
  - Input:  mel_spectrogram  (batch, 1, N_MELS, time_steps)  float32
  - Output: logits           (batch, num_classes)             float32
  Sigmoid is applied at inference time (not baked into the graph).

Usage:
    python export_onnx.py
    python export_onnx.py --model_path output/best_model.pt --output output/model.onnx
"""

import argparse
import os

import torch
import yaml
import pandas as pd

from src.models.model import get_model


def compute_time_steps(sample_rate: int, duration: float, hop_length: int) -> int:
    return int(sample_rate * duration) // hop_length + 1


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    n_mels: int,
    time_steps: int,
    opset: int = 17,
) -> None:
    model.eval()
    dummy = torch.randn(1, 1, n_mels, time_steps)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        opset_version=opset,
        input_names=['mel_spectrogram'],
        output_names=['logits'],
        dynamic_axes={
            'mel_spectrogram': {0: 'batch_size'},
            'logits':          {0: 'batch_size'},
        },
        do_constant_folding=True,
    )
    print(f"Exported → {output_path}")

    import onnx
    onnx.checker.check_model(onnx.load(output_path))
    print("ONNX model validation passed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Resolve model weights
    model_path = args.model_path
    if model_path is None:
        for fname in ('best_model.pt', 'final_model.pt'):
            candidate = os.path.join(config['OUTPUT_DIR'], fname)
            if os.path.exists(candidate):
                model_path = candidate
                break
    if model_path is None:
        raise FileNotFoundError(f"No trained model found in {config['OUTPUT_DIR']}. Run train.py first.")
    print(f"Loading weights from {model_path}")

    # Label count from sample_submission
    submission_df = pd.read_csv(config['SAMPLE_SUBMISSION_CSV'])
    num_classes = len([c for c in submission_df.columns if c != 'row_id'])
    print(f"Classes: {num_classes}")

    # Build model on CPU (ONNX export must be CPU)
    model = get_model(
        num_classes=num_classes,
        backbone=config.get('BACKBONE', 'efficientnet_b0'),
        device=torch.device('cpu'),
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    time_steps = compute_time_steps(
        config['SAMPLE_RATE'], config.get('DURATION', 5.0), config['HOP_LENGTH']
    )
    print(f"Input shape: (1, 1, {config['N_MELS']}, {time_steps})")

    output_path = args.output or config.get('ONNX_MODEL_PATH', 'output/model.onnx')
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    export_to_onnx(
        model=model,
        output_path=output_path,
        n_mels=config['N_MELS'],
        time_steps=time_steps,
        opset=config.get('ONNX_OPSET', 17),
    )

    # Quick size report
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"Model size: {size_mb:.1f} MB")
    print(f"\nDone! Set USE_ONNX: true and ONNX_MODEL_PATH: \"{output_path}\" in config.yaml")


if __name__ == '__main__':
    main()
