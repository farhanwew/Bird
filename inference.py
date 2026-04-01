import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from src.data.dataset import TestSoundscapeDataset, AudioTransform
from src.models.model import get_model


def predict(model, dataloader, device):
    model.eval()
    predictions = []
    row_ids = []
    
    with torch.no_grad():
        for inputs, rids in tqdm(dataloader, desc="Predicting"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            
            predictions.append(probs.cpu().numpy())
            row_ids.extend(rids)
    
    predictions = np.vstack(predictions)
    return predictions, row_ids


def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    transform = AudioTransform(
        sample_rate=config['SAMPLE_RATE'],
        n_mels=config['N_MELS'],
        n_fft=config['N_FFT'],
        hop_length=config['HOP_LENGTH']
    )
    
    submission_df = pd.read_csv(config['SAMPLE_SUBMISSION_CSV'])
    label_list = [col for col in submission_df.columns if col != 'row_id']
    print(f"Number of classes: {len(label_list)}")
    
    test_dataset = TestSoundscapeDataset(
        soundscape_dir=config['TEST_SOUNDSCAPES_DIR'],
        sample_submission_path=config['SAMPLE_SUBMISSION_CSV'],
        transform=transform,
        label_list=label_list
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['BATCH_SIZE'],
        shuffle=False,
        num_workers=config['NUM_WORKERS'],
        pin_memory=True
    )
    
    model = get_model(
        num_classes=config['NUM_CLASSES'],
        backbone=config.get('BACKBONE', 'simple_cnn'),
        device=device
    )
    
    checkpoint_path = os.path.join(config['OUTPUT_DIR'], 'final_model.pt')
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded model from {checkpoint_path}")
    else:
        print("Warning: No trained model found, using random weights")
    
    predictions, row_ids = predict(model, test_loader, device)
    
    result_df = pd.DataFrame(predictions, columns=label_list)
    result_df.insert(0, 'row_id', row_ids)
    
    output_path = os.path.join(config['OUTPUT_DIR'], 'submission.csv')
    result_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(f"Shape: {result_df.shape}")


if __name__ == '__main__':
    main()