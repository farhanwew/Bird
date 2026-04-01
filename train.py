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


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(tqdm(dataloader, desc="Training")):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        _, true_idx = labels.max(1)
        total += labels.size(0)
        correct += predicted.eq(true_idx).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            _, true_idx = labels.max(1)
            total += labels.size(0)
            correct += predicted.eq(true_idx).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


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
    
    train_df = pd.read_csv(config['TRAIN_CSV'])
    label_list = sorted(train_df['primary_label'].unique().tolist())
    print(f"Number of classes: {len(label_list)}")
    
    train_dataset = TrainAudioDataset(
        csv_path=config['TRAIN_CSV'],
        audio_dir=config['TRAIN_AUDIO_DIR'],
        transform=transform,
        label_list=label_list
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['BATCH_SIZE'],
        shuffle=True,
        num_workers=config['NUM_WORKERS'],
        pin_memory=True
    )
    
    model = get_model(
        num_classes=config['NUM_CLASSES'],
        backbone=config.get('BACKBONE', 'simple_cnn'),
        device=device
    )
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['LEARNING_RATE'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    
    os.makedirs(config['OUTPUT_DIR'], exist_ok=True)
    
    for epoch in range(config['MAX_EPOCHS']):
        print(f"\nEpoch {epoch + 1}/{config['MAX_EPOCHS']}")
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        scheduler.step()
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(config['OUTPUT_DIR'], f'checkpoint_epoch_{epoch + 1}.pt'))
    
    print("Training complete!")
    torch.save(model.state_dict(), os.path.join(config['OUTPUT_DIR'], 'final_model.pt'))