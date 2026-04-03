import torch
import torch.nn as nn
import torch.nn.functional as F


class BirdClassifier(nn.Module):
    def __init__(self, num_classes=234, backbone='efficientnet_b0'):
        super(BirdClassifier, self).__init__()
        self.num_classes = num_classes
        
        # Convert 1-channel mel spectrograms to 3-channel for pre-trained backbones
        self.channel_adapter = nn.Conv2d(1, 3, kernel_size=1, padding=0)
        
        if backbone == 'efficientnet_b0':
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == 'resnet18':
            from torchvision.models import resnet18, ResNet18_Weights
            self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.channel_adapter(x)
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=234):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def get_model(num_classes=234, backbone='simple_cnn', device='cpu'):
    if backbone == 'simple_cnn':
        model = SimpleCNN(num_classes=num_classes)
    else:
        model = BirdClassifier(num_classes=num_classes, backbone=backbone)
    
    return model.to(device)