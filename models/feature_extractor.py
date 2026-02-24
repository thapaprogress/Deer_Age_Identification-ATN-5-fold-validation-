"""
Feature Extractor (CNN Backbone) for ATN
Supports custom CNN and pretrained models (ResNet, EfficientNet)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CustomCNN(nn.Module):
    """Custom CNN backbone for feature extraction"""
    
    def __init__(self, embedding_dim=128, dropout=0.3):
        """
        Args:
            embedding_dim: Dimension of output embedding
            dropout: Dropout probability
        """
        super(CustomCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
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
        
        # Calculate flattened size (for 224x224 input)
        # After 4 pooling layers: 224 -> 112 -> 56 -> 28 -> 14
        self.flat_size = 256 * 14 * 14
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flat_size, 512)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(256, embedding_dim)
    
    def forward(self, x):
        # Conv block 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Conv block 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Conv block 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Conv block 4
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x


class ResNetBackbone(nn.Module):
    """ResNet backbone for feature extraction"""
    
    def __init__(self, embedding_dim=128, pretrained=True, resnet_type='resnet18'):
        """
        Args:
            embedding_dim: Dimension of output embedding
            pretrained: Use pretrained ImageNet weights
            resnet_type: 'resnet18', 'resnet34', 'resnet50', etc.
        """
        super(ResNetBackbone, self).__init__()
        
        # Load pretrained ResNet
        if resnet_type == 'resnet18':
            self.resnet = models.resnet18(weights='DEFAULT' if pretrained else None)
            resnet_output_dim = 512
        elif resnet_type == 'resnet34':
            self.resnet = models.resnet34(weights='DEFAULT' if pretrained else None)
            resnet_output_dim = 512
        elif resnet_type == 'resnet50':
            self.resnet = models.resnet50(weights='DEFAULT' if pretrained else None)
            resnet_output_dim = 2048
        else:
            raise ValueError(f"Unknown resnet_type: {resnet_type}")
        
        # Remove the final classification layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # Add embedding layer
        self.embedding = nn.Linear(resnet_output_dim, embedding_dim)
    
    def forward(self, x):
        # Extract features with ResNet
        x = self.resnet(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Project to embedding space
        x = self.embedding(x)
        
        return x


class EfficientNetBackbone(nn.Module):
    """EfficientNet backbone for feature extraction"""
    
    def __init__(self, embedding_dim=128, pretrained=True, efficientnet_type='b0'):
        """
        Args:
            embedding_dim: Dimension of output embedding
            pretrained: Use pretrained ImageNet weights
            efficientnet_type: 'b0', 'b1', 'b2', etc.
        """
        super(EfficientNetBackbone, self).__init__()
        
        # Load pretrained EfficientNet
        if efficientnet_type == 'b0':
            self.efficientnet = models.efficientnet_b0(weights='DEFAULT' if pretrained else None)
            efficientnet_output_dim = 1280
        elif efficientnet_type == 'b1':
            self.efficientnet = models.efficientnet_b1(weights='DEFAULT' if pretrained else None)
            efficientnet_output_dim = 1280
        else:
            raise ValueError(f"Unknown efficientnet_type: {efficientnet_type}")
        
        # Remove the final classification layer
        self.efficientnet.classifier = nn.Identity()
        
        # Add embedding layer
        self.embedding = nn.Linear(efficientnet_output_dim, embedding_dim)
    
    def forward(self, x):
        # Extract features with EfficientNet
        x = self.efficientnet(x)
        
        # Project to embedding space
        x = self.embedding(x)
        
        return x


class FeatureExtractor(nn.Module):
    """
    Unified feature extractor with L2 normalization
    Supports multiple backbone architectures
    """
    
    def __init__(self, embedding_dim=128, backbone='resnet18', pretrained=True, 
                 normalize_embeddings=True):
        """
        Args:
            embedding_dim: Dimension of output embedding
            backbone: 'custom_cnn', 'resnet18', 'resnet34', 'resnet50', 'efficientnet_b0'
            pretrained: Use pretrained weights (for ResNet/EfficientNet)
            normalize_embeddings: Apply L2 normalization to embeddings
        """
        super(FeatureExtractor, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.normalize_embeddings = normalize_embeddings
        
        # Select backbone
        if backbone == 'custom_cnn':
            self.backbone = CustomCNN(embedding_dim=embedding_dim)
        elif backbone.startswith('resnet'):
            self.backbone = ResNetBackbone(
                embedding_dim=embedding_dim,
                pretrained=pretrained,
                resnet_type=backbone
            )
        elif backbone.startswith('efficientnet'):
            efficientnet_type = backbone.split('_')[1]  # e.g., 'b0' from 'efficientnet_b0'
            self.backbone = EfficientNetBackbone(
                embedding_dim=embedding_dim,
                pretrained=pretrained,
                efficientnet_type=efficientnet_type
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
    
    def forward(self, x):
        # Extract features
        embeddings = self.backbone(x)
        
        # L2 normalization (important for metric learning)
        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def get_embedding_dim(self):
        return self.embedding_dim


def create_feature_extractor(embedding_dim=128, backbone='resnet18', 
                             pretrained=True, device='cuda'):
    """
    Factory function to create feature extractor
    
    Args:
        embedding_dim: Dimension of output embedding
        backbone: Backbone architecture
        pretrained: Use pretrained weights
        device: Device to place model on
    
    Returns:
        FeatureExtractor model
    """
    model = FeatureExtractor(
        embedding_dim=embedding_dim,
        backbone=backbone,
        pretrained=pretrained,
        normalize_embeddings=True
    )
    
    model = model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nFeature Extractor created:")
    print(f"  Backbone: {backbone}")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Device: {device}")
    
    return model


if __name__ == "__main__":
    # Test feature extractor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Testing Custom CNN...")
    model_custom = create_feature_extractor(
        embedding_dim=128,
        backbone='custom_cnn',
        device=device
    )
    
    # Test forward pass
    x = torch.randn(4, 3, 224, 224).to(device)
    embeddings = model_custom(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {embeddings.shape}")
    print(f"Embedding norm: {torch.norm(embeddings, dim=1)}")  # Should be ~1 due to L2 norm
    
    print("\n" + "="*60)
    print("Testing ResNet18...")
    model_resnet = create_feature_extractor(
        embedding_dim=128,
        backbone='resnet18',
        pretrained=True,
        device=device
    )
    
    embeddings = model_resnet(x)
    print(f"Output shape: {embeddings.shape}")
    print(f"Embedding norm: {torch.norm(embeddings, dim=1)}")
