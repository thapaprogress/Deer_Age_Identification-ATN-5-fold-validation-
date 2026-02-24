"""
Data Loader for ATN Deer Age Recognition
Handles dataset loading, splitting, and triplet sampling
"""

import os
import json
from pathlib import Path
from collections import defaultdict
import random

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import pillow_heif
pillow_heif.register_heif_opener()
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from tqdm import tqdm

from utils.augmentation import get_training_transforms, get_validation_transforms
from training import config


class DeerAgeDataset(Dataset):
    """Dataset for deer age classification with triplet sampling support"""
    
    def __init__(self, image_paths, labels, transform=None, return_triplets=False):
        """
        Args:
            image_paths: List of paths to images
            labels: List of age labels corresponding to images
            transform: Image transformations
            return_triplets: If True, returns triplets (anchor, positive, negative)
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.return_triplets = return_triplets
        
        # Group images by label for triplet sampling
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)
        
        self.unique_labels = list(self.label_to_indices.keys())
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if self.return_triplets:
            return self._get_triplet(idx)
        else:
            return self._get_single(idx)
    
    def _get_single(self, idx):
        """Get a single image and its label"""
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def _get_triplet(self, idx):
        """
        Get a triplet: (anchor, positive, negative)
        Anchor and positive have same label, negative has different label
        """
        # Anchor
        anchor_path = self.image_paths[idx]
        anchor_label = self.labels[idx]
        anchor_img = Image.open(anchor_path).convert('RGB')
        
        # Positive: same label as anchor, but different image
        positive_indices = [i for i in self.label_to_indices[anchor_label] if i != idx]
        if len(positive_indices) == 0:
            # If no other image with same label, use the same image
            positive_idx = idx
        else:
            positive_idx = random.choice(positive_indices)
        
        positive_path = self.image_paths[positive_idx]
        positive_img = Image.open(positive_path).convert('RGB')
        
        # Negative: different label from anchor
        negative_label = random.choice([l for l in self.unique_labels if l != anchor_label])
        negative_idx = random.choice(self.label_to_indices[negative_label])
        negative_path = self.image_paths[negative_idx]
        negative_img = Image.open(negative_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
        
        return (anchor_img, positive_img, negative_img), anchor_label


def get_transforms(image_size=224, augment=True):
    """
    Get image transformations
    
    Args:
        image_size: Target image size
        augment: Whether to apply data augmentation
    
    Returns:
        torchvision.transforms.Compose
    """
    if augment:
        transform = transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    return transform


def load_deer_dataset(data_dir, age_classes=[2, 3, 4, 5, 6, 7, 8]):
    """
    Load deer dataset from directory structure
    
    Args:
        data_dir: Directory containing deer images organized by deer_id folders
        age_classes: List of age classes to include
    
    Returns:
        image_paths: List of image paths
        labels: List of age labels
        dataset_info: Dictionary with dataset statistics
    """
    data_dir = Path(data_dir)
    image_paths = []
    labels = []
    
    # Statistics
    age_distribution = defaultdict(int)
    deer_count = defaultdict(int)
    
    # Iterate through deer folders
    for folder in data_dir.iterdir():
        if not folder.is_dir():
            continue
        
        # Extract age from folder name
        folder_name = folder.name
        if 'age' not in folder_name.lower():
            continue
        
        try:
            # Parse age from folder name like "Deer_id_1 (age 5)"
            age_str = folder_name.split('age')[1].strip()
            age_str = age_str.replace(')', '').replace('(', '').strip()
            age = int(age_str.split()[0])
            
            if age not in age_classes:
                continue
            
            # Find all images in this deer's folder
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.HEIC', '*.heic']
            for ext in extensions:
                for img_path in folder.rglob(ext):
                    image_paths.append(str(img_path))
                    labels.append(age)
                    age_distribution[age] += 1
            
            deer_count[age] += 1
            
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse age from folder: {folder_name}")
            continue
    
    dataset_info = {
        'total_images': len(image_paths),
        'age_distribution': dict(age_distribution),
        'deer_count': dict(deer_count),
        'age_classes': sorted(age_distribution.keys())
    }
    
    return image_paths, labels, dataset_info


def create_stratified_split(image_paths, labels, train_ratio=0.7, val_ratio=0.15, 
                            test_ratio=0.15, random_seed=42):
    """
    Create stratified train/val/test split
    
    Args:
        image_paths: List of image paths
        labels: List of labels
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary with train/val/test splits
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    # First split: train vs (val + test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels,
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=random_seed
    )
    
    # Second split: val vs test
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=(1 - val_ratio_adjusted),
        stratify=temp_labels,
        random_state=random_seed
    )
    
    return {
        'train': {'paths': train_paths, 'labels': train_labels},
        'val': {'paths': val_paths, 'labels': val_labels},
        'test': {'paths': test_paths, 'labels': test_labels}
    }


def get_weighted_sampler(labels):
    """
    Create weighted sampler to handle class imbalance
    
    Args:
        labels: List of labels
    
    Returns:
        WeightedRandomSampler
    """
    # Count samples per class
    class_counts = defaultdict(int)
    for label in labels:
        class_counts[label] += 1
    
    # Calculate weights (inverse frequency)
    weights = []
    for label in labels:
        weight = 1.0 / class_counts[label]
        weights.append(weight)
    
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )
    
    return sampler


def create_data_loaders(data_dir, batch_size=32, image_size=224, 
                       num_workers=4, use_weighted_sampling=True,
                       return_triplets=False, random_seed=42):
    """
    Create train/val/test data loaders
    
    Args:
        data_dir: Directory containing processed deer images
        batch_size: Batch size
        image_size: Image size
        num_workers: Number of data loading workers
        use_weighted_sampling: Use weighted sampling for class imbalance
        return_triplets: Return triplets instead of single images
        random_seed: Random seed
    
    Returns:
        Dictionary with train/val/test loaders and dataset info
    """
    # Load dataset
    image_paths, labels, dataset_info = load_deer_dataset(data_dir)
    
    print(f"\nDataset loaded:")
    print(f"  Total images: {dataset_info['total_images']}")
    print(f"  Age distribution: {dataset_info['age_distribution']}")
    print(f"  Deer count per age: {dataset_info['deer_count']}")
    
    # Create splits
    splits = create_stratified_split(
        image_paths, labels,
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
        random_seed=random_seed
    )
    
    # Extract paths and labels for convenience
    X_train, y_train = splits['train']['paths'], splits['train']['labels']
    X_val, y_val = splits['val']['paths'], splits['val']['labels']
    X_test, y_test = splits['test']['paths'], splits['test']['labels']
    
    # Define transformations
    train_transform = get_training_transforms(image_size)
    val_transform = get_validation_transforms(image_size)
    
    # Create Datasets
    train_dataset = DeerAgeDataset(
        image_paths=X_train,
        labels=y_train,
        transform=train_transform,
        return_triplets=return_triplets
    )
    
    val_dataset = DeerAgeDataset(
        image_paths=X_val,
        labels=y_val,
        transform=val_transform,
        return_triplets=False
    )
    
    test_dataset = DeerAgeDataset(
        image_paths=X_test,
        labels=y_test,
        transform=val_transform,
        return_triplets=False
    )
    
    # Create samplers
    train_sampler = None
    if use_weighted_sampling and not return_triplets:
        train_sampler = get_weighted_sampler(splits['train']['labels'])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"\nData loaders created:")
    print(f"  Train: {len(train_dataset)} images, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} images, {len(val_loader)} batches")
    print(f"  Test:  {len(test_dataset)} images, {len(test_loader)} batches")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'dataset_info': dataset_info,
        'splits': splits
    }


if __name__ == "__main__":
    # Test data loader
    data_dir = r"c:\Users\PRAJNA WORLD TECH\OneDrive\Desktop\atn\data\raw"
    
    print("Testing data loader...")
    loaders = create_data_loaders(
        data_dir,
        batch_size=8,
        image_size=224,
        num_workers=0,
        return_triplets=False
    )
    
    # Test a batch
    for images, labels in loaders['train']:
        print(f"\nBatch shape: {images.shape}")
        print(f"Labels: {labels}")
        break
