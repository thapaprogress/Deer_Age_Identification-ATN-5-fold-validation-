"""
K-Fold Data Loader Utility
Provides stratified k-fold splits for model ensembling
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from utils.data_loader import load_deer_dataset, DeerAgeDataset, get_weighted_sampler
from utils.augmentation import get_training_transforms, get_validation_transforms
from training import config

def create_kfold_loaders(data_dir, num_folds=5, fold_idx=0, batch_size=32, 
                         image_size=224, num_workers=4, use_weighted_sampling=True,
                         random_seed=42):
    """
    Create data loaders for a specific fold in k-fold cross-validation
    
    Args:
        data_dir: Directory containing processed deer images
        num_folds: Total number of folds
        fold_idx: Index of the current fold (0 to num_folds-1)
        batch_size: Batch size
        image_size: Image size
        num_workers: Number of workers
        use_weighted_sampling: Use weighted sampling
        random_seed: Random seed
        
    Returns:
        dict: train and val loaders for the specific fold
    """
    # Load full dataset
    image_paths, labels, dataset_info = load_deer_dataset(data_dir)
    image_paths = np.array(image_paths)
    labels = np.array(labels)
    
    # Initialize Stratified K-Fold
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_seed)
    
    # Get indices for the requested fold
    folds = list(skf.split(image_paths, labels))
    train_idx, val_idx = folds[fold_idx]
    
    # Split data
    X_train, y_train = image_paths[train_idx], labels[train_idx]
    X_val, y_val = image_paths[val_idx], labels[val_idx]
    
    # Define transformations
    train_transform = get_training_transforms(image_size)
    val_transform = get_validation_transforms(image_size)
    
    # Create Datasets
    train_dataset = DeerAgeDataset(X_train, y_train, transform=train_transform)
    val_dataset = DeerAgeDataset(X_val, y_val, transform=val_transform)
    
    # Sampler
    train_sampler = None
    if use_weighted_sampling:
        train_sampler = get_weighted_sampler(y_train)
        
    # Data Loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        shuffle=(train_sampler is None), num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'info': dataset_info
    }
