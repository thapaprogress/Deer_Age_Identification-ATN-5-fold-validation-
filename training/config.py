"""
Training Configuration for ATN Deer Age Recognition
"""

import os
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = BASE_DIR / "deer data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
AUGMENTED_DATA_DIR = DATA_DIR / "augmented"

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
LOG_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"

# Create directories if they don't exist
for dir_path in [CHECKPOINT_DIR, LOG_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
AGE_CLASSES = [2, 3, 4, 5, 6, 7, 8]  # Age groups
NUM_CLASSES = len(AGE_CLASSES)

# Train/Val/Test split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================
IMAGE_SIZE = 224  # Input image size (224x224)
IMAGE_MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
IMAGE_STD = [0.229, 0.224, 0.225]   # ImageNet std

# Data augmentation parameters
AUGMENTATION = {
    'horizontal_flip': True,
    'rotation_degrees': 15,
    'brightness': 0.2,
    'contrast': 0.2,
    'saturation': 0.2,
    'hue': 0.1,
    'random_crop': True,
    'color_jitter': True,
}

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
EMBEDDING_DIM = 128  # Dimension of embedding space
BACKBONE = 'resnet18'  # Options: 'custom_cnn', 'resnet18', 'resnet50', 'efficientnet_b0'
PRETRAINED = True  # Use pretrained weights for backbone

# Custom CNN architecture (if BACKBONE == 'custom_cnn')
CUSTOM_CNN_CONFIG = {
    'conv_channels': [32, 64, 128, 256],
    'fc_layers': [512, 256],
    'dropout': 0.3,
}

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================

# Phase 1: Standard Triplet Loss
PHASE1_EPOCHS = 50
PHASE1_BATCH_SIZE = 32
PHASE1_LEARNING_RATE = 0.001
PHASE1_WEIGHT_DECAY = 1e-4

# Phase 2: Augmented Triplet Loss
PHASE2_EPOCHS = 50
PHASE2_BATCH_SIZE = 32
PHASE2_LEARNING_RATE = 0.0005
PHASE2_WEIGHT_DECAY = 1e-4
BACKBONE_LR_FACTOR = 0.1  # Set backbone LR to 10% of head LR (as per paper)

# Optimizer
OPTIMIZER = 'adam'  # Options: 'adam', 'sgd', 'adamw'
MOMENTUM = 0.9  # For SGD

# Learning rate scheduler
LR_SCHEDULER = 'reduce_on_plateau'  # Options: 'reduce_on_plateau', 'step', 'cosine'
LR_PATIENCE = 5  # For ReduceLROnPlateau
LR_FACTOR = 0.5  # Learning rate reduction factor
LR_MIN = 1e-6  # Minimum learning rate

# Early stopping
EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 15

# ============================================================================
# TRIPLET LOSS CONFIGURATION
# ============================================================================
TRIPLET_MARGIN = 0.2  # Margin for triplet loss (as per paper)
DISTANCE_METRIC = 'cosine'  # Options: 'cosine', 'euclidean'

# Triplet mining strategy
MINING_STRATEGY = 'batch_hard'  # Options: 'batch_hard', 'batch_all', 'semi_hard'

# Threshold reducer (as per paper)
TRIPLET_THRESHOLD = 0.1  # Reduce triplets with loss > threshold

# ============================================================================
# AUGMENTED TRIPLET LOSS (ATN) CONFIGURATION
# ============================================================================
ATN_ALPHA = 0.1  # Maximum distance for same-class elements to dummy anchor
ATN_BETA = 0.3   # Minimum distance for different-class elements to dummy anchor

# Switch to ATN when standard triplet loss plateaus
ATN_SWITCH_CRITERION = 'auto'  # Options: 'auto', 'manual'
ATN_SWITCH_PATIENCE = 5  # Epochs without improvement before switching

# ============================================================================
# TRAINING UTILITIES
# ============================================================================
NUM_WORKERS = 4  # DataLoader workers
PIN_MEMORY = True  # Pin memory for faster GPU transfer

# Checkpoint saving
SAVE_BEST_ONLY = False  # Save all checkpoints or only best
SAVE_FREQUENCY = 5  # Save checkpoint every N epochs

# Logging
LOG_INTERVAL = 10  # Log every N batches
TENSORBOARD_LOG = True

# ============================================================================
# EVALUATION
# ============================================================================
# K-NN classifier for age prediction from embeddings
KNN_K = 5  # Number of neighbors

# Embedding visualization
TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1000

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CUDA_VISIBLE_DEVICES = '0'  # GPU device ID

# Mixed precision training
USE_AMP = True  # Automatic Mixed Precision for faster training

# ============================================================================
# CLASS IMBALANCE HANDLING
# ============================================================================
# Handle imbalanced age distribution
USE_WEIGHTED_SAMPLING = True  # Use weighted random sampler
OVERSAMPLE_MINORITY = True  # Oversample minority classes during augmentation

# Minimum samples per class after augmentation
MIN_SAMPLES_PER_CLASS = 100

# ============================================================================
# CONFIGURATION SUMMARY
# ============================================================================
def print_config():
    """Print configuration summary"""
    print("="*80)
    print("ATN DEER AGE RECOGNITION - CONFIGURATION")
    print("="*80)
    print(f"\nDataset:")
    print(f"  Age Classes: {AGE_CLASSES}")
    print(f"  Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  Train/Val/Test: {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")
    
    print(f"\nModel:")
    print(f"  Backbone: {BACKBONE}")
    print(f"  Embedding Dim: {EMBEDDING_DIM}")
    print(f"  Pretrained: {PRETRAINED}")
    
    print(f"\nTraining:")
    print(f"  Phase 1 Epochs: {PHASE1_EPOCHS}")
    print(f"  Phase 2 Epochs: {PHASE2_EPOCHS}")
    print(f"  Batch Size: {PHASE1_BATCH_SIZE}")
    print(f"  Learning Rate: {PHASE1_LEARNING_RATE} -> {PHASE2_LEARNING_RATE}")
    print(f"  Device: {DEVICE}")
    
    print(f"\nTriplet Loss:")
    print(f"  Margin: {TRIPLET_MARGIN}")
    print(f"  Distance: {DISTANCE_METRIC}")
    print(f"  Mining: {MINING_STRATEGY}")
    
    print(f"\nATN Loss:")
    print(f"  Alpha: {ATN_ALPHA}")
    print(f"  Beta: {ATN_BETA}")
    
    print("="*80)


if __name__ == "__main__":
    print_config()
