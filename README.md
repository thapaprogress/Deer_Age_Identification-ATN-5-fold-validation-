streamlit run dashboard/app.py

# ATN Deer Age Recognition System

Deep learning system for classifying deer age using **Augmented Triplet Network (ATN)** based on the research paper: *"Augmented Triplet Network for Individual Organism and Unique Object Classification for Reliable Monitoring of Ezoshika Deer"*.

## 📋 Overview

This project implements a deer age recognition system that classifies deer into age groups (2-8 years) using visual pattern recognition. The system uses a two-phase training approach:

1. **Phase 1**: Standard Triplet Loss for initial embedding learning
2. **Phase 2**: Augmented Triplet Loss (ATN) with dummy anchors for refining class separation

## 🎯 Key Features

- **Multi-backbone Support**: Custom CNN, ResNet-18/34/50, EfficientNet-B0/B1
- **Two-Phase Training**: Triplet Loss → Augmented Triplet Loss
- **Class Imbalance Handling**: Weighted sampling and data augmentation
- **Comprehensive Evaluation**: K-NN classification, confusion matrix, t-SNE visualization
- **TensorBoard Integration**: Real-time training monitoring
- **HEIC Image Support**: Automatic conversion from Apple HEIC format

## 📊 Dataset

- **Total Deer**: 50 individuals
- **Total Images**: 1,169 images
- **Age Groups**: 7 classes (ages 2, 3, 4, 5, 6, 7, 8)
- **Image Categories**: Antler, frontal, head closeup, left/right side, rear views
- **Formats**: HEIC, JPG

### Age Distribution
```
Age 2: 12 deer (24%)
Age 3: 15 deer (30%)
Age 4:  5 deer (10%)
Age 5:  9 deer (18%)
Age 6:  3 deer ( 6%)
Age 7:  1 deer ( 2%)
Age 8:  4 deer ( 8%)
```

## 🏗️ Project Structure

```
atn/
├── data/
│   ├── raw/                    # Converted JPG images
│   ├── processed/              # Preprocessed data
│   └── augmented/              # Augmented data
├── models/
│   ├── __init__.py
│   ├── feature_extractor.py   # CNN backbone (Custom/ResNet/EfficientNet)
│   └── atn_loss.py            # Triplet & ATN loss functions
├── utils/
│   ├── __init__.py
│   ├── image_converter.py     # HEIC to JPG converter
│   └── data_loader.py         # Dataset & DataLoader
├── training/
│   ├── __init__.py
│   ├── config.py              # Training configuration
│   ├── train.py               # Main training script
│   └── evaluate.py            # Evaluation script
├── checkpoints/               # Saved model weights
├── logs/                      # TensorBoard logs
├── results/                   # Evaluation results & plots
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Convert HEIC Images to JPG

```bash
python utils/image_converter.py
```

This will convert all HEIC images from `deer data/` to `data/raw/` in JPG format.

### 3. Train the Model

```bash
python training/train.py
```

Training consists of two phases:
- **Phase 1** (50 epochs): Standard Triplet Loss
- **Phase 2** (50 epochs): Augmented Triplet Loss

### 4. Evaluate the Model

```bash
python training/evaluate.py
```

Generates:
- Classification accuracy
- Confusion matrix
- t-SNE embedding visualization
- Per-class metrics (precision, recall, F1-score)
- Training curves

## ⚙️ Configuration

Edit `training/config.py` to customize:

### Model Architecture
```python
EMBEDDING_DIM = 128           # Embedding dimension
BACKBONE = 'resnet18'         # 'custom_cnn', 'resnet18', 'resnet50', etc.
PRETRAINED = True             # Use ImageNet pretrained weights
```

### Training Hyperparameters
```python
PHASE1_EPOCHS = 50            # Phase 1 epochs
PHASE2_EPOCHS = 50            # Phase 2 epochs
PHASE1_BATCH_SIZE = 32        # Batch size
PHASE1_LEARNING_RATE = 0.001  # Learning rate
```

### Triplet Loss
```python
TRIPLET_MARGIN = 0.2          # Triplet loss margin
DISTANCE_METRIC = 'cosine'    # 'cosine' or 'euclidean'
```

### ATN Loss
```python
ATN_ALPHA = 0.1               # Max distance for same-class to centroid
ATN_BETA = 0.3                # Min distance for different-class to centroid
```

## 📈 Model Architecture

### Feature Extractor
```
Input (224x224x3)
    ↓
Backbone (ResNet-18/Custom CNN)
    ↓
Embedding Layer (128-dim)
    ↓
L2 Normalization
    ↓
Output Embeddings
```

### Loss Functions

**Standard Triplet Loss:**
```
L = max(0, d(anchor, positive) - d(anchor, negative) + margin)
```

**Augmented Triplet Loss (ATN):**
- Computes class centroids (dummy anchors)
- Finds close class pairs (distance < β)
- Applies constraints:
  - Intra-class: d(centroid, same_class_sample) ≤ α
  - Inter-class: d(centroid, diff_class_sample) ≥ β

## 📊 Expected Results

### Performance Targets
- **Age Classification Accuracy**: >85%
- **Embedding Quality**: Clear age group clusters in t-SNE
- **Training Time**: ~4-6 hours (with GPU)

### Outputs
1. **Trained Models**: `checkpoints/best_model.pth`, `checkpoints/final_model.pth`
2. **Visualizations**:
   - `results/confusion_matrix.png`
   - `results/tsne_visualization.png`
   - `results/per_class_metrics.png`
   - `results/training_curves.png`
3. **Metrics**: `results/evaluation_results.json`

## 🔬 Research Paper Implementation

This implementation is based on:

**Title**: "Augmented Triplet Network for Individual Organism and Unique Object Classification for Reliable Monitoring of Ezoshika Deer"

**Key Differences from Paper**:
| Aspect | Paper | Our Implementation |
|--------|-------|-------------------|
| Task | Individual deer ID | Age classification |
| Classes | 3 deer | 7 age groups |
| Dataset | 1,400 images | 1,169 images |
| Image Size | 32×32 | 224×224 |
| Embedding Dim | 10 | 128 |
| Backbone | Simple 2-layer CNN | ResNet-18/Custom CNN |

## 🛠️ Advanced Usage

### Resume Training from Checkpoint

```python
# In train.py, before Phase 2
trainer.load_checkpoint('checkpoint_epoch_25.pth')
```

### Use Different Backbone

```python
# In config.py
BACKBONE = 'resnet50'  # or 'efficientnet_b0', 'custom_cnn'
```

### Monitor Training with TensorBoard

```bash
tensorboard --logdir logs/
```

### Evaluate on Custom Images

```python
from models.feature_extractor import create_feature_extractor
from PIL import Image
from torchvision import transforms

# Load model
model = create_feature_extractor(embedding_dim=128, backbone='resnet18')
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img = Image.open('deer_image.jpg').convert('RGB')
img_tensor = transform(img).unsqueeze(0)

# Get embedding
embedding = model(img_tensor)
```

## 📝 Key Challenges & Solutions

### 1. HEIC Format
- **Challenge**: Images in Apple's HEIC format
- **Solution**: Automatic conversion using `pillow-heif`

### 2. Class Imbalance
- **Challenge**: Age 7 has only 1 deer, Age 3 has 15 deer
- **Solution**: Weighted sampling + data augmentation

### 3. Limited Data
- **Challenge**: Average 23 images per deer
- **Solution**: Transfer learning with pretrained ResNet + aggressive augmentation

### 4. Similar Ages
- **Challenge**: Ages 3-5 visually similar
- **Solution**: ATN loss to separate close classes using dummy anchors

## 📚 Dependencies

- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- pillow-heif >= 0.10.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- tensorboard >= 2.13.0
- tqdm >= 4.65.0

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Multi-view fusion (combine different angles)
- Attention mechanisms
- Semi-supervised learning
- Real-time inference optimization

## 📄 License

This project is for research and educational purposes.

## 🙏 Acknowledgments

Based on the research paper by Harie et al. on Augmented Triplet Networks for wildlife monitoring.

---

**Status**: ✅ Implementation Complete | 🚀 Ready for Training

For questions or issues, please refer to the documentation or create an issue.
