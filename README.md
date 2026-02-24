# 🦌 ATN Deer Age Recognition System
### Augmented Triplet Network with 5-Fold Ensemble Validation

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Research-blue.svg)](LICENSE)

An advanced deep learning system for high-accuracy wildlife monitoring, specializing in **Deer Age Recognition**. This project implements and extends the **Augmented Triplet Network (ATN)** architecture across two distinct research phases.

---

## 🔬 Scientific Foundation

### 1. The First ATN: Individual Identification (Core Paper)
Based on the research: *"Augmented Triplet Network for Individual Organism and Unique Object Classification for Reliable Monitoring of Ezoshika Deer"*.
*   **Innovation**: Introduced **Dummy Anchors (Class Centroids)** to the standard triplet network.
*   **Result**: Achieved **99.64% accuracy** in identifying individual deer.
*   **Challenge**: While excellent for ID, it required generalization for biological categories (Age).

### 2. 5-Fold Validation & Generalization (The Next Paper)
Extending ATN for **Age Group Classification** through robust statistical validation.
*   **Innovation**: Implementation of a **Professional 5-Fold Cross-Validation** ensemble.
*   **Result**: Significant improvement in generalization across previously unseen individuals.
*   **Accuracy**: Target accuracy of **>85%** for management decisions and **>80%** for peer-reviewed research.
*   **Fusion**: Integration of **Ensemble Inference** and **Test-Time Augmentation (TTA)** for +3% stability.

---

## 🛠️ Core Technology: The ATN Protocol

The system solves the "Visual Similarity" problem where deer aged 3, 4, and 5 look remarkably similar to standard CNNs.

### The Two-Phase Training Strategy
1.  **Phase 1: Triplet Discovery**: Standard Triplet Loss initializes the embedding space, clustering similar age groups together.
2.  **Phase 2: ATN Refinement**: Automatically identifies "Close Classes" and applies **Augmented Constraints** using dummy anchors to push boundary classes apart.

### 🧬 Professional 5-Fold Ensemble
Instead of relying on a single model, the system trains **5 independent experts** on different data folds.
*   **Majority Voting**: Predictions are fused from all 5 models.
*   **Confidence Scoring**: Provides a "Reliability Delta" for every prediction.
*   **Edge Case Handling**: Models specializing in "Young" vs "Senior" deer complement each other.

---

## 📊 Interactive Dashboards

### 💻 Standard Dashboard (`app.py`)
*Best for: Quick field monitoring and baseline tests.*
- Single-model inference.
- Real-time video tracking.
- Simple age gauge and confidence metrics.

### 🧬 Advanced Research Dashboard (`app1.py`)
*Best for: Scientific validation and interpretability.*
- **Ensemble Engine**: Fuses 5-Fold weights automatically.
- **Grad-CAM 3-Panel View**: Original | Attention Map | Scientific Overlay.
- **Readiness Benchmarks**: Color-coded grades (Research vs. Management Grade).
- **TTA Integration**: Predicts across horizontal flips for max stability.

---

## � Dataset & Model Weights

Due to GitHub's file size limits (100MB per file) and to keep the repository lightweight, the raw dataset (~7.4GB) and pre-trained model checkpoints are hosted externally.

### 📥 Download Links
*   **Raw Dataset (7.4GB)**: [🔗 Download from OneDrive](https://mbustb-my.sharepoint.com/:f:/g/personal/progress_jung_thapa_mbust_edu_np/IgA3CNY1W8elR6yZyTNvgvRgAXMzT3a5ydbwe4Lifuu67xE?e=1V2ye6)
*   **Pre-trained Checkpoints**: [🔗 Download Trained Weights](https://mbustb-my.sharepoint.com/:f:/g/personal/progress_jung_thapa_mbust_edu_np/IgDLFVOR6yZnR4BmlAkgfYsGATC3w2oGnhz3-Gm5xaRjBzw?e=HfhrO3)

### 📂 Setup & Data Walkthrough

To ensure the scripts can find your data and models, follow this specific directory structure.

#### 1. Initial Setup
Place your downloaded files so the root directory looks like this:
```text
atn/
├── deer data/             <-- Place extracted raw images here
│   ├── Deer_id_1 (age 5)/
│   ├── Deer_id_2 (age 3)/
│   └── ...
├── checkpoints/           <-- Place .pth files here
├── training/              <-- (Already in Repo)
├── utils/                 <-- (Already in Repo)
└── dashboard/             <-- (Already in Repo)
```

#### 2. Data Pipeline Walkthrough
The system follows a strict pipeline to prepare raw wildlife data for the ATN model:

1.  **Stage 1: Raw Ingestion**: The system looks into `deer data/`. This folder contains original high-resolution images, often in `.HEIC` format from mobile devices.
2.  **Stage 2: Standardization**: Running `python utils/image_converter.py` performs three tasks:
    *   Finds every `.HEIC` and `.JPG` file recursively.
    *   Standardizes filenames and converts everything to `.JPG`.
    *   Creates a new folder `data/raw/` where the processed images are stored.
3.  **Stage 3: Dataset Mapping**: The `utils/data_loader.py` then reads from `data/raw/`, mapping the folder names (e.g., "age 5") to integer labels for the neural network.

> **Pro Tip**: Keep the folder names in `deer data/` exactly as they were (containing the word "age X") so the regex parser can correctly identify the age labels.

---

## �🚀 Detailed Implementation Guide

### 1. Prerequisites & Installation
Ensure you have an NVIDIA GPU for optimal training performance.
```bash
# Clone the repository
git clone https://github.com/thapaprogress/Deer_Age_Identification-ATN-5-fold-validation-
cd Deer_Age_Identification-ATN-5-fold-validation-

# Install mandatory libraries
pip install torch torchvision torchaudio numpy pandas matplotlib seaborn streamlit plotly pillow-heif tqdm
```

### 2. Data Standardizing (HEIC ➔ JPG)
Wilderness datasets often come in Apple's HEIC format. Convert them before training:
```bash
python utils/image_converter.py
```
*   **Input**: `deer data/`
*   **Output**: `data/raw/`

### 3. Launching the Training Suite
**Mode A: Standard Single Training**
```bash
python training/train.py
```

**Mode B: Research-Grade 5-Fold Training (Recommended)**
```bash
python training/train.py --kfold
```
*Creates 5 checkpoints (`best_model_fold_1.pth` ... `best_model_fold_5.pth`)*

### 4. Comprehensive Evaluation
Evaluate using the high-accuracy ensemble suite:
```bash
python training/evaluate.py --ensemble
```
*Outputs confusion matrices and t-SNE maps to `results/ensemble_results/`.*

### 5. Running the Deployment Apps
```bash
# For Standard UI
streamlit run dashboard/app.py

# For Advanced Research Mode
streamlit run dashboard/app1.py
```

---

## ⚙️ Advanced Technical Specs
- **Backbone**: ResNet-18/50 with Differential Learning Rates.
- **Distance Metric**: Cosine Similarity.
- **Embedding Space**: 128-Dimensional Age-Features.
- **Interpretability**: Layer-wise Grad-CAM for biological marker identification.

---
**Status**: 🚀 System Ready | **Stability**: High | **Version**: 2.5.0
