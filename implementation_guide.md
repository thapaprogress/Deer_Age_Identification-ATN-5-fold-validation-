# ATN Deer Age Recognition System: Implementation Guide

This guide provides a detailed walkthrough for setting up, training, evaluating, and deploying the ATN (Augmented Triplet Network) Deer Age Recognition System. This system is designed for high-accuracy wildlife monitoring, utilizing advanced deep learning techniques inspired by the latest research.

---

## 📂 Table of Contents
1. [Prerequisites & Setup](#1-prerequisites--setup)
2. [Data Preparation](#2-data-preparation)
3. [Training Pipeline](#3-training-pipeline)
   - [A. Standard Training (Single Model)](#a-standard-training-single-model)
   - [B. Professional K-Fold Ensemble Training](#b-professional-k-fold-ensemble-training)
4. [Testing & Evaluation](#4-testing--evaluation)
   - [A. Single Model Evaluation](#a-single-model-evaluation)
   - [B. Ensemble Evaluation (High Accuracy)](#b-ensemble-evaluation-high-accuracy)
5. [Dashboard App Operations](#5-dashboard-app-operations)
   - [A. Classic Dashboard (`app.py`)](#a-classic-dashboard-apppy)
   - [B. Research Dashboard (`app1.py`)](#b-research-dashboard-app1py)
6. [Advanced Technical Features](#6-advanced-technical-features)

---

## 1. Prerequisites & Setup

### Environment Requirements
- **Hardware**: NVIDIA GPU (8GB+ VRAM recommended) with CUDA installed.
- **Python**: 3.9+ (Python 3.13 supported).

### Installation
Run the following command to install all necessary libraries:
```powershell
pip install torch torchvision torchaudio numpy pandas matplotlib seaborn scikit-learn streamlit plotly pillow-heif tqdm
```

---

## 2. Data Preparation

Before training, you must convert the raw dataset into a format the model understands.

### Convert HEIC to JPG
Most deer datasets from mobile devices use HEIC format. Run the converter:
```powershell
python utils/image_converter.py
```
- **Source**: `deer data/`
- **Output**: `data/raw/` (standardized JPGs)

---

## 3. Training Pipeline

The system uses a **Two-Phase Training Protocol** to maximize class separation.

### A. Standard Training (Single Model)
Use this for quick experimentation or baseline results.
```powershell
python training/train.py
```
- **Phase 1 (Triplet Loss)**: Learns general features and clusters similar ages.
- **Phase 2 (ATN Loss)**: Refines clusters using "Dummy Anchors" to solve class boundaries.

### B. Professional K-Fold Ensemble Training
This is the **Research-Grade** mode. It trains 5 separate models on different data folds to reduce bias and maximize accuracy.
```powershell
python training/train.py --kfold
```
- **Output**: Generates 5 checkpoints (`best_model_fold_1.pth` ... `best_model_fold_5.pth`).
- **Advantage**: Provides the most robust predictions by averaging multiple expert "opinions."

---

## 4. Testing & Evaluation

Once training is complete, characterize the performance using the evaluation suite.

### A. Single Model Evaluation
Evaluates your `best_model.pth`.
```powershell
python training/evaluate.py
```

### B. Ensemble Evaluation (High Accuracy)
Specifically designed to test the power of your 5-fold models.
```powershell
python training/evaluate.py --ensemble
```
- **Result Isolation**: All ensemble graphs and metrics are saved separately in `results/ensemble_results/`.
- **Metrics Generated**: Accuracy, JSON result logs, Confusion Matrix, and t-SNE Clustering maps.

---

## 5. Dashboard App Operations

We provide two distinct dashboard experiences depending on your goals.

### A. Classic Dashboard (`app.py`)
**Best for**: Fast viewing and basic field monitoring.
```powershell
streamlit run dashboard/app.py
```
- Uses the standard single model.
- Minimalist UI for quick results.

### B. Research Dashboard (`app1.py`)
**Best for**: Scientific validation and detailed analysis.
- **🧬 Ensemble Engine**: Automatically detects if 5 folds exist and fuses them.
- **🏆 Readiness Benchmarks**: Color-coded grades based on Scientific standards.
- **🔍 3-Panel Interpretability**: Original Image | Attention Map | Scientific Overlay.

### C. Architecture Deep-Dive (`app2.py`)
**Best for**: Understanding the AI internal logic and "Blueprints".
- **🧬 K-Fold Visualizer**: Shows how the 5 models fuse data.
- **📊 Training Protocol**: Visual breakdown of Phase 1 vs Phase 2.
- **⚙️ Parameter Specification**: Live view of all `config.py` settings.

---

## 6. Advanced Technical Features

### Test-Time Augmentation (TTA)
Activated in predicted modes (Dashboard/Research), the model analyzes both the original and flipped versions of every image, averaging the result for **+2-3% stability**.

### Differential Learning Rates
The training script automatically uses a slower learning rate for the ResNet backbone (ResNet weights are preserved) and a faster rate for the ATN head (where deer age specialization happens).

### Grad-CAM Interpretability
Visualizes exactly what the model "sees" (e.g., antler shape, jaw length) to provide biologically grounded explanations for its predictions.

---
**Status**: 🚀 System Ready | **Stability**: High | **Version**: 2.5.0 (Research Alignment)
