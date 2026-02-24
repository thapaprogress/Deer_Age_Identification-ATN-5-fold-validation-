# ATN Deer Age Recognition - Manual Execution Guide

This guide provides step-by-step terminal commands to set up, train, and evaluate the Deer Age Recognition System from scratch.

## 📋 Prerequisites

- **OS**: Windows (Command Prompt or PowerShell)
- **Python**: 3.8 or higher
- **GPU**: NVIDIA GPU with CUDA drivers (Recommended) / CPU (Supported but slow)

---

## 🚀 Step 1: Environment Setup

First, ensure you are in the project root directory:

```powershell
cd "c:\Users\PRAJNA WORLD TECH\OneDrive\Desktop\atn"
```

### Install Dependencies

Install all required Python packages. We recommend using the CUDA-enabled version of PyTorch for faster training.

```powershell
# 1. Uninstall any existing PyTorch versions to avoid conflicts
pip uninstall -y torch torchvision

# 2. Install PyTorch with CUDA 11.8 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. Install remaining dependencies from requirements.txt
pip install -r requirements.txt
```

### Verify Installation

Run this script to confirm your GPU is detected:

```powershell
python verify_pytorch.py
```

**Expected Output:**
> CUDA available: True
> GPU Device: NVIDIA GeForce RTX 4070 (or your GPU)

---

## 🖼️ Step 2: Data Preprocessing

The system works with JPG images. We need to convert the provided HEIC images first.

```powershell
# Run the image converter utility
python utils/image_converter.py
```

**What this does:**
- Scans `deer data/` for HEIC and JPG images.
- Converts HEIC files to JPG format.
- Saves processed images to `data/raw/` preserving the directory structure.
- **Note**: This step may take a few minutes (processed ~3,580 images).

---

## 🏋️ Step 3: Train the Model

Start the two-phase training process. This script handles everything automatically:

```powershell
python training/train.py
```

**Training Process Overview:**
1.  **Phase 1 (Epochs 1-50)**: Trains using Standard Triplet Loss.
2.  **Phase 2 (Epochs 51-100)**: Switches to Augmented Triplet Loss (ATN) for fine-tuning.

**Key Outputs:**
- Logs are saved to `logs/`.
- Checkpoints are saved to `checkpoints/`.
- Best model is always saved as `checkpoints/best_model.pth`.

---

## 📈 Step 4: Monitor Training

You can visualize loss curves and metrics in real-time using TensorBoard.

Open a **new terminal window** in the project folder and run:

```powershell
tensorboard --logdir logs/
```

Then open your browser and go to: [http://localhost:6006](http://localhost:6006)

---

## 📊 Step 5: Evaluate Performance

Once training is complete (or to test an intermediate checkpoint), run the evaluation script:

```powershell
python training/evaluate.py
```

**What this generates in `results/`:**
- `confusion_matrix.png`: Shows which ages are being confused.
- `tsne_visualization.png`: 2D plot showing how well age groups are separated.
- `per_class_metrics.png`: Precision/Recall for each age.
- `evaluation_results.json`: Raw accuracy numbers.

---

## 🛠️ Troubleshooting Common Issues

### 1. "CUDA not available"
- **Cause**: Wrong PyTorch version or missing drivers.
- **Fix**: Re-run the PyTorch installation command in Step 1. Ensure you have NVIDIA drivers installed.

### 2. "Out of Memory" (OOM)
- **Cause**: Batch size too large for your GPU.
- **Fix**: Open `training/config.py` and reduce `PHASE1_BATCH_SIZE` and `PHASE2_BATCH_SIZE` (e.g., change 32 to 16).

### 3. "File not found" errors
- **Cause**: Running commands from the wrong directory.
- **Fix**: Always ensure you are in the root `atn` folder before running scripts.

---

## 📝 Quick Command Cheat Sheet

```powershell
# 1. Setup
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# 2. Data
python utils/image_converter.py

# 3. Train
python training/train.py

# 4. Monitor (New Terminal)
tensorboard --logdir logs/

# 5. Evaluate
python training/evaluate.py
```
