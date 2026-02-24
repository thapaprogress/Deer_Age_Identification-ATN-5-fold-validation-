
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path
from tqdm import tqdm
import re

# Ensure root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.inference import InferenceEngine
from training import config

def add_gaussian_noise(image, severity=1):
    """Add Gaussian noise to an image"""
    img_array = np.array(image)
    noise_sigma = severity * 10  # 10, 20, 30, 40, 50
    noise = np.random.normal(0, noise_sigma, img_array.shape)
    noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

def add_gaussian_blur(image, severity=1):
    """Add Gaussian blur to an image"""
    radius = severity * 1.0 # 1, 2, 3, 4, 5
    return image.filter(ImageFilter.GaussianBlur(radius))

def adjust_brightness(image, severity=1):
    """Adjust brightness (lower is darker/harder)"""
    # severity 1-5 maps to factor 0.9 to 0.1
    factor = 1.0 - (severity * 0.15) 
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def adjust_contrast(image, severity=1):
    """Adjust contrast"""
    # severity 1-5 maps to factor 0.9 to 0.1
    factor = 1.0 - (severity * 0.15)
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def add_occlusion(image, severity=1):
    """Add black squared occlusion"""
    img_array = np.array(image)
    h, w, _ = img_array.shape
    
    # Size of occlusion increases with severity
    box_size = int(min(h, w) * (severity * 0.1)) # 10% to 50%
    
    # Random position
    top = np.random.randint(0, h - box_size)
    left = np.random.randint(0, w - box_size)
    
    img_array[top:top+box_size, left:left+box_size, :] = 0
    return Image.fromarray(img_array)

def evaluate_robustness():
    print("Initializing Robustness Analysis...", flush=True)
    engine = InferenceEngine(model_path=str(config.CHECKPOINT_DIR / "best_model.pth"))
    
    # Collect test images and labels
    image_paths = []
    labels = []
    
    # Walk through data directory
    target_dir = config.RAW_DATA_DIR
    print(f"Scanning {target_dir}...", flush=True)
    
    try:
        items = os.listdir(target_dir)
        print(f"DEBUG: Found {len(items)} items in directory", flush=True)
    except Exception as e:
        print(f"Error listing directory: {e}", flush=True)
        return

    for folder_name in items:
        folder_path = target_dir / folder_name
        if not folder_path.is_dir():
            continue
            
        print(f"Checking folder: {folder_name}", flush=True)
        # Parse age from folder name "Deer_id_1 (age 5)"
        match = re.search(r'age\s*(\d+)', folder_name, re.IGNORECASE)
        if match:
            age = int(match.group(1))
            print(f"  -> Matched Age: {age}", flush=True)
            
            # Find images (Recursively using rglob)
            files = list(folder_path.rglob("*.jpg")) + list(folder_path.rglob("*.png")) + list(folder_path.rglob("*.jpeg"))
            print(f"  -> Found {len(files)} images", flush=True)
            
            # Take a subset for testing
            num_test = max(1, int(len(files) * 0.2)) 
            test_files = files[-num_test:] if files else []
            
            for f in test_files:
                image_paths.append(f)
                labels.append(age)
        else:
            print(f"  -> No regex match for age in '{folder_name}'", flush=True)

    if not image_paths:
        print("No images found for testing.", flush=True)
        return

    print(f"Found {len(image_paths)} images for testing.", flush=True)

    perturbations = {
        "Gaussian Noise": add_gaussian_noise,
        "Gaussian Blur": add_gaussian_blur,
        "Brightness": adjust_brightness,
        "Contrast": adjust_contrast,
        "Occlusion": add_occlusion
    }
    
    severities = [0, 1, 2, 3, 4, 5]
    results = {name: [] for name in perturbations.keys()}
    
    # Baseline (Severity 0)
    print("\nEvaluating Baseline (Severity 0)...", flush=True)
    correct = 0
    total = len(image_paths)
    
    for i, path in enumerate(tqdm(image_paths)):
        try:
            img = Image.open(path).convert("RGB")
            pred, _, _ = engine.predict(img)
            if pred == labels[i]:
                correct += 1
        except Exception as e:
            print(f"Error processing {path}: {e}", flush=True)
            
    baseline_acc = correct / total
    print(f"Baseline Accuracy: {baseline_acc:.2%}", flush=True)
    
    # Add baseline to all
    for name in perturbations:
        results[name].append(baseline_acc)
        
    # Evaluate Perturbations
    for name, func in perturbations.items():
        print(f"\nEvaluating {name}...", flush=True)
        for severity in severities[1:]: # 1 to 5
            correct = 0
            for i, path in enumerate(tqdm(image_paths, leave=False)):
                try:
                    img = Image.open(path).convert("RGB")
                    # Apply perturbation
                    distorted_img = func(img, severity)
                    
                    pred, _, _ = engine.predict(distorted_img)
                    if pred == labels[i]:
                        correct += 1
                except Exception as e:
                    pass
            
            acc = correct / total
            results[name].append(acc)
            print(f"Severity {severity}: {acc:.2%}", flush=True)

    # Plotting
    plt.figure(figsize=(10, 8))
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, (name, accs) in enumerate(results.items()):
        plt.plot(severities, accs, marker=markers[i], linewidth=2, label=name)
        
    plt.title("Model Robustness Analysis", fontsize=16, fontweight='bold')
    plt.xlabel("Severity Level (0-5)", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    out_path = config.RESULTS_DIR / "robustness_analysis.png"
    plt.savefig(out_path, dpi=300)
    print(f"\nRobustness plot saved to {out_path}", flush=True)

if __name__ == "__main__":
    evaluate_robustness()
