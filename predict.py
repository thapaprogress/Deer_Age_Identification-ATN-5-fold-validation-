"""
Standalone script for Deer Age Prediction
Supports single image, directory, and multi-view fusion.
"""

import argparse
from pathlib import Path
from PIL import Image
import os
import sys

# Ensure root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.inference import InferenceEngine
from utils.gradcam import GradCAM, overlay_heatmap
import cv2
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="ATN Deer Age Prediction")
    parser.add_argument("--image", type=str, help="Path to single image")
    parser.add_argument("--dir", type=str, help="Path to directory of images")
    parser.add_argument("--model", type=str, default="checkpoints/best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--fusion", action="store_true", help="Enable Multi-View Fusion (average embeddings for dir)")
    parser.add_argument("--visualize", action="store_true", help="Save Grad-CAM attention map")
    parser.add_argument("--output", type=str, default="results", help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    # Initialize Engine
    print("Initializing Inference Engine...")
    engine = InferenceEngine(model_path=args.model)
    
    # Prepare output dir
    if args.visualize:
        os.makedirs(args.output, exist_ok=True)
        # Initialize GradCAM
        # Hook into last layer of ResNet backbone (layer4)
        # Structure: model.backbone.resnet[-2]
        try:
            target_layer = engine.model.backbone.resnet[-2][-1] # Last block of layer4
            gradcam = GradCAM(engine.model, target_layer)
            print("Grad-CAM initialized.")
        except Exception as e:
            print(f"Warning: Could not initialize Grad-CAM: {e}")
            gradcam = None
    
    # Process Single Image
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image {args.image} not found.")
            return
            
        print(f"\nProcessing {args.image}...")
        img = Image.open(args.image).convert("RGB")
        age, conf, _ = engine.predict(img)
        
        print(f"Predicted Age: {age} years")
        print(f"Confidence: {conf:.2%}")
        
        if args.visualize and gradcam:
            # Prepare tensor
            img_tensor = engine.transform(img).unsqueeze(0).to(engine.device)
            heatmap = gradcam(img_tensor)
            
            # Save
            vis_path = os.path.join(args.output, f"gradcam_{Path(args.image).name}")
            result = overlay_heatmap(args.image, heatmap)
            cv2.imwrite(vis_path, result)
            print(f"Saved visualization to {vis_path}")
            
    # Process Directory
    elif args.dir:
        if not os.path.exists(args.dir):
            print(f"Error: Directory {args.dir} not found.")
            return
            
        image_files = [f for f in Path(args.dir).glob("*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.heic']]
        print(f"\nFound {len(image_files)} images in {args.dir}")
        
        if args.fusion:
            print("Running in Multi-View Fusion Mode...")
            images = [Image.open(f).convert("RGB") for f in image_files]
            age, conf = engine.predict_multi_view(images)
            print(f"Agregated Prediction (Fusion): {age} years")
            print(f"Confidence: {conf:.2%}")
        else:
            print("Running Individual Predictions...")
            for f in image_files:
                img = Image.open(f).convert("RGB")
                age, conf, _ = engine.predict(img)
                print(f"{f.name}: Age {age} ({conf:.2%})")

    else:
        print("Please provide --image or --dir")

if __name__ == "__main__":
    main()
