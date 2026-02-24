"""
Export trained PyTorch model to ONNX format
"""

import torch
import torch.onnx
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.feature_extractor import create_feature_extractor
from training import config

def export_to_onnx(checkpoint_path, output_path, device='cpu'):
    """
    Export model to ONNX
    
    Args:
        checkpoint_path: Path to best_model.pth
        output_path: Where to save .onnx file
        device: 'cpu' or 'cuda'
    """
    print(f"Loading model from {checkpoint_path}...")
    
    # Create model
    model = create_feature_extractor(
        embedding_dim=config.EMBEDDING_DIM,
        backbone=config.BACKBONE,
        pretrained=False,
        device=device
    )
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE, device=device)
    
    # Export
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['embedding'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'embedding': {0: 'batch_size'}
        }
    )
    print("Export successful!")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export ATN model to ONNX")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Path to checkpoint")
    parser.add_argument("--output", type=str, default="checkpoints/model.onnx", help="Output path")
    args = parser.parse_args()
    
    if os.path.exists(args.checkpoint):
        export_to_onnx(args.checkpoint, args.output)
    else:
        print(f"Checkpoint not found: {args.checkpoint}")
