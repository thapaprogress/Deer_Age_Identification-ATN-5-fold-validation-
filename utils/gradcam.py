"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Implementation
Visualizes where the model is looking.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer):
        """
        Args:
            model: The PyTorch model
            target_layer: The layer to hook into (e.g. last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        """
        Compute Grad-CAM heatmap
        """
        self.model.eval()
        self.model.zero_grad()
        
        # Forward pass
        # If it's a FeatureExtractor, we need raw features before L2 normalization
        # because normalized vectors have constant energy (length=1) -> zero gradients.
        from models.feature_extractor import FeatureExtractor
        if isinstance(self.model, FeatureExtractor):
            output = self.model.backbone(x) # Get raw embeddings from backbone
        else:
            output = self.model(x)
        
        # For metric learning or embedding models, we maximize the 'energy' 
        # of the representation to highlight important features.
        score = torch.sum(output**2)
        score.backward()
        
        # Weights: Global average of gradients per channel
        weights = torch.mean(self.gradients, dim=[2, 3]) # (B, C)
        
        # Weighted sum of activations
        activations = self.activations.detach()
        # Compute heatmap: sum (weights * activations) across channels
        heatmap = torch.sum(weights.view(weights.size(0), weights.size(1), 1, 1) * activations, dim=1)
        heatmap = heatmap.squeeze(0) # Remove batch dimension
            
        # ReLU to keep only positive contributions
        heatmap = F.relu(heatmap)
        
        # Normalize heatmap to [0, 1]
        heatmap = heatmap.cpu().numpy()
        h_min, h_max = np.min(heatmap), np.max(heatmap)
        if h_max > h_min:
            heatmap = (heatmap - h_min) / (h_max - h_min)
        else:
            heatmap = np.zeros_like(heatmap)
        
        return heatmap

def overlay_heatmap(img_path, heatmap, alpha=0.6, colormap=cv2.COLORMAP_JET):
    """
    Overlay heatmap on original image while preserving aspect ratio
    """
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    h, w, _ = img.shape
    
    # Resize heatmap to match original image dimensions
    heatmap_resized = cv2.resize(heatmap, (w, h))
    
    # 8-bit colormap conversion
    heatmap_8bit = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_8bit, colormap)
    
    # Blend using cv2.addWeighted for better color saturation
    superimposed_img = cv2.addWeighted(heatmap_color, alpha, img, 1 - alpha, 0)
    
    return superimposed_img
