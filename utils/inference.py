import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import sys
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier

from models.feature_extractor import create_feature_extractor
from utils.data_loader import create_data_loaders
from training import config

class InferenceEngine:
    def __init__(self, model_path=None, device='cuda'):
        self.device = 'cuda' if torch.cuda.is_available() and device=='cuda' else 'cpu'
        
        # Load Model
        self.model = create_feature_extractor(
            embedding_dim=config.EMBEDDING_DIM,
            backbone=config.BACKBONE,
            pretrained=False, # Weights loaded from checkpoint
            device=self.device
        )
        
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.epoch = checkpoint['epoch']
            print(f"Loaded model from epoch {self.epoch}")
        elif model_path:
             print(f"Warning: Model path {model_path} not found. Using random weights.")
             self.epoch = 0
        else:
            print("Warning: No model path provided. Using random weights.")
            self.epoch = 0
            
        self.model.eval()
        
        # Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize KNN reference
        self.knn = None
        self.reference_images = []
        self.reference_labels = []
        self._initialize_reference_data()
        
    def _initialize_reference_data(self):
        """Load training data and compute embeddings for KNN"""
        print("Initializing reference data...")
        try:
            # We use the training set as reference
            data_loaders = create_data_loaders(
                data_dir=config.RAW_DATA_DIR,
                batch_size=32,
                image_size=config.IMAGE_SIZE,
                num_workers=0, # Avoid multiproc issues in streamlit
                return_triplets=False
            )
            train_loader = data_loaders['train']
            
            embeddings = []
            labels = []
            
            with torch.no_grad():
                for images, batch_labels in train_loader:
                    images = images.to(self.device)
                    emb = self.model(images)
                    embeddings.append(emb.cpu().numpy())
                    labels.append(batch_labels.numpy())
            
            if embeddings:
                self.reference_embeddings = np.concatenate(embeddings)
                self.reference_labels = np.concatenate(labels)
                
                # Fit KNN
                self.knn = KNeighborsClassifier(n_neighbors=config.KNN_K, metric='cosine')
                self.knn.fit(self.reference_embeddings, self.reference_labels)
                print(f"Reference data initialized: {len(self.reference_labels)} samples")
            else:
                print("Warning: No training data found.")
                
        except Exception as e:
            print(f"Error initializing reference data: {e}")

    def predict(self, image, use_tta=True):
        """
        Predict age for a single PIL Image
        
        Args:
            image: PIL Image
            use_tta: Use Test-Time Augmentation (average original + flipped)
        
        Returns: predicted_age, confidence, embedding
        """
        if self.knn is None:
            return None, 0.0, None

        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model(img_tensor)
            
            if use_tta:
                # Flip image horizontally
                flipped_img = transforms.functional.hflip(image)
                flipped_tensor = self.transform(flipped_img).unsqueeze(0).to(self.device)
                flipped_emb = self.model(flipped_tensor)
                
                # Average embeddings
                combined_emb = (embedding + flipped_emb) / 2
                # Re-normalize to unit sphere (L2 norm)
                embedding = torch.nn.functional.normalize(combined_emb, p=2, dim=1)
            
            embedding = embedding.cpu().numpy()
            
        # Predict using KNN
        predicted_age = self.knn.predict(embedding)[0]
        probs = self.knn.predict_proba(embedding)[0]
        confidence = np.max(probs)
        
        return int(predicted_age), float(confidence), embedding

    def predict_multi_view(self, images):
        """
        Predict age from multiple images of the same deer (Fusion)
        Averages the embeddings before classification.
        
        Args:
            images: List of PIL Images
        """
        if self.knn is None or not images:
            return None, 0.0
            
        tensors = [self.transform(img).unsqueeze(0) for img in images]
        batch = torch.cat(tensors).to(self.device)
        
        with torch.no_grad():
            embeddings = self.model(batch)
            # Average embeddings (Fusion)
            # Since vectors are normalized, simple average direction is good
            avg_embedding = torch.mean(embeddings, dim=0, keepdim=True)
            # Re-normalize to unit sphere
            avg_embedding = torch.nn.functional.normalize(avg_embedding, p=2, dim=1)
            avg_embedding = avg_embedding.cpu().numpy()
            
        # Predict
        predicted_age = self.knn.predict(avg_embedding)[0]
        probs = self.knn.predict_proba(avg_embedding)[0]
        confidence = np.max(probs)
        
        return int(predicted_age), float(confidence)

    def get_neighbors(self, embedding, k=3):
        """Find nearest neighbors in embedding space"""
        if self.knn is None:
            return []
            
        # Ensure embedding is 2D
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
            
        distances, indices = self.knn.kneighbors(embedding, n_neighbors=k)
        
        # Return list of (index, distance, label)
        neighbors = []
        for i, idx in enumerate(indices[0]):
            neighbors.append({
                'index': int(idx),
                'distance': float(distances[0][i]),
                'label': int(self.reference_labels[idx])
            })
        return neighbors
