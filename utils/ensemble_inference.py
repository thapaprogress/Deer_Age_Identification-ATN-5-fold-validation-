"""
Ensemble Inference Engine for ATN Deer Age Recognition
Averages embeddings from multiple models (folds) for higher accuracy.
"""

import torch
import numpy as np
from pathlib import Path
from utils.inference import InferenceEngine
from training import config

class EnsembleInferenceEngine(InferenceEngine):
    """
    Ensemble version of the InferenceEngine.
    Loads 5 models and averages their embeddings before KNN search.
    """
    
    def __init__(self, model_dir=None, num_folds=5, device='cuda'):
        """
        Args:
            model_dir: Directory containing best_model_fold_X.pth checkpoints
            num_folds: Number of folds to ensemble
            device: Device to run on
        """
        self.device = 'cuda' if torch.cuda.is_available() and device=='cuda' else 'cpu'
        self.num_folds = num_folds
        self.models = []
        
        # Super init will try to load ref data, but we want to do it after loading models
        # We pass a flag to tell it to skip if possible, or just let it happen and we'll re-do it
        super().__init__(model_path=None, device=self.device)
        
        # Load all models
        model_dir = Path(model_dir) if model_dir else config.CHECKPOINT_DIR
        for i in range(1, num_folds + 1):
            fold_path = model_dir / f"best_model_fold_{i}.pth"
            if fold_path.exists():
                model = self._load_fold_model(fold_path)
                self.models.append(model)
                print(f"Loaded fold {i} from {fold_path}")
            else:
                print(f"Warning: Fold {i} checkpoint not found at {fold_path}")

        if not self.models:
            print("Error: No fold models loaded! Ensemble will fail.")
        else:
            # Re-initialize reference data using the ENSEMBLE instead of self.model (random)
            self._initialize_reference_data()

    def _initialize_reference_data(self):
        """Override to use the ensemble for extracting reference embeddings"""
        if not hasattr(self, 'models') or not self.models:
            # If models aren't loaded yet (during super init), do nothing
            return
            
        print("Initializing ENSEMBLE reference data...")
        try:
            data_loaders = create_data_loaders(
                data_dir=config.RAW_DATA_DIR,
                batch_size=32,
                image_size=config.IMAGE_SIZE,
                num_workers=0,
                return_triplets=False
            )
            train_loader = data_loaders['train']
            
            embeddings = []
            labels = []
            
            with torch.no_grad():
                for images, batch_labels in train_loader:
                    images = images.to(self.device)
                    # Use the ensemble predict logic for the batch
                    # Note: We don't use TTA for reference data to keep it clean/aligned with paper
                    
                    batch_fold_embs = []
                    for model in self.models:
                        emb = model(images)
                        batch_fold_embs.append(emb)
                    
                    # Average across folds
                    ensemble_emb = torch.mean(torch.stack(batch_fold_embs), dim=0)
                    # Normalize
                    ensemble_emb = torch.nn.functional.normalize(ensemble_emb, p=2, dim=-1)
                    
                    embeddings.append(ensemble_emb.cpu().numpy())
                    labels.append(batch_labels.numpy())
            
            if embeddings:
                self.reference_embeddings = np.concatenate(embeddings)
                self.reference_labels = np.concatenate(labels)
                
                # Fit KNN
                self.knn = KNeighborsClassifier(n_neighbors=config.KNN_K, metric='cosine')
                self.knn.fit(self.reference_embeddings, self.reference_labels)
                print(f"Ensemble reference data initialized: {len(self.reference_embeddings)} samples")
        except Exception as e:
            print(f"Warning: Ensemble reference data initialization failed: {e}")

    def _load_fold_model(self, model_path):
        """Re-create and load a specific fold model"""
        from models.feature_extractor import create_feature_extractor
        model = create_feature_extractor(
            embedding_dim=config.EMBEDDING_DIM,
            backbone=config.BACKBONE,
            pretrained=False,
            device=self.device
        )
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def predict(self, image, use_tta=True):
        """
        Predict age by averaging embeddings from all ensemble members.
        """
        if self.knn is None or not self.models:
            return super().predict(image, use_tta=use_tta)

        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # TTA support for ensemble
        if use_tta:
            flipped_img = torch.flip(img_tensor, [3]) # Fast horizontal flip on tensor
            inputs = [img_tensor, flipped_img]
        else:
            inputs = [img_tensor]

        ensemble_embeddings = []
        
        with torch.no_grad():
            for model in self.models:
                # Average TTA results for this model first
                model_embeddings = []
                for x in inputs:
                    emb = model(x)
                    model_embeddings.append(emb)
                
                model_avg = torch.mean(torch.stack(model_embeddings), dim=0)
                ensemble_embeddings.append(model_avg)
                
            # Average across all models
            final_embedding = torch.mean(torch.stack(ensemble_embeddings), dim=0)
            # Normalize to unit sphere (normalize across the embedding dimension)
            final_embedding = torch.nn.functional.normalize(final_embedding, p=2, dim=-1).squeeze(0)
            final_embedding = final_embedding.cpu().numpy()
            
        # Predict using the global KNN
        # Reshape to 2D array (1 sample) for sklearn
        final_embedding_2d = final_embedding.reshape(1, -1)
        predicted_age = self.knn.predict(final_embedding_2d)[0]
        probs = self.knn.predict_proba(final_embedding_2d)[0]
        confidence = np.max(probs)
        
        return int(predicted_age), float(confidence), final_embedding
