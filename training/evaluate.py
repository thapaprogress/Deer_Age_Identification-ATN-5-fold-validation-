"""
Evaluation Script for ATN Deer Age Recognition
Generates metrics, confusion matrix, and t-SNE visualizations
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.feature_extractor import create_feature_extractor
from torchvision import transforms
from utils.data_loader import create_data_loaders
from training import config


def extract_embeddings(model, data_loader, device='cuda'):
    """
    Extract embeddings and labels from data loader
    
    Args:
        model: Feature extractor model
        data_loader: Data loader
        device: Device
    
    Returns:
        embeddings: numpy array of embeddings
        labels: numpy array of labels
    """
    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Extracting embeddings"):
            images = images.to(device)
            
            # Get embeddings
            embeddings = model(images)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())
    
    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    return embeddings, labels


from utils.visualization import (
    plot_confusion_matrix, 
    plot_tsne, 
    plot_training_curves, 
    plot_per_class_metrics
)

def evaluate_knn_classifier(train_embeddings, train_labels, test_embeddings, test_labels, k=5):
    """
    Evaluate age classification using K-NN classifier on embeddings
    
    Args:
        train_embeddings: Training embeddings
        train_labels: Training labels
        test_embeddings: Test embeddings
        test_labels: Test labels
        k: Number of neighbors
    
    Returns:
        predictions: Predicted labels
        accuracy: Classification accuracy
        report: Classification report
    """
    # Train K-NN classifier
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    knn.fit(train_embeddings, train_labels)
    
    # Predict
    predictions = knn.predict(test_embeddings)
    
    # Metrics
    accuracy = accuracy_score(test_labels, predictions)
    report = classification_report(test_labels, predictions, output_dict=True)
    
    return predictions, accuracy, report


def main():
    """Main evaluation function"""
    print("="*80)
    print("ATN DEER AGE RECOGNITION - EVALUATION")
    print("="*80)
    
    # Load configuration
    config.print_config()
    
    # Create data loaders
    print("\nLoading dataset...")
    data_loaders = create_data_loaders(
        data_dir=config.RAW_DATA_DIR,
        batch_size=config.PHASE1_BATCH_SIZE,
        image_size=config.IMAGE_SIZE,
        num_workers=config.NUM_WORKERS,
        use_weighted_sampling=False,  # No weighted sampling for evaluation
        return_triplets=False,
        random_seed=config.RANDOM_SEED
    )
    
    train_loader = data_loaders['train']
    val_loader = data_loaders['val']
    test_loader = data_loaders['test']
    
    # Create model
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ensemble', action='store_true', help='Evaluate using the 5-fold ensemble')
    args = parser.parse_args()

    if args.ensemble:
        print("\n" + "="*80)
        print(" ENSEMBLE EVALUATION ACTIVE ")
        print("="*80)
        from utils.ensemble_inference import EnsembleInferenceEngine
        engine = EnsembleInferenceEngine(model_dir=config.CHECKPOINT_DIR)
        
        # Helper to extract ensemble embeddings
        def extract_ensemble_embeddings(engine, data_loader):
            model_eval_embeddings = []
            all_labels = []
            for images, labels in tqdm(data_loader, desc="Extracting ensemble embeddings"):
                # We use the engine's internal predict logic to get averaged embeddings
                for i in range(images.size(0)):
                    img_pil = transforms.ToPILImage()(images[i])
                    _, _, emb = engine.predict(img_pil, use_tta=False) # Get raw single-pass embedding
                    model_eval_embeddings.append(emb)
                all_labels.append(labels.numpy())
            return np.array(model_eval_embeddings), np.concatenate(all_labels)

        train_embeddings, train_labels = extract_ensemble_embeddings(engine, train_loader)
        val_embeddings, val_labels = extract_ensemble_embeddings(engine, val_loader)
        test_embeddings, test_labels = extract_ensemble_embeddings(engine, test_loader)
        
    else:
        print("\nCreating single model...")
        model = create_feature_extractor(
            embedding_dim=config.EMBEDDING_DIM,
            backbone=config.BACKBONE,
            pretrained=config.PRETRAINED,
            device=config.DEVICE
        )
        
        # Load best model
        checkpoint_path = config.CHECKPOINT_DIR / 'best_model.pth'
        if checkpoint_path.exists():
            print(f"\nLoading best model from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from epoch {checkpoint['epoch']}")
        else:
            print(f"\nWarning: Best model not found at {checkpoint_path}")
            print("Using untrained model for evaluation")
            
        # Extract embeddings
        print("\nExtracting embeddings...")
        train_embeddings, train_labels = extract_embeddings(model, train_loader, config.DEVICE)
        val_embeddings, val_labels = extract_embeddings(model, val_loader, config.DEVICE)
        test_embeddings, test_labels = extract_embeddings(model, test_loader, config.DEVICE)
    
    print(f"Train embeddings: {train_embeddings.shape}")
    print(f"Val embeddings: {val_embeddings.shape}")
    print(f"Test embeddings: {test_embeddings.shape}")
    
    # Evaluate with K-NN classifier
    print(f"\nEvaluating with K-NN classifier (k={config.KNN_K})...")
    predictions, accuracy, report = evaluate_knn_classifier(
        train_embeddings, train_labels,
        test_embeddings, test_labels,
        k=config.KNN_K
    )
    
    print(f"\nTest Accuracy: {accuracy*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(test_labels, predictions, target_names=[f'Age {age}' for age in config.AGE_CLASSES]))
    
    # Set results directory
    output_dir = config.RESULTS_DIR
    if args.ensemble:
        output_dir = config.RESULTS_DIR / 'ensemble_results'
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Ensemble results will be saved in: {output_dir}")

    # Save results
    results = {
        'accuracy': float(accuracy),
        'classification_report': report,
        'num_train_samples': len(train_labels),
        'num_val_samples': len(val_labels),
        'num_test_samples': len(test_labels),
        'embedding_dim': config.EMBEDDING_DIM,
        'backbone': config.BACKBONE,
        'knn_k': config.KNN_K,
        'is_ensemble': bool(args.ensemble)
    }
    
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_path}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # 1. Confusion Matrix
    plot_confusion_matrix(
        test_labels, predictions, config.AGE_CLASSES,
        output_dir / 'confusion_matrix.png'
    )
    
    # 2. t-SNE Visualization
    plot_tsne(
        test_embeddings, test_labels, config.AGE_CLASSES,
        output_dir / 'tsne_visualization.png',
        perplexity=config.TSNE_PERPLEXITY
    )
    
    # 3. Per-class Metrics
    plot_per_class_metrics(
        report, config.AGE_CLASSES,
        output_dir / 'per_class_metrics.png'
    )
    
    # 4. Training Curves
    history_path = config.RESULTS_DIR / 'training_history.json'
    if history_path.exists():
        plot_training_curves(
            history_path,
            output_dir / 'training_curves.png'
        )
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETED!")
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print(f"Results saved in: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
