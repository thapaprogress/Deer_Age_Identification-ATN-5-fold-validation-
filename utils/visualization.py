"""
Visualization Utilities for Deer Age Recognition
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import numpy as np
import json

def plot_confusion_matrix(y_true, y_pred, age_classes, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred, labels=age_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=age_classes, yticklabels=age_classes)
    plt.title('Confusion Matrix - Deer Age Classification', fontsize=16, fontweight='bold')
    plt.ylabel('True Age', fontsize=12)
    plt.xlabel('Predicted Age', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_tsne(embeddings, labels, age_classes, save_path, perplexity=30):
    """Plot and save t-SNE visualization"""
    # Compute t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=1000, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(12, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, len(age_classes)))
    
    for i, age in enumerate(age_classes):
        mask = labels == age
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                   c=[colors[i]], label=f'Age {age}', alpha=0.6, s=50)
    
    plt.title('t-SNE Visualization of Deer Age Embeddings', fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.legend(title='Age Group', fontsize=10, title_fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_curves(history_path, save_path):
    """Plot training curves from history"""
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    phases = history['phase']
    
    # Find phase transition
    phase_transition = None
    for i, phase in enumerate(phases):
        if i > 0 and phase != phases[i-1]:
            phase_transition = i
            break
    
    plt.figure(figsize=(12, 6))
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    
    if phase_transition:
        plt.axvline(x=phase_transition, color='green', linestyle='--', 
                   linewidth=2, label=f'Phase Transition (Epoch {phase_transition})')
    
    plt.title('Training Curves - ATN Deer Age Recognition', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_per_class_metrics(report, age_classes, save_path):
    """Plot per-class precision, recall, F1-score"""
    metrics = {'Precision': [], 'Recall': [], 'F1-Score': []}
    
    for age in age_classes:
        age_str = str(age)
        if age_str in report:
            metrics['Precision'].append(report[age_str]['precision'])
            metrics['Recall'].append(report[age_str]['recall'])
            metrics['F1-Score'].append(report[age_str]['f1-score'])
        else:
            metrics['Precision'].append(0)
            metrics['Recall'].append(0)
            metrics['F1-Score'].append(0)
    
    x = np.arange(len(age_classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, metrics['Precision'], width, label='Precision', color='skyblue')
    ax.bar(x, metrics['Recall'], width, label='Recall', color='lightcoral')
    ax.bar(x + width, metrics['F1-Score'], width, label='F1-Score', color='lightgreen')
    
    ax.set_xlabel('Age Group', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Metrics - Deer Age Classification', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Age {age}' for age in age_classes])
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
