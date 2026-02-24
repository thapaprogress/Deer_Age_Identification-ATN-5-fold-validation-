"""
Augmented Triplet Network (ATN) Loss Functions
Implements both standard Triplet Loss and Augmented Triplet Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


class TripletLoss(nn.Module):
    """Standard Triplet Loss"""
    
    def __init__(self, margin=0.2, distance_metric='cosine'):
        """
        Args:
            margin: Margin for triplet loss
            distance_metric: 'cosine' or 'euclidean'
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.distance_metric = distance_metric
    
    def compute_distance(self, x1, x2):
        """Compute distance between two embeddings"""
        if self.distance_metric == 'cosine':
            # Cosine distance = 1 - cosine similarity
            return 1 - F.cosine_similarity(x1, x2)
        elif self.distance_metric == 'euclidean':
            return F.pairwise_distance(x1, x2)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def forward(self, anchor, positive, negative):
        """
        Compute triplet loss
        
        Args:
            anchor: Anchor embeddings (batch_size, embedding_dim)
            positive: Positive embeddings (batch_size, embedding_dim)
            negative: Negative embeddings (batch_size, embedding_dim)
        
        Returns:
            loss: Triplet loss value
            stats: Dictionary with loss statistics
        """
        # Compute distances
        pos_dist = self.compute_distance(anchor, positive)
        neg_dist = self.compute_distance(anchor, negative)
        
        # Triplet loss: max(0, d(a,p) - d(a,n) + margin)
        losses = F.relu(pos_dist - neg_dist + self.margin)
        
        # Statistics
        num_triplets = losses.size(0)
        num_hard_triplets = torch.sum(losses > 0).item()
        
        stats = {
            'loss': losses.mean().item(),
            'pos_dist': pos_dist.mean().item(),
            'neg_dist': neg_dist.mean().item(),
            'num_triplets': num_triplets,
            'num_hard_triplets': num_hard_triplets,
            'hard_triplet_ratio': num_hard_triplets / num_triplets if num_triplets > 0 else 0
        }
        
        return losses.mean(), stats


class AugmentedTripletLoss(nn.Module):
    """
    Augmented Triplet Loss (ATN)
    Uses dummy anchors (class centroids) to separate close classes
    """
    
    def __init__(self, alpha=0.1, beta=0.3, distance_metric='cosine'):
        """
        Args:
            alpha: Maximum distance for same-class elements to dummy anchor
            beta: Minimum distance for different-class elements to dummy anchor
            distance_metric: 'cosine' or 'euclidean'
        """
        super(AugmentedTripletLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.distance_metric = distance_metric
    
    def compute_distance(self, x1, x2):
        """Compute distance between two embeddings"""
        if self.distance_metric == 'cosine':
            return 1 - F.cosine_similarity(x1, x2)
        elif self.distance_metric == 'euclidean':
            return F.pairwise_distance(x1, x2)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def compute_class_centroids(self, embeddings, labels):
        """
        Compute class centroids (dummy anchors)
        
        Args:
            embeddings: Embeddings (batch_size, embedding_dim)
            labels: Labels (batch_size,)
        
        Returns:
            centroids: Dictionary mapping label to centroid
        """
        centroids = {}
        unique_labels = torch.unique(labels)
        
        for label in unique_labels:
            mask = labels == label
            class_embeddings = embeddings[mask]
            centroid = class_embeddings.mean(dim=0)
            centroids[label.item()] = centroid
        
        return centroids
    
    def find_close_class_pairs(self, centroids):
        """
        Find pairs of classes that are close in embedding space
        
        Args:
            centroids: Dictionary mapping label to centroid
        
        Returns:
            close_pairs: List of (label_i, label_j) tuples where d(centroid_i, centroid_j) <= beta
        """
        close_pairs = []
        labels = list(centroids.keys())
        
        for i, label_i in enumerate(labels):
            for j, label_j in enumerate(labels):
                if i >= j:
                    continue
                
                centroid_i = centroids[label_i]
                centroid_j = centroids[label_j]
                
                # Compute distance between centroids
                dist = self.compute_distance(
                    centroid_i.unsqueeze(0),
                    centroid_j.unsqueeze(0)
                )
                
                if dist <= self.beta:
                    close_pairs.append((label_i, label_j))
        
        return close_pairs
    
    def forward(self, embeddings, labels):
        """
        Compute Augmented Triplet Loss
        
        Args:
            embeddings: Embeddings (batch_size, embedding_dim)
            labels: Labels (batch_size,)
        
        Returns:
            loss: ATN loss value
            stats: Dictionary with loss statistics
        """
        # Compute class centroids (dummy anchors)
        centroids = self.compute_class_centroids(embeddings, labels)
        
        # Find close class pairs
        close_pairs = self.find_close_class_pairs(centroids)
        
        if len(close_pairs) == 0:
            # No close pairs found, return zero loss but keep it attached to graph
            return embeddings.sum() * 0.0, {
                'loss': 0.0,
                'num_close_pairs': 0,
                'intra_class_loss': 0.0,
                'inter_class_loss': 0.0
            }
        
        # Compute ATN loss for close class pairs
        intra_class_losses = []
        inter_class_losses = []
        
        for label_i, label_j in close_pairs:
            centroid_i = centroids[label_i]
            centroid_j = centroids[label_j]
            
            # Get embeddings for class i and class j
            mask_i = labels == label_i
            mask_j = labels == label_j
            
            embeddings_i = embeddings[mask_i]
            embeddings_j = embeddings[mask_j]
            
            # Intra-class constraint: d(centroid_i, embedding_i) <= alpha
            # Loss: max(0, d(centroid_i, embedding_i) - alpha)
            for emb in embeddings_i:
                dist = self.compute_distance(
                    centroid_i.unsqueeze(0),
                    emb.unsqueeze(0)
                )
                intra_loss = F.relu(dist - self.alpha)
                intra_class_losses.append(intra_loss)
            
            # Inter-class constraint: d(centroid_i, embedding_j) >= beta
            # Loss: max(0, beta - d(centroid_i, embedding_j))
            for emb in embeddings_j:
                dist = self.compute_distance(
                    centroid_i.unsqueeze(0),
                    emb.unsqueeze(0)
                )
                inter_loss = F.relu(self.beta - dist)
                inter_class_losses.append(inter_loss)
            
            # Also apply constraints for centroid_j
            for emb in embeddings_j:
                dist = self.compute_distance(
                    centroid_j.unsqueeze(0),
                    emb.unsqueeze(0)
                )
                intra_loss = F.relu(dist - self.alpha)
                intra_class_losses.append(intra_loss)
            
            for emb in embeddings_i:
                dist = self.compute_distance(
                    centroid_j.unsqueeze(0),
                    emb.unsqueeze(0)
                )
                inter_loss = F.relu(self.beta - dist)
                inter_class_losses.append(inter_loss)
        
        # Compute total loss
        if len(intra_class_losses) > 0:
            intra_class_loss = torch.stack(intra_class_losses).mean()
        else:
            intra_class_loss = torch.tensor(0.0, device=embeddings.device)
        
        if len(inter_class_losses) > 0:
            inter_class_loss = torch.stack(inter_class_losses).mean()
        else:
            inter_class_loss = torch.tensor(0.0, device=embeddings.device)
        
        total_loss = intra_class_loss + inter_class_loss
        
        stats = {
            'loss': total_loss.item(),
            'num_close_pairs': len(close_pairs),
            'intra_class_loss': intra_class_loss.item(),
            'inter_class_loss': inter_class_loss.item(),
            'close_pairs': close_pairs
        }
        
        return total_loss, stats


class CombinedLoss(nn.Module):
    """
    Combined loss: Triplet Loss + Augmented Triplet Loss
    Can switch between phases or use both simultaneously
    """
    
    def __init__(self, triplet_margin=0.2, atn_alpha=0.1, atn_beta=0.3,
                 distance_metric='cosine', phase='triplet', atn_weight=1.0):
        """
        Args:
            triplet_margin: Margin for triplet loss
            atn_alpha: Alpha for ATN loss
            atn_beta: Beta for ATN loss
            distance_metric: Distance metric
            phase: 'triplet', 'atn', or 'combined'
            atn_weight: Weight for ATN loss when using combined
        """
        super(CombinedLoss, self).__init__()
        
        self.triplet_loss = TripletLoss(margin=triplet_margin, distance_metric=distance_metric)
        self.atn_loss = AugmentedTripletLoss(alpha=atn_alpha, beta=atn_beta, distance_metric=distance_metric)
        self.phase = phase
        self.atn_weight = atn_weight
    
    def set_phase(self, phase):
        """Set training phase: 'triplet', 'atn', or 'combined'"""
        self.phase = phase
    
    def forward(self, anchor=None, positive=None, negative=None, 
                embeddings=None, labels=None):
        """
        Compute loss based on current phase
        
        For triplet phase: provide anchor, positive, negative
        For ATN phase: provide embeddings, labels
        For combined: provide all
        """
        stats = {}
        
        if self.phase == 'triplet':
            loss, triplet_stats = self.triplet_loss(anchor, positive, negative)
            stats.update({'triplet_' + k: v for k, v in triplet_stats.items()})
            return loss, stats
        
        elif self.phase == 'atn':
            loss, atn_stats = self.atn_loss(embeddings, labels)
            stats.update({'atn_' + k: v for k, v in atn_stats.items()})
            return loss, stats
        
        elif self.phase == 'combined':
            triplet_loss_val, triplet_stats = self.triplet_loss(anchor, positive, negative)
            atn_loss_val, atn_stats = self.atn_loss(embeddings, labels)
            
            total_loss = triplet_loss_val + self.atn_weight * atn_loss_val
            
            stats.update({'triplet_' + k: v for k, v in triplet_stats.items()})
            stats.update({'atn_' + k: v for k, v in atn_stats.items()})
            stats['total_loss'] = total_loss.item()
            
            return total_loss, stats
        
        else:
            raise ValueError(f"Unknown phase: {self.phase}")


if __name__ == "__main__":
    # Test loss functions
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Testing Triplet Loss...")
    triplet_loss = TripletLoss(margin=0.2, distance_metric='cosine')
    
    anchor = torch.randn(8, 128).to(device)
    positive = torch.randn(8, 128).to(device)
    negative = torch.randn(8, 128).to(device)
    
    # Normalize embeddings
    anchor = F.normalize(anchor, p=2, dim=1)
    positive = F.normalize(positive, p=2, dim=1)
    negative = F.normalize(negative, p=2, dim=1)
    
    loss, stats = triplet_loss(anchor, positive, negative)
    print(f"Loss: {loss.item():.4f}")
    print(f"Stats: {stats}")
    
    print("\n" + "="*60)
    print("Testing Augmented Triplet Loss...")
    atn_loss = AugmentedTripletLoss(alpha=0.1, beta=0.3, distance_metric='cosine')
    
    embeddings = torch.randn(16, 128).to(device)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]).to(device)
    
    loss, stats = atn_loss(embeddings, labels)
    print(f"Loss: {loss.item():.4f}")
    print(f"Stats: {stats}")
