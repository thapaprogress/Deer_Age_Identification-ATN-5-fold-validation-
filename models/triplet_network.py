"""
Triplet Network Wrapper
Wraps the feature extractor to facilitate triplet training (anchor, positive, negative)
"""

import torch
import torch.nn as nn
from models.feature_extractor import FeatureExtractor

class TripletNetwork(nn.Module):
    """
    Siamese Network for Triplet Loss Training
    Passes (anchor, positive, negative) through the same embedding network
    """
    def __init__(self, embedding_net):
        """
        Args:
            embedding_net: The feature extractor network (CNN)
        """
        super(TripletNetwork, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2=None, x3=None):
        """
        Forward pass.
        If just x1 provided, returns embeddings for x1.
        If x1, x2, x3 provided, returns tuple of embeddings (emb1, emb2, emb3).
        """
        if x2 is None and x3 is None:
            return self.embedding_net(x1)
        
        emb1 = self.embedding_net(x1)
        emb2 = self.embedding_net(x2)
        emb3 = self.embedding_net(x3)
        
        return emb1, emb2, emb3

    def get_embedding(self, x):
        """Helper to get embedding for single input"""
        return self.embedding_net(x)
