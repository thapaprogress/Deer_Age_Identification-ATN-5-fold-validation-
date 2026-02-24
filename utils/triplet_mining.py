"""
Triplet Mining Strategies for Metric Learning
"""

import torch
import torch.nn.functional as F

def get_triplet_mask(labels):
    """
    Return a 3D mask where mask[a, p, n] is True iff (a, p, n) is a valid triplet.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    """
    # Check that i, j and k are distinct
    indices_equal = torch.eye(labels.size(0)).byte().to(labels.device)
    indices_not_equal = ~indices_equal
    i_not_equal_j = indices_not_equal.unsqueeze(2)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    j_not_equal_k = indices_not_equal.unsqueeze(0)

    distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k

    label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    i_equal_j = label_equal.unsqueeze(2)
    i_equal_k = label_equal.unsqueeze(1)

    valid_labels = i_equal_j & (~i_equal_k)

    return valid_labels & distinct_indices

def batch_all_triplet_loss(labels, embeddings, margin, squared=False):
    """
    Compute the triplet loss using all valid triplets
    """
    pairwise_dist = torch.cdist(embeddings, embeddings, p=2)
    
    anchor_positive_dist = pairwise_dist.unsqueeze(2)
    anchor_negative_dist = pairwise_dist.unsqueeze(1)

    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    mask = get_triplet_mask(labels)
    triplet_loss = mask.float() * triplet_loss

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = F.relu(triplet_loss)

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = triplet_loss[triplet_loss > 1e-16]
    num_positive_triplets = valid_triplets.size(0)
    num_valid_triplets = mask.sum()

    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets.float() + 1e-16)

    # Get final mean triplet loss associated with positive triplets
    triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)

    return triplet_loss, fraction_positive_triplets

def batch_hard_triplet_loss(labels, embeddings, margin, squared=False):
    """
    Compute the triplet loss using hard triplets
    """
    pairwise_dist = torch.cdist(embeddings, embeddings, p=2)

    # For each anchor, get the hardest positive
    mask_anchor_positive = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    mask_anchor_positive = mask_anchor_positive - torch.eye(labels.size(0)).to(labels.device) # exclude self
    
    # We put 0 where it's a negative, so max will get positive
    anchor_positive_dist = mask_anchor_positive * pairwise_dist
    hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)

    # For each anchor, get the hardest negative
    mask_anchor_negative = (labels.unsqueeze(0) != labels.unsqueeze(1)).float()
    
    # We add max_dist to positives so min will ignore them
    max_dist = pairwise_dist.max()
    anchor_negative_dist = pairwise_dist + max_dist * (1.0 - mask_anchor_negative)
    hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)

    # Combine largest d(a, p) and smallest d(a, n) into final loss
    triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + margin)
    triplet_loss = triplet_loss.mean()

    return triplet_loss
