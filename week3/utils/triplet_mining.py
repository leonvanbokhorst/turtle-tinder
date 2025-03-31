"""
Triplet mining strategies for deep metric learning.

This module provides various strategies for mining informative triplets
to improve the training efficiency of triplet networks.
"""

import torch
import numpy as np
from typing import Tuple, List, Dict, Optional, Union, Any, Callable


def get_triplet_mask(labels: torch.Tensor) -> torch.Tensor:
    """
    Return a 3D mask where mask[a, p, n] is True if the triplet (a, p, n) is valid.
    A triplet is valid if:
    - a, p, n are distinct indices
    - a and p have the same label
    - a and n have different labels

    Args:
        labels: Tensor of shape [batch_size]

    Returns:
        Mask of shape [batch_size, batch_size, batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
    indices_not_equal = ~indices_equal
    i_not_equal_j = indices_not_equal.unsqueeze(2)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    j_not_equal_k = indices_not_equal.unsqueeze(0)

    distinct_indices = i_not_equal_j & i_not_equal_k & j_not_equal_k

    # Check if a and p have the same label and a and n have different labels
    label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    i_equal_j = label_equal.unsqueeze(2)
    i_not_equal_k = ~label_equal.unsqueeze(1)

    valid_labels = i_equal_j & i_not_equal_k

    # Combine the two masks
    mask = distinct_indices & valid_labels

    return mask


def batch_hard_triplet_mining(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.3,
    squared: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    For each anchor, mine the hardest positive and negative samples within the batch.

    Args:
        embeddings: Tensor of shape [batch_size, embedding_dim]
        labels: Tensor of shape [batch_size]
        margin: Minimum desired distance between positive and negative samples
        squared: Whether to use squared distances

    Returns:
        Tuple of (loss, num_hard_triplets, anchor_idx, positive_idx, negative_idx)
    """
    # Get the device of the tensors
    device = embeddings.device

    # Calculate pairwise distances
    if squared:
        # Calculate squared Euclidean distance matrix
        distances = torch.cdist(embeddings, embeddings, p=2) ** 2
    else:
        # Calculate Euclidean distance matrix
        distances = torch.cdist(embeddings, embeddings, p=2)

    # For each anchor, find the hardest positive and hardest negative
    batch_size = len(labels)

    # Initialize indices for hardest positive and negative
    anchor_idx = []
    positive_idx = []
    negative_idx = []

    for i in range(batch_size):
        anchor_label = labels[i]

        # Find positives (same label as anchor)
        pos_indices = torch.where(labels == anchor_label)[0]
        # Remove the anchor itself
        pos_indices = pos_indices[pos_indices != i]

        if len(pos_indices) == 0:
            continue  # Skip if no positives

        # Find negatives (different label from anchor)
        neg_indices = torch.where(labels != anchor_label)[0]

        if len(neg_indices) == 0:
            continue  # Skip if no negatives

        # Get distances from anchor to positives and negatives
        pos_distances = distances[i, pos_indices]
        neg_distances = distances[i, neg_indices]

        # Find hardest positive (furthest positive)
        hardest_pos_idx = pos_indices[torch.argmax(pos_distances)]

        # Find hardest negative (closest negative)
        hardest_neg_idx = neg_indices[torch.argmin(neg_distances)]

        # Add indices to lists
        anchor_idx.append(i)
        positive_idx.append(hardest_pos_idx.item())
        negative_idx.append(hardest_neg_idx.item())

    # Convert lists to tensors
    anchor_idx = torch.tensor(anchor_idx, device=device)
    positive_idx = torch.tensor(positive_idx, device=device)
    negative_idx = torch.tensor(negative_idx, device=device)

    # Get embeddings for the selected triplets
    if len(anchor_idx) > 0:
        anchors = embeddings[anchor_idx]
        positives = embeddings[positive_idx]
        negatives = embeddings[negative_idx]

        # Compute triplet loss
        pos_dist = torch.pairwise_distance(anchors, positives)
        neg_dist = torch.pairwise_distance(anchors, negatives)

        loss = torch.clamp(pos_dist - neg_dist + margin, min=0.0)
        num_hard_triplets = torch.sum(loss > 1e-16).item()

        return (
            torch.mean(loss),
            num_hard_triplets,
            anchor_idx,
            positive_idx,
            negative_idx,
        )
    else:
        # No valid triplets found
        return (
            torch.tensor(0.0, device=device),
            0,
            anchor_idx,
            positive_idx,
            negative_idx,
        )


def batch_all_triplet_mining(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.3,
    squared: bool = False,
) -> Tuple[torch.Tensor, int, float]:
    """
    Mine all valid triplets in the batch and calculate the average loss over
    the positive ones (those with loss > 0).

    Args:
        embeddings: Tensor of shape [batch_size, embedding_dim]
        labels: Tensor of shape [batch_size]
        margin: Minimum desired distance between positive and negative samples
        squared: Whether to use squared distances

    Returns:
        Tuple of (loss, num_triplets, fraction_positive_triplets)
    """
    # Get the device of the tensors
    device = embeddings.device

    # Calculate pairwise distances
    if squared:
        # Calculate squared Euclidean distance matrix
        distances = torch.cdist(embeddings, embeddings, p=2) ** 2
    else:
        # Calculate Euclidean distance matrix
        distances = torch.cdist(embeddings, embeddings, p=2)

    # Get a 3D mask of valid triplets
    mask = get_triplet_mask(labels)

    # Calculate triplet loss for all valid triplets
    # First, compute distance difference for all anchor-positive and anchor-negative pairs
    anchor_positive_dist = distances.unsqueeze(2)  # [batch_size, batch_size, 1]
    anchor_negative_dist = distances.unsqueeze(1)  # [batch_size, 1, batch_size]

    # Compute triplet loss: max(ap_dist - an_dist + margin, 0)
    loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets (based on the mask)
    loss = loss * mask.float()

    # Remove negative losses (easy triplets)
    loss = torch.clamp(loss, min=0.0)

    # Count the number of positive triplets (where loss > 0)
    positive_triplets = (loss > 1e-16).float()
    num_positive_triplets = positive_triplets.sum().item()

    # Count the number of valid triplets
    num_valid_triplets = mask.sum().item()

    # Calculate fraction of positive triplets
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    # Get the mean loss over positive triplets
    loss = loss.sum() / (num_positive_triplets + 1e-16)

    return loss, num_positive_triplets, fraction_positive_triplets


def semi_hard_triplet_mining(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.3,
    squared: bool = False,
) -> Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Mine semi-hard triplets, where the negative is not closer to the anchor than the positive,
    but still within the margin.

    Args:
        embeddings: Tensor of shape [batch_size, embedding_dim]
        labels: Tensor of shape [batch_size]
        margin: Minimum desired distance between positive and negative samples
        squared: Whether to use squared distances

    Returns:
        Tuple of (loss, num_valid_triplets, anchor_idx, positive_idx, negative_idx)
    """
    # Get the device of the tensors
    device = embeddings.device

    # Calculate pairwise distances
    if squared:
        # Calculate squared Euclidean distance matrix
        distances = torch.cdist(embeddings, embeddings, p=2) ** 2
    else:
        # Calculate Euclidean distance matrix
        distances = torch.cdist(embeddings, embeddings, p=2)

    # For each anchor, find semi-hard triplets
    batch_size = len(labels)

    # Initialize indices for semi-hard triplets
    anchor_idx = []
    positive_idx = []
    negative_idx = []

    for i in range(batch_size):
        anchor_label = labels[i]

        # Find positives (same label as anchor)
        pos_indices = torch.where(labels == anchor_label)[0]
        # Remove the anchor itself
        pos_indices = pos_indices[pos_indices != i]

        if len(pos_indices) == 0:
            continue  # Skip if no positives

        # Find negatives (different label from anchor)
        neg_indices = torch.where(labels != anchor_label)[0]

        if len(neg_indices) == 0:
            continue  # Skip if no negatives

        # Get distances from anchor to positives and negatives
        pos_distances = distances[i, pos_indices]
        neg_distances = distances[i, neg_indices]

        # Find a random positive
        random_pos_idx = pos_indices[torch.randint(0, len(pos_indices), (1,)).item()]

        # Find the distance to this positive
        pos_distance = distances[i, random_pos_idx]

        # Find semi-hard negatives: further than positive but within margin
        semi_hard_negs = neg_indices[
            (neg_distances > pos_distance) & (neg_distances < pos_distance + margin)
        ]

        if len(semi_hard_negs) > 0:
            # Choose a random semi-hard negative
            random_idx = torch.randint(0, len(semi_hard_negs), (1,)).item()
            hard_neg_idx = semi_hard_negs[random_idx]
        else:
            # If no semi-hard negative, choose the closest negative
            hard_neg_idx = neg_indices[torch.argmin(neg_distances)]

        # Add indices to lists
        anchor_idx.append(i)
        positive_idx.append(random_pos_idx.item())
        negative_idx.append(hard_neg_idx.item())

    # Convert lists to tensors
    anchor_idx = torch.tensor(anchor_idx, device=device)
    positive_idx = torch.tensor(positive_idx, device=device)
    negative_idx = torch.tensor(negative_idx, device=device)

    # Get embeddings for the selected triplets
    if len(anchor_idx) > 0:
        anchors = embeddings[anchor_idx]
        positives = embeddings[positive_idx]
        negatives = embeddings[negative_idx]

        # Compute triplet loss
        pos_dist = torch.pairwise_distance(anchors, positives)
        neg_dist = torch.pairwise_distance(anchors, negatives)

        loss = torch.clamp(pos_dist - neg_dist + margin, min=0.0)
        num_valid_triplets = len(anchor_idx)

        return (
            torch.mean(loss),
            num_valid_triplets,
            anchor_idx,
            positive_idx,
            negative_idx,
        )
    else:
        # No valid triplets found
        return (
            torch.tensor(0.0, device=device),
            0,
            anchor_idx,
            positive_idx,
            negative_idx,
        )


def distance_weighted_triplet_mining(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.3,
    cutoff: float = 0.5,
    nonzero_loss_cutoff: float = 1.4,
    squared: bool = False,
) -> Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Mine triplets with distance weighted sampling, which samples negatives
    according to their distance to the anchor (Wu et al., Sampling Matters in Deep Embedding Learning).

    Args:
        embeddings: Tensor of shape [batch_size, embedding_dim]
        labels: Tensor of shape [batch_size]
        margin: Minimum desired distance between positive and negative samples
        cutoff: Distance cutoff for sampling
        nonzero_loss_cutoff: Cutoff for non-zero loss
        squared: Whether to use squared distances

    Returns:
        Tuple of (loss, num_valid_triplets, anchor_idx, positive_idx, negative_idx)
    """
    # Get the device of the tensors
    device = embeddings.device

    # Calculate pairwise distances
    if squared:
        # Calculate squared Euclidean distance matrix
        distances = torch.cdist(embeddings, embeddings, p=2) ** 2
    else:
        # Calculate Euclidean distance matrix
        distances = torch.cdist(embeddings, embeddings, p=2)

    # For each anchor, find distance weighted triplets
    batch_size = len(labels)

    # Initialize indices for triplets
    anchor_idx = []
    positive_idx = []
    negative_idx = []

    for i in range(batch_size):
        anchor_label = labels[i]

        # Find positives (same label as anchor)
        pos_indices = torch.where(labels == anchor_label)[0]
        # Remove the anchor itself
        pos_indices = pos_indices[pos_indices != i]

        if len(pos_indices) == 0:
            continue  # Skip if no positives

        # Find negatives (different label from anchor)
        neg_indices = torch.where(labels != anchor_label)[0]

        if len(neg_indices) == 0:
            continue  # Skip if no negatives

        # Choose a random positive
        random_pos_idx = pos_indices[torch.randint(0, len(pos_indices), (1,)).item()]

        # Get distances from anchor to negatives
        neg_distances = distances[i, neg_indices]

        # Apply distance weighting as in the paper
        # p(d) ~ exp(-d) / (1 + exp(-d))
        neg_weights = torch.exp(-neg_distances)
        neg_weights = neg_weights / (1.0 + neg_weights)

        # Apply cutoffs to avoid sampling too close or too far negatives
        neg_weights = neg_weights * (neg_distances < cutoff).float()

        if neg_weights.sum() > 0:
            # Normalize weights
            neg_weights = neg_weights / neg_weights.sum()

            # Sample a negative based on the weights
            neg_idx = torch.multinomial(neg_weights, 1).item()
            hard_neg_idx = neg_indices[neg_idx]
        else:
            # If no valid negative with weight > 0, choose a random one
            hard_neg_idx = neg_indices[torch.randint(0, len(neg_indices), (1,)).item()]

        # Add indices to lists
        anchor_idx.append(i)
        positive_idx.append(random_pos_idx.item())
        negative_idx.append(hard_neg_idx.item())

    # Convert lists to tensors
    anchor_idx = torch.tensor(anchor_idx, device=device)
    positive_idx = torch.tensor(positive_idx, device=device)
    negative_idx = torch.tensor(negative_idx, device=device)

    # Get embeddings for the selected triplets
    if len(anchor_idx) > 0:
        anchors = embeddings[anchor_idx]
        positives = embeddings[positive_idx]
        negatives = embeddings[negative_idx]

        # Compute triplet loss
        pos_dist = torch.pairwise_distance(anchors, positives)
        neg_dist = torch.pairwise_distance(anchors, negatives)

        loss = torch.clamp(pos_dist - neg_dist + margin, min=0.0)
        num_valid_triplets = len(anchor_idx)

        return (
            torch.mean(loss),
            num_valid_triplets,
            anchor_idx,
            positive_idx,
            negative_idx,
        )
    else:
        # No valid triplets found
        return (
            torch.tensor(0.0, device=device),
            0,
            anchor_idx,
            positive_idx,
            negative_idx,
        )


def get_triplet_mining_fn(
    method: str, margin: float = 0.3, squared: bool = False, **kwargs: Any
) -> Callable:
    """
    Get triplet mining function based on the method name.

    Args:
        method: Triplet mining method, one of ['batch_hard', 'batch_all', 'semi_hard', 'distance_weighted']
        margin: Margin for triplet loss
        squared: Whether to use squared distances
        **kwargs: Additional arguments for specific mining methods

    Returns:
        Triplet mining function
    """
    if method == "batch_hard":
        return lambda embeddings, labels: batch_hard_triplet_mining(
            embeddings, labels, margin=margin, squared=squared
        )
    elif method == "batch_all":
        return lambda embeddings, labels: batch_all_triplet_mining(
            embeddings, labels, margin=margin, squared=squared
        )
    elif method == "semi_hard":
        return lambda embeddings, labels: semi_hard_triplet_mining(
            embeddings, labels, margin=margin, squared=squared
        )
    elif method == "distance_weighted":
        cutoff = kwargs.get("cutoff", 0.5)
        nonzero_loss_cutoff = kwargs.get("nonzero_loss_cutoff", 1.4)
        return lambda embeddings, labels: distance_weighted_triplet_mining(
            embeddings,
            labels,
            margin=margin,
            cutoff=cutoff,
            nonzero_loss_cutoff=nonzero_loss_cutoff,
            squared=squared,
        )
    else:
        raise ValueError(f"Unknown triplet mining method: {method}")


if __name__ == "__main__":
    # Test the triplet mining implementations

    # Create random embeddings
    batch_size = 32
    embedding_dim = 128
    num_classes = 8

    # Create embeddings with each class forming a cluster
    embeddings = torch.randn(batch_size, embedding_dim)

    # Create labels: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, ...]
    labels = torch.tensor([i // (batch_size // num_classes) for i in range(batch_size)])

    print(f"Created embeddings of shape {embeddings.shape} with {num_classes} classes")

    # Test batch hard mining
    loss, num_triplets, anchor_idx, pos_idx, neg_idx = batch_hard_triplet_mining(
        embeddings, labels, margin=0.3
    )
    print(f"\nBatch Hard Mining:")
    print(f"Loss: {loss.item():.4f}")
    print(f"Number of hard triplets: {num_triplets}")

    # Test batch all mining
    loss, num_triplets, fraction = batch_all_triplet_mining(
        embeddings, labels, margin=0.3
    )
    print(f"\nBatch All Mining:")
    print(f"Loss: {loss.item():.4f}")
    print(f"Number of positive triplets: {num_triplets}")
    print(f"Fraction of positive triplets: {fraction:.4f}")

    # Test semi-hard mining
    loss, num_triplets, anchor_idx, pos_idx, neg_idx = semi_hard_triplet_mining(
        embeddings, labels, margin=0.3
    )
    print(f"\nSemi-Hard Mining:")
    print(f"Loss: {loss.item():.4f}")
    print(f"Number of valid triplets: {num_triplets}")

    # Test distance weighted mining
    loss, num_triplets, anchor_idx, pos_idx, neg_idx = distance_weighted_triplet_mining(
        embeddings, labels, margin=0.3
    )
    print(f"\nDistance Weighted Mining:")
    print(f"Loss: {loss.item():.4f}")
    print(f"Number of valid triplets: {num_triplets}")
