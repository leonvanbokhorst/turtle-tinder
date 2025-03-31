"""
Metrics utility for evaluating embeddings for re-identification tasks.

This module provides functions to calculate various metrics for evaluating
the quality of embeddings, including precision@k, mAP, CMC curve, and more.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Union, Optional
from sklearn.metrics import precision_recall_curve, average_precision_score
from scipy.spatial.distance import cdist


def euclidean_distance(x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
    """
    Compute euclidean distance between x and y.

    Args:
        x: First tensor of shape (n, d) where n is number of samples and d is embedding dim
        y: Second tensor of shape (m, d). If None, use x as y

    Returns:
        Distance matrix of shape (n, m)
    """
    if y is None:
        y = x

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    if d != y.size(1):
        raise ValueError("Embedding dimensions mismatch")

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.sqrt(torch.pow(x - y, 2).sum(2) + 1e-12)


def cosine_similarity(x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
    """
    Compute cosine similarity between x and y.

    Args:
        x: First tensor of shape (n, d) where n is number of samples and d is embedding dim
        y: Second tensor of shape (m, d). If None, use x as y

    Returns:
        Similarity matrix of shape (n, m)
    """
    if y is None:
        y = x

    # Normalize along embedding dimension
    x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
    y_norm = torch.nn.functional.normalize(y, p=2, dim=1)

    # Compute similarity
    return torch.mm(x_norm, y_norm.t())


def compute_distance_matrix(
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
    metric: str = "euclidean",
) -> torch.Tensor:
    """
    Compute distance matrix between query and gallery features.

    Args:
        query_features: Query features of shape (n_query, dim)
        gallery_features: Gallery features of shape (n_gallery, dim)
        metric: Distance metric to use ('euclidean' or 'cosine')

    Returns:
        Distance matrix of shape (n_query, n_gallery)
    """
    if metric == "euclidean":
        return euclidean_distance(query_features, gallery_features)
    elif metric == "cosine":
        # Convert cosine similarity to distance (1 - similarity)
        return 1 - cosine_similarity(query_features, gallery_features)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def compute_precision_at_k(
    distance_matrix: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_labels: torch.Tensor,
    k: int = 5,
) -> float:
    """
    Compute precision@k metric.

    Args:
        distance_matrix: Distance matrix of shape (n_query, n_gallery)
        query_labels: Query labels of shape (n_query,)
        gallery_labels: Gallery labels of shape (n_gallery,)
        k: Number of top elements to consider

    Returns:
        precision@k score
    """
    n_query = distance_matrix.size(0)

    # For each query, find top-k nearest neighbors
    _, indices = torch.topk(distance_matrix, k=k, dim=1, largest=False)

    # Convert to binary relevance
    positive_indicator = query_labels.view(-1, 1) == gallery_labels[indices]

    # Compute precision for each query
    precision = torch.sum(positive_indicator, dim=1).float() / k

    # Average over all queries
    return precision.mean().item()


def compute_mean_average_precision(
    distance_matrix: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_labels: torch.Tensor,
) -> float:
    """
    Compute mean average precision (mAP).

    Args:
        distance_matrix: Distance matrix of shape (n_query, n_gallery)
        query_labels: Query labels of shape (n_query,)
        gallery_labels: Gallery labels of shape (n_gallery,)

    Returns:
        mAP score
    """
    n_query = distance_matrix.size(0)
    ap_list = []

    for i in range(n_query):
        # Get distances and labels for this query
        distances = distance_matrix[i]
        query_label = query_labels[i]

        # Sort by distance
        sorted_indices = torch.argsort(distances)
        sorted_labels = gallery_labels[sorted_indices]

        # Find relevant items (same class as query)
        relevant = (sorted_labels == query_label).float()

        if relevant.sum() == 0:
            continue  # Skip queries with no relevant gallery items

        # Compute precision at each position
        cumulative_relevant = torch.cumsum(relevant, dim=0)
        positions = torch.arange(1, len(relevant) + 1, device=relevant.device).float()
        precision_at_i = cumulative_relevant / positions

        # Average over relevant positions
        ap = (precision_at_i * relevant).sum() / relevant.sum()
        ap_list.append(ap.item())

    # Mean average precision
    return np.mean(ap_list) if ap_list else 0.0


def compute_cmc_curve(
    distance_matrix: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_labels: torch.Tensor,
    max_rank: int = 20,
) -> np.ndarray:
    """
    Compute Cumulative Matching Characteristics (CMC) curve.

    Args:
        distance_matrix: Distance matrix of shape (n_query, n_gallery)
        query_labels: Query labels of shape (n_query,)
        gallery_labels: Gallery labels of shape (n_gallery,)
        max_rank: Maximum rank to compute CMC for

    Returns:
        CMC curve up to max_rank
    """
    n_query = distance_matrix.size(0)
    match_at_k = torch.zeros(max_rank)

    for i in range(n_query):
        # Get distances and labels for this query
        distances = distance_matrix[i]
        query_label = query_labels[i]

        # Sort by distance
        sorted_indices = torch.argsort(distances)
        sorted_labels = gallery_labels[sorted_indices]

        # Find positions where the label matches the query
        matches = sorted_labels == query_label

        # If there are no matches, skip this query
        if not matches.any():
            continue

        # Find the first k positions where there's a match
        match_indices = torch.where(matches)[0]

        # Update match_at_k counts
        for k in range(max_rank):
            if (match_indices <= k).any():
                match_at_k[k] += 1

    # Compute CMC curve by dividing by number of queries
    cmc_curve = match_at_k / n_query

    return cmc_curve.cpu().numpy()


def evaluate_embeddings(
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_labels: torch.Tensor,
    distance_metric: str = "euclidean",
    k_values: List[int] = [1, 5, 10],
    compute_cmc: bool = True,
    max_rank: int = 20,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Evaluate embeddings using various metrics.

    Args:
        query_features: Query features of shape (n_query, dim)
        gallery_features: Gallery features of shape (n_gallery, dim)
        query_labels: Query labels of shape (n_query,)
        gallery_labels: Gallery labels of shape (n_gallery,)
        distance_metric: Distance metric to use ('euclidean' or 'cosine')
        k_values: List of k values for precision@k
        compute_cmc: Whether to compute CMC curve
        max_rank: Maximum rank for CMC curve

    Returns:
        Dictionary with metrics
    """
    # Compute distance matrix
    dist_matrix = compute_distance_matrix(
        query_features, gallery_features, distance_metric
    )

    # Initialize results dictionary
    results = {}

    # Compute precision@k for different k values
    for k in k_values:
        results[f"precision@{k}"] = compute_precision_at_k(
            dist_matrix, query_labels, gallery_labels, k
        )

    # Compute mAP
    results["mAP"] = compute_mean_average_precision(
        dist_matrix, query_labels, gallery_labels
    )

    # Compute CMC curve if requested
    if compute_cmc:
        cmc_curve = compute_cmc_curve(
            dist_matrix, query_labels, gallery_labels, max_rank
        )
        results["cmc_curve"] = cmc_curve

        # Extract common rank values from CMC curve
        for rank in [1, 5, 10]:
            if rank <= max_rank:
                results[f"rank-{rank}"] = cmc_curve[rank - 1]

    return results


def compute_normalized_relative_rank(
    distance_matrix: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_labels: torch.Tensor,
) -> float:
    """
    Compute Normalized Relative Rank (NRR) metric.
    NRR measures how well the embedding ranks similar examples closer than dissimilar ones.

    Args:
        distance_matrix: Distance matrix of shape (n_query, n_gallery)
        query_labels: Query labels of shape (n_query,)
        gallery_labels: Gallery labels of shape (n_gallery,)

    Returns:
        NRR score (lower is better)
    """
    n_query = distance_matrix.size(0)
    n_gallery = distance_matrix.size(1)

    # Compute the number of positive and negative pairs for normalization
    total_positives = 0
    nrr_sum = 0.0

    for i in range(n_query):
        # Get the current query label and distances
        query_label = query_labels[i]
        distances = distance_matrix[i]

        # Find positive and negative gallery samples
        positives = gallery_labels == query_label
        negatives = ~positives

        # Skip if no positive or negative samples
        if not positives.any() or not negatives.any():
            continue

        # Count positive samples
        n_positives = positives.sum().item()
        total_positives += n_positives

        # Sort distances
        sorted_indices = torch.argsort(distances)

        # Find positions of positive samples
        positive_positions = torch.where(positives[sorted_indices])[0]

        # Compute the number of misranked negative samples for each positive
        for pos in positive_positions:
            # Number of negatives ranked before this positive
            misranked = torch.sum(
                (sorted_indices[:pos])[negatives[sorted_indices[:pos]]].numel()
            )
            nrr_sum += misranked / (n_gallery - n_positives)

    # Normalize by total positives
    if total_positives > 0:
        return nrr_sum / total_positives
    else:
        return 0.0


def compute_hard_pair_stats(
    distance_matrix: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_labels: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute statistics about hard positive and hard negative pairs.

    Args:
        distance_matrix: Distance matrix of shape (n_query, n_gallery)
        query_labels: Query labels of shape (n_query,)
        gallery_labels: Gallery labels of shape (n_gallery,)

    Returns:
        Dictionary with hard pair statistics
    """
    results = {}

    # Process each query
    all_pos_dists = []
    all_neg_dists = []
    hardest_pos_dists = []
    hardest_neg_dists = []

    for i in range(distance_matrix.size(0)):
        # Get distances for this query
        distances = distance_matrix[i]
        query_label = query_labels[i]

        # Find positive and negative gallery samples
        positives = gallery_labels == query_label
        negatives = ~positives

        # Skip if no positive or negative samples
        if not positives.any() or not negatives.any():
            continue

        # Collect distances for positive and negative pairs
        pos_dists = distances[positives]
        neg_dists = distances[negatives]

        all_pos_dists.append(pos_dists)
        all_neg_dists.append(neg_dists)

        # Get hardest positive and negative
        hardest_pos_dists.append(pos_dists.max().item())
        hardest_neg_dists.append(neg_dists.min().item())

    # Concatenate all distances
    if all_pos_dists and all_neg_dists:
        all_pos_dists = torch.cat(all_pos_dists)
        all_neg_dists = torch.cat(all_neg_dists)

        # Compute statistics
        results["mean_pos_dist"] = all_pos_dists.mean().item()
        results["mean_neg_dist"] = all_neg_dists.mean().item()
        results["mean_hardest_pos_dist"] = np.mean(hardest_pos_dists)
        results["mean_hardest_neg_dist"] = np.mean(hardest_neg_dists)

        # Compute the percentage of pairs that violate the triplet margin
        margin = 0.2  # Typical margin value
        violations = (
            ((all_pos_dists.unsqueeze(1) - all_neg_dists.unsqueeze(0)) + margin > 0)
            .float()
            .mean()
            .item()
        )
        results["triplet_violation_rate"] = violations

    return results


def compute_embedding_statistics(embeddings: torch.Tensor) -> Dict[str, float]:
    """
    Compute statistics about the embedding space.

    Args:
        embeddings: Embeddings tensor of shape (n, d)

    Returns:
        Dictionary with embedding statistics
    """
    results = {}

    # Compute statistics across samples
    mean_norm = torch.norm(embeddings, dim=1).mean().item()
    std_norm = torch.norm(embeddings, dim=1).std().item()
    min_norm = torch.norm(embeddings, dim=1).min().item()
    max_norm = torch.norm(embeddings, dim=1).max().item()

    results["mean_norm"] = mean_norm
    results["std_norm"] = std_norm
    results["min_norm"] = min_norm
    results["max_norm"] = max_norm

    # Compute statistics across dimensions
    mean_by_dim = embeddings.mean(dim=0)
    std_by_dim = embeddings.std(dim=0)

    results["mean_abs_dim"] = torch.abs(mean_by_dim).mean().item()
    results["mean_std_dim"] = std_by_dim.mean().item()

    # Compute covariance matrix and eigenvalues
    centered = embeddings - embeddings.mean(dim=0)
    cov = torch.mm(centered.t(), centered) / (embeddings.size(0) - 1)
    eigenvalues = torch.linalg.eigvalsh(cov)

    # Compute condition number (ratio of largest to smallest eigenvalue)
    # Add small epsilon to avoid division by zero
    condition_number = eigenvalues[-1] / (eigenvalues[0] + 1e-12)
    results["condition_number"] = condition_number.item()

    # Compute effective rank (number of significant dimensions)
    normalized_eigenvalues = eigenvalues / eigenvalues.sum()
    entropy = -torch.sum(
        normalized_eigenvalues * torch.log(normalized_eigenvalues + 1e-12)
    )
    effective_rank = torch.exp(entropy).item()
    results["effective_rank"] = effective_rank

    return results


if __name__ == "__main__":
    # Example usage
    import torch

    # Create random embeddings and labels for demonstration
    n_query = 50
    n_gallery = 200
    dim = 64

    query_features = torch.rand(n_query, dim)
    gallery_features = torch.rand(n_gallery, dim)

    # Create some overlapping labels for evaluation
    query_labels = torch.randint(0, 10, (n_query,))
    gallery_labels = torch.randint(0, 10, (n_gallery,))

    # Compute distance matrix
    dist_matrix = compute_distance_matrix(query_features, gallery_features, "euclidean")

    # Evaluate embeddings
    results = evaluate_embeddings(
        query_features, gallery_features, query_labels, gallery_labels
    )

    # Print results
    print("Evaluation Results:")
    for k, v in results.items():
        if isinstance(v, np.ndarray):
            print(f"{k}: shape={v.shape}")
        else:
            print(f"{k}: {v:.4f}")

    # Compute hard pair statistics
    hard_pair_stats = compute_hard_pair_stats(dist_matrix, query_labels, gallery_labels)
    print("\nHard Pair Statistics:")
    for k, v in hard_pair_stats.items():
        print(f"{k}: {v:.4f}")

    # Compute embedding statistics
    embed_stats = compute_embedding_statistics(query_features)
    print("\nEmbedding Statistics:")
    for k, v in embed_stats.items():
        print(f"{k}: {v:.4f}")
