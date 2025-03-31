"""
Triplet Network implementation for sea turtle re-identification.

This module implements a Triplet neural network for learning embeddings
that place similar sea turtle images close together in feature space and
dissimilar images far apart. It uses triplet loss with hard triplet mining.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple, Dict, List, Optional, Any, Callable


class FeatureExtractor(nn.Module):
    """
    Feature extractor backbone for the Triplet network.
    Extracts features from input images using a pre-trained CNN.
    """

    def __init__(self, backbone: str = "resnet50", pretrained: bool = True):
        """
        Initialize the feature extractor.

        Args:
            backbone: Name of the backbone model to use
            pretrained: Whether to use pre-trained weights
        """
        super(FeatureExtractor, self).__init__()

        # Available model options
        self.available_models = {
            "resnet18": (models.resnet18, 512),
            "resnet34": (models.resnet34, 512),
            "resnet50": (models.resnet50, 2048),
            "resnet101": (models.resnet101, 2048),
            "efficientnet_b0": (models.efficientnet_b0, 1280),
            "efficientnet_b1": (models.efficientnet_b1, 1280),
            "efficientnet_b2": (models.efficientnet_b2, 1408),
            "efficientnet_b3": (models.efficientnet_b3, 1536),
            "mobilenet_v2": (models.mobilenet_v2, 1280),
            "mobilenet_v3_small": (models.mobilenet_v3_small, 576),
            "mobilenet_v3_large": (models.mobilenet_v3_large, 960),
        }

        if backbone not in self.available_models:
            raise ValueError(
                f"Backbone {backbone} not supported. Choose from: {list(self.available_models.keys())}"
            )

        # Get model constructor and feature dimension
        model_constructor, self.output_dim = self.available_models[backbone]

        # Create the base model
        base_model = model_constructor(pretrained=pretrained)

        # Remove the classification head
        if backbone.startswith("resnet"):
            self.features = nn.Sequential(*list(base_model.children())[:-1])
        elif backbone.startswith("efficientnet"):
            self.features = base_model.features
        elif backbone.startswith("mobilenet"):
            self.features = base_model.features
        else:
            raise ValueError(f"Feature extraction not implemented for {backbone}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feature extractor.

        Args:
            x: Input tensor of shape [batch_size, channels, height, width]

        Returns:
            Feature tensor of shape [batch_size, output_dim]
        """
        features = self.features(x)

        # Handle different model outputs
        if len(features.shape) == 4:  # [batch_size, channels, h, w]
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = torch.flatten(features, 1)

        return features


class TripletNetwork(nn.Module):
    """
    Triplet network for learning embeddings that capture similarity
    between sea turtle images.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        embedding_dim: int = 128,
        pretrained: bool = True,
        dropout: float = 0.5,
    ):
        """
        Initialize the Triplet network.

        Args:
            backbone: Name of the backbone model to use
            embedding_dim: Dimension of the output embedding
            pretrained: Whether to use pre-trained weights
            dropout: Dropout probability in the embedding head
        """
        super(TripletNetwork, self).__init__()

        # Create the feature extractor backbone
        self.backbone = FeatureExtractor(backbone=backbone, pretrained=pretrained)

        # Add an embedding projection head
        self.embedding = nn.Sequential(
            nn.Linear(self.backbone.output_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features and create normalized embedding for one image.

        Args:
            x: Input tensor of shape [batch_size, channels, height, width]

        Returns:
            Normalized embedding tensor of shape [batch_size, embedding_dim]
        """
        # Extract features from the backbone
        features = self.backbone(x)

        # Create embedding
        embedding = self.embedding(features)

        # Normalize embedding to unit length
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a batch of images.
        Unlike Siamese networks, Triplet networks process each
        input independently and triplets are formed during loss calculation.

        Args:
            x: Input tensor of shape [batch_size, channels, height, width]

        Returns:
            Embedding tensor of shape [batch_size, embedding_dim]
        """
        return self.forward_one(x)


def triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 0.3,
) -> torch.Tensor:
    """
    Compute triplet loss.

    Args:
        anchor: Anchor embeddings of shape [batch_size, embedding_dim]
        positive: Positive embeddings (same class as anchor) of shape [batch_size, embedding_dim]
        negative: Negative embeddings (different class from anchor) of shape [batch_size, embedding_dim]
        margin: Minimum desired distance between (anchor, negative) and (anchor, positive)

    Returns:
        Loss value
    """
    # Calculate distances
    pos_dist = torch.pairwise_distance(anchor, positive)
    neg_dist = torch.pairwise_distance(anchor, negative)

    # Compute triplet loss: max(0, pos_dist - neg_dist + margin)
    loss = torch.clamp(pos_dist - neg_dist + margin, min=0.0)

    # Return mean loss and the number of triplets with non-zero loss (hard triplets)
    num_hard_triplets = torch.sum(loss > 1e-16).item()
    return torch.mean(loss), num_hard_triplets


def batch_hard_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.3,
    squared: bool = False,
) -> Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
    """
    Compute batch hard triplet loss, mining the hardest triplets within a batch.

    Args:
        embeddings: Tensor of shape [batch_size, embedding_dim]
        labels: Tensor of shape [batch_size]
        margin: Minimum desired distance between (anchor, negative) and (anchor, positive)
        squared: Whether to use squared distances

    Returns:
        Tuple of (loss, num_hard_triplets, hardest_positive_dist, hardest_negative_dist)
    """
    # Get the device of the tensors
    device = embeddings.device

    # Calculate pairwise distances
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*<a, b>
    # But embeddings are normalized, so ||a||^2 = ||b||^2 = 1
    # So ||a - b||^2 = 2 - 2*<a, b> = 2*(1 - <a, b>)

    # Calculate dot product
    dot_product = torch.matmul(embeddings, embeddings.t())

    # Calculate squared L2 distance matrix
    distances = 2.0 * (1.0 - dot_product)

    # Because of computation errors, some distances might be negative
    distances = torch.maximum(distances, torch.tensor(0.0).to(device))

    if not squared:
        # Get actual L2 distances
        distances = torch.sqrt(distances + 1e-8)

    # Create a mask for valid triplets
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    labels_not_equal = ~labels_equal

    # For each anchor, find the hardest positive and hardest negative
    hardest_positive_dist = torch.max(distances * labels_equal.float(), dim=1)[0]

    # For each anchor, find the hardest negative
    # First, get the maximum distance to use as placeholder for invalid negatives
    max_dist = torch.max(distances).item()

    # Then get hardest negative: minimum distance to any negative example
    # We add max_dist to invalid negatives to make sure they are not selected
    hardest_negative_dist = torch.min(
        distances + max_dist * labels_equal.float(), dim=1
    )[0]

    # Compute triplet loss
    triplet_loss = torch.clamp(
        hardest_positive_dist - hardest_negative_dist + margin, min=0.0
    )

    # Count number of hard triplets (triplets that incur a non-zero loss)
    num_hard_triplets = torch.sum(triplet_loss > 1e-16).item()

    # Get the mean triplet loss
    return (
        torch.mean(triplet_loss),
        num_hard_triplets,
        hardest_positive_dist,
        hardest_negative_dist,
    )


def mine_hard_triplets(
    embeddings: torch.Tensor, labels: torch.Tensor, margin: float = 0.3
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Mine hard triplets from a batch of embeddings.

    Args:
        embeddings: Tensor of shape [batch_size, embedding_dim]
        labels: Tensor of shape [batch_size]
        margin: Triplet loss margin

    Returns:
        Tensors for anchor, positive, and negative indices
    """
    # Calculate pairwise distances
    distances = torch.cdist(embeddings, embeddings)

    # Initialize lists for triplet indices
    anchor_idx, positive_idx, negative_idx = [], [], []

    # For each anchor
    for i in range(len(embeddings)):
        anchor_label = labels[i]

        # Find positive indices (same label as anchor)
        pos_indices = torch.where(labels == anchor_label)[0]
        # Remove the anchor itself
        pos_indices = pos_indices[pos_indices != i]

        if len(pos_indices) == 0:
            continue  # Skip if no positives

        # Find negative indices (different label from anchor)
        neg_indices = torch.where(labels != anchor_label)[0]

        if len(neg_indices) == 0:
            continue  # Skip if no negatives

        # Get distances to positives and negatives
        pos_distances = distances[i, pos_indices]
        neg_distances = distances[i, neg_indices]

        # Find hardest positive (furthest positive)
        hardest_pos_idx = pos_indices[torch.argmax(pos_distances)]

        # Find semi-hard negatives (negatives that are further than positives but within margin)
        hardest_pos_dist = torch.max(pos_distances)
        semi_hard_negs = neg_indices[
            (neg_distances > hardest_pos_dist)
            & (neg_distances < hardest_pos_dist + margin)
        ]

        # If no semi-hard negatives found, use the closest negative
        if len(semi_hard_negs) == 0:
            hardest_neg_idx = neg_indices[torch.argmin(neg_distances)]
        else:
            # Choose a random semi-hard negative
            random_idx = torch.randint(0, len(semi_hard_negs), (1,))[0]
            hardest_neg_idx = semi_hard_negs[random_idx]

        # Add to triplet lists
        anchor_idx.append(i)
        positive_idx.append(hardest_pos_idx)
        negative_idx.append(hardest_neg_idx)

    return (
        torch.tensor(anchor_idx),
        torch.tensor(positive_idx),
        torch.tensor(negative_idx),
    )


def create_model(model_config: Dict) -> Tuple[TripletNetwork, Callable]:
    """
    Factory function to create a Triplet model with loss function.

    Args:
        model_config: Dictionary with model configuration

    Returns:
        Tuple of (model, loss_function)
    """
    # Extract model parameters from config
    backbone = model_config.get("backbone", "resnet50")
    embedding_dim = model_config.get("embedding_dim", 128)
    pretrained = model_config.get("pretrained", True)
    dropout = model_config.get("dropout", 0.5)
    margin = model_config.get("margin", 0.3)
    mining_strategy = model_config.get("mining_strategy", "batch_hard")

    # Create model
    model = TripletNetwork(
        backbone=backbone,
        embedding_dim=embedding_dim,
        pretrained=pretrained,
        dropout=dropout,
    )

    # Create loss function based on mining strategy
    if mining_strategy == "batch_hard":
        loss_fn = lambda embeddings, labels: batch_hard_triplet_loss(
            embeddings, labels, margin=margin
        )
    elif mining_strategy == "batch_all":
        # This would typically mine all valid triplets in a batch
        # For simplicity, we'll just use the batch_hard implementation
        loss_fn = lambda embeddings, labels: batch_hard_triplet_loss(
            embeddings, labels, margin=margin
        )
    else:
        # Custom mining - requires a function that returns anchor, positive, negative indices
        def custom_loss_fn(embeddings, labels):
            anchor_idx, pos_idx, neg_idx = mine_hard_triplets(
                embeddings, labels, margin=margin
            )

            if len(anchor_idx) == 0:
                # No valid triplets found
                return torch.tensor(0.0, device=embeddings.device), 0

            # Get embeddings for the triplets
            anchor = embeddings[anchor_idx]
            positive = embeddings[pos_idx]
            negative = embeddings[neg_idx]

            # Compute triplet loss
            return triplet_loss(anchor, positive, negative, margin=margin)

        loss_fn = custom_loss_fn

    return model, loss_fn


if __name__ == "__main__":
    # Demonstrate model creation and forward pass
    model = TripletNetwork(backbone="resnet50", embedding_dim=128)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass with random input
    batch_size = 8
    x = torch.randn(batch_size, 3, 224, 224)

    embeddings = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Embedding shape: {embeddings.shape}")

    # Test batch hard triplet loss
    # Create fake labels: [0, 0, 1, 1, 2, 2, 3, 3]
    labels = torch.tensor([i // 2 for i in range(batch_size)])

    loss, num_hard_triplets, hardest_pos_dist, hardest_neg_dist = (
        batch_hard_triplet_loss(embeddings, labels, margin=0.3)
    )

    print(f"Loss value: {loss.item():.4f}")
    print(f"Number of hard triplets: {num_hard_triplets}")
    print(
        f"Average hardest positive distance: {torch.mean(hardest_pos_dist).item():.4f}"
    )
    print(
        f"Average hardest negative distance: {torch.mean(hardest_neg_dist).item():.4f}"
    )
