"""
Siamese Network implementation for sea turtle re-identification.

This module implements a Siamese neural network for learning embeddings
that place similar sea turtle images close together in feature space and
dissimilar images far apart.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple, Dict, Optional


class FeatureExtractor(nn.Module):
    """
    Feature extractor backbone for the Siamese network.
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


class SiameseNetwork(nn.Module):
    """
    Siamese network for learning embeddings that capture similarity
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
        Initialize the Siamese network.

        Args:
            backbone: Name of the backbone model to use
            embedding_dim: Dimension of the output embedding
            pretrained: Whether to use pre-trained weights
            dropout: Dropout probability in the embedding head
        """
        super(SiameseNetwork, self).__init__()

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

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for a pair of images.

        Args:
            x1: First input tensor of shape [batch_size, channels, height, width]
            x2: Second input tensor of shape [batch_size, channels, height, width]

        Returns:
            Tuple of (embedding1, embedding2, distance) where distance is
            the Euclidean distance between the embeddings
        """
        # Get embeddings for both images
        embedding1 = self.forward_one(x1)
        embedding2 = self.forward_one(x2)

        # Calculate Euclidean distance between embeddings
        distance = F.pairwise_distance(embedding1, embedding2)

        return embedding1, embedding2, distance


def contrastive_loss(
    embedding1: torch.Tensor,
    embedding2: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """
    Contrastive loss function for Siamese networks.

    Args:
        embedding1: First image embedding of shape [batch_size, embedding_dim]
        embedding2: Second image embedding of shape [batch_size, embedding_dim]
        labels: 1 if images are from same class, 0 otherwise. Shape [batch_size]
        margin: Minimum distance for negative pairs

    Returns:
        Loss value
    """
    # Convert labels to float
    labels = labels.float()

    # Calculate Euclidean distance
    distances = F.pairwise_distance(embedding1, embedding2)

    # Contrastive loss has two components:
    # Same class (label=1): minimize distance
    # Different class (label=0): maximize distance, up to margin
    same_class_loss = labels * torch.pow(distances, 2)
    diff_class_loss = (1.0 - labels) * torch.pow(
        torch.clamp(margin - distances, min=0.0), 2
    )

    # Combine both components
    loss = torch.mean(same_class_loss + diff_class_loss) / 2.0

    return loss


def create_model(model_config: Dict) -> Tuple[SiameseNetwork, nn.Module]:
    """
    Factory function to create a Siamese model with loss function.

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
    margin = model_config.get("margin", 1.0)

    # Create model
    model = SiameseNetwork(
        backbone=backbone,
        embedding_dim=embedding_dim,
        pretrained=pretrained,
        dropout=dropout,
    )

    # Create loss function
    loss_fn = lambda e1, e2, l: contrastive_loss(e1, e2, l, margin=margin)

    return model, loss_fn


if __name__ == "__main__":
    # Demonstrate model creation and forward pass
    model = SiameseNetwork(backbone="resnet50", embedding_dim=128)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass with random input
    batch_size = 8
    x1 = torch.randn(batch_size, 3, 224, 224)
    x2 = torch.randn(batch_size, 3, 224, 224)

    embeddings1, embeddings2, distances = model(x1, x2)

    print(f"Input shape: {x1.shape}")
    print(f"Embedding shape: {embeddings1.shape}")
    print(f"Distance shape: {distances.shape}")

    # Test loss function
    # 0: different turtles, 1: same turtle
    labels = torch.randint(0, 2, (batch_size,))
    loss = contrastive_loss(embeddings1, embeddings2, labels)
    print(f"Loss shape: {loss.shape}")
    print(f"Loss value: {loss.item():.4f}")
