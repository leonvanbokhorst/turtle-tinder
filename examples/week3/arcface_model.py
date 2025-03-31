"""
ArcFace model implementation for sea turtle re-identification.

This module implements the ArcFace model as described in the paper:
"ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
by Deng et al. (2019).

ArcFace adds an angular margin penalty to the target logit in the
softmax loss to enhance intra-class compactness and inter-class
discrepancy for face recognition.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple, Optional, Dict, Any


class ArcMarginProduct(nn.Module):
    """
    ArcFace loss function implementation.

    Implements the ArcFace loss with an angular margin penalty to the target logit
    in the softmax loss to enhance intra-class compactness and inter-class discrepancy.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        scale: float = 30.0,
        margin: float = 0.5,
        easy_margin: bool = False,
    ):
        """
        Initialize ArcMarginProduct.

        Args:
            in_features: Size of input features
            out_features: Size of output features (number of classes)
            scale: Scale factor for logits
            margin: Margin for target logit
            easy_margin: Use easy margin formulation
        """
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin

        # Weight matrix representing class centers
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Pre-compute angular margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(
        self, input: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with ArcFace margin.

        Args:
            input: Input features [batch_size, in_features]
            labels: Ground truth labels [batch_size]

        Returns:
            Logits with ArcFace margin applied
        """
        # Normalize features and weights
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        # If no labels provided, just return cosine similarity
        if labels is None:
            return self.scale * cosine

        # Convert to one-hot encoding
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        # Compute cos(theta + m) for the target class
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))

        # cos(θ+m) = cos(θ)cos(m) - sin(θ)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # Handle easy margin cases
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Apply the margin only to the target class
        output = torch.where(one_hot == 1, phi, cosine)

        # Scale output
        output = output * self.scale

        return output


class FeatureExtractor(nn.Module):
    """
    Feature extractor for ArcFace model using a pre-trained backbone.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        embedding_dim: int = 512,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
    ):
        """
        Initialize feature extractor.

        Args:
            backbone: Backbone model to use ('resnet18', 'resnet34', 'resnet50', 'efficientnet_b0', etc.)
            embedding_dim: Dimension of the embedding
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate for feature extractor
        """
        super(FeatureExtractor, self).__init__()
        self.backbone_name = backbone
        self.embedding_dim = embedding_dim

        # Load backbone model
        if backbone.startswith("resnet"):
            base_model = getattr(models, backbone)(pretrained=pretrained)
            in_features = base_model.fc.in_features
            self.backbone = nn.Sequential(*list(base_model.children())[:-1])

        elif backbone.startswith("efficientnet"):
            base_model = getattr(models, backbone)(pretrained=pretrained)
            in_features = base_model.classifier[1].in_features
            self.backbone = base_model.features

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Feature projection layers
        self.neck = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feature extractor.

        Args:
            x: Input tensor of shape [batch_size, channels, height, width]

        Returns:
            Feature embedding of shape [batch_size, embedding_dim]
        """
        # Extract features
        if self.backbone_name.startswith("resnet"):
            x = self.backbone(x)
            x = x.view(x.size(0), -1)
        elif self.backbone_name.startswith("efficientnet"):
            x = self.backbone(x)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = x.view(x.size(0), -1)

        # Project to embedding space
        x = self.neck(x)

        return x


class ArcFaceModel(nn.Module):
    """
    Complete ArcFace model for sea turtle re-identification.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 1000,
        embedding_dim: int = 512,
        margin: float = 0.5,
        scale: float = 30.0,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
    ):
        """
        Initialize ArcFace model.

        Args:
            backbone: Backbone model to use
            num_classes: Number of classes for classification
            embedding_dim: Dimension of the embedding
            margin: ArcFace margin
            scale: ArcFace scale factor
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate for feature extractor
        """
        super(ArcFaceModel, self).__init__()

        # Feature extractor
        self.feature_extractor = FeatureExtractor(
            backbone=backbone,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
        )

        # ArcFace classifier
        self.arc_margin = ArcMarginProduct(
            in_features=embedding_dim,
            out_features=num_classes,
            scale=scale,
            margin=margin,
        )

        # Other attributes
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

    def forward(
        self, x: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through ArcFace model.

        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            labels: Ground truth labels of shape [batch_size]

        Returns:
            Tuple of (logits, features, normalized_features)
        """
        # Extract features
        features = self.feature_extractor(x)

        # Normalize features
        norm_features = F.normalize(features)

        # Apply ArcFace margin and compute logits
        logits = self.arc_margin(features, labels)

        return logits, features, norm_features

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract normalized embedding features for inference.

        Args:
            x: Input tensor of shape [batch_size, channels, height, width]

        Returns:
            Normalized feature embedding of shape [batch_size, embedding_dim]
        """
        with torch.no_grad():
            features = self.feature_extractor(x)
            norm_features = F.normalize(features)

        return norm_features

    def save(self, path: str) -> None:
        """
        Save model checkpoint.

        Args:
            path: Path to save model
        """
        checkpoint = {
            "state_dict": self.state_dict(),
            "num_classes": self.num_classes,
            "embedding_dim": self.embedding_dim,
            "config": {
                "backbone": self.feature_extractor.backbone_name,
                "margin": self.arc_margin.margin,
                "scale": self.arc_margin.scale,
            },
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "ArcFaceModel":
        """
        Load model from checkpoint.

        Args:
            path: Path to load model from
            device: Device to load model to

        Returns:
            Loaded ArcFaceModel
        """
        checkpoint = torch.load(path, map_location=device)

        # Create model
        model = cls(
            backbone=checkpoint["config"]["backbone"],
            num_classes=checkpoint["num_classes"],
            embedding_dim=checkpoint["embedding_dim"],
            margin=checkpoint["config"]["margin"],
            scale=checkpoint["config"]["scale"],
        )

        # Load weights
        model.load_state_dict(checkpoint["state_dict"])

        return model


if __name__ == "__main__":
    # Test model creation and forward pass
    model = ArcFaceModel(
        backbone="resnet18", num_classes=100, embedding_dim=512, margin=0.5, scale=30.0
    )

    # Print model summary
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {num_params:,} trainable parameters")

    # Test forward pass
    batch_size = 8
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    labels = torch.randint(0, 100, (batch_size,))

    # Forward pass with labels
    logits, features, norm_features = model(input_tensor, labels)
    print(f"Logits shape: {logits.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Normalized features shape: {norm_features.shape}")

    # Forward pass without labels (inference mode)
    inference_logits, _, _ = model(input_tensor)
    print(f"Inference logits shape: {inference_logits.shape}")

    # Test get_embedding
    embeddings = model.get_embedding(input_tensor)
    print(f"Embeddings shape: {embeddings.shape}")

    # Check that embeddings are normalized
    norms = torch.norm(embeddings, dim=1)
    print(f"Embedding norms: {norms}")  # Should be close to 1.0
