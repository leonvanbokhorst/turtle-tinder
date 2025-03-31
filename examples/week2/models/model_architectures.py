#!/usr/bin/env python3
# Sea Turtle Re-Identification - Week 2
# Model architecture definitions

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List, Optional, Tuple, Union


class TurtleFeatureExtractor(nn.Module):
    """
    A feature extractor for sea turtle images based on pre-trained models.

    This model uses a pre-trained backbone and adds custom layers
    for the specific task of sea turtle identification.
    """

    def __init__(
        self,
        base_model: str = "resnet50",
        num_classes: int = 10,
        dropout_rate: float = 0.5,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ):
        """
        Initialize the TurtleFeatureExtractor.

        Args:
            base_model: Name of the base model architecture
            num_classes: Number of turtle classes to identify
            dropout_rate: Dropout rate for the final classification layer
            pretrained: Whether to use pre-trained weights
            freeze_backbone: Whether to freeze the backbone weights
        """
        super(TurtleFeatureExtractor, self).__init__()

        self.base_model_name = base_model
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Create the base model
        self.base_model, self.feature_dim = self._create_base_model(
            base_model, pretrained
        )

        # Create the classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )

        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()

    def _create_base_model(
        self, base_model: str, pretrained: bool
    ) -> Tuple[nn.Module, int]:
        """
        Create the base model and return it along with its feature dimension.

        Args:
            base_model: Name of the base model architecture
            pretrained: Whether to use pre-trained weights

        Returns:
            Tuple of (base_model, feature_dimension)
        """
        # ResNet models
        if base_model == "resnet18":
            model = models.resnet18(pretrained=pretrained)
            feature_dim = model.fc.in_features
            model.fc = nn.Identity()  # Remove the final FC layer

        elif base_model == "resnet34":
            model = models.resnet34(pretrained=pretrained)
            feature_dim = model.fc.in_features
            model.fc = nn.Identity()

        elif base_model == "resnet50":
            model = models.resnet50(pretrained=pretrained)
            feature_dim = model.fc.in_features
            model.fc = nn.Identity()

        # EfficientNet models
        elif base_model == "efficientnet_b0":
            model = models.efficientnet_b0(pretrained=pretrained)
            feature_dim = model.classifier[1].in_features
            model.classifier = nn.Identity()

        elif base_model == "efficientnet_b3":
            model = models.efficientnet_b3(pretrained=pretrained)
            feature_dim = model.classifier[1].in_features
            model.classifier = nn.Identity()

        # MobileNet
        elif base_model == "mobilenet_v2":
            model = models.mobilenet_v2(pretrained=pretrained)
            feature_dim = model.classifier[1].in_features
            model.classifier = nn.Identity()

        else:
            raise ValueError(f"Unsupported base model: {base_model}")

        return model, feature_dim

    def _freeze_backbone(self):
        """Freeze all parameters in the base model."""
        for param in self.base_model.parameters():
            param.requires_grad = False

    def unfreeze_layers(self, layer_names: List[str]):
        """
        Unfreeze specific layers in the base model.

        Args:
            layer_names: List of layer name patterns to unfreeze
        """
        import re

        for name, param in self.base_model.named_parameters():
            for layer_pattern in layer_names:
                if re.search(layer_pattern, name):
                    param.requires_grad = True
                    break

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape [batch_size, channels, height, width]

        Returns:
            Class logits of shape [batch_size, num_classes]
        """
        # Extract features using the base model
        features = self.base_model(x)

        # Pass through the classifier
        logits = self.classifier(features)

        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features without classification.

        Args:
            x: Input tensor of shape [batch_size, channels, height, width]

        Returns:
            Features of shape [batch_size, feature_dim]
        """
        return self.base_model(x)


class TurtleEnsemble(nn.Module):
    """
    An ensemble of multiple models for sea turtle identification.

    This model combines predictions from multiple pre-trained models
    for more robust identification.
    """

    def __init__(
        self,
        base_models: List[str] = ["resnet50", "efficientnet_b0"],
        num_classes: int = 10,
        pretrained: bool = True,
    ):
        """
        Initialize the ensemble.

        Args:
            base_models: List of base model architectures to include
            num_classes: Number of turtle classes to identify
            pretrained: Whether to use pre-trained weights
        """
        super(TurtleEnsemble, self).__init__()

        # Create a model for each base architecture
        self.models = nn.ModuleList(
            [
                TurtleFeatureExtractor(
                    base_model=model_name,
                    num_classes=num_classes,
                    pretrained=pretrained,
                    freeze_backbone=False,  # We assume the models are already fine-tuned
                )
                for model_name in base_models
            ]
        )

        self.num_models = len(base_models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all models and average predictions.

        Args:
            x: Input tensor of shape [batch_size, channels, height, width]

        Returns:
            Averaged logits of shape [batch_size, num_classes]
        """
        # Get predictions from each model
        all_logits = []
        for model in self.models:
            logits = model(x)
            all_logits.append(logits)

        # Average the logits
        avg_logits = torch.stack(all_logits, dim=0).mean(dim=0)

        return avg_logits


# Dictionary of available models for easy access
AVAILABLE_MODELS = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "efficientnet_b0": models.efficientnet_b0,
    "efficientnet_b3": models.efficientnet_b3,
    "mobilenet_v2": models.mobilenet_v2,
}


def create_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    """
    Factory function to create models with proper initialization.

    Args:
        model_name: Name of the model architecture
        num_classes: Number of classes for classification
        pretrained: Whether to use pre-trained weights
        freeze_backbone: Whether to freeze the backbone weights

    Returns:
        The initialized model
    """
    if "ensemble" in model_name.lower():
        # Create an ensemble of models
        base_models = model_name.split("_")[1:]  # Extract base model names
        return TurtleEnsemble(
            base_models=base_models, num_classes=num_classes, pretrained=pretrained
        )

    elif model_name in AVAILABLE_MODELS:
        # Use our feature extractor wrapper
        return TurtleFeatureExtractor(
            base_model=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")


# Example usage when run directly
if __name__ == "__main__":
    # Test model creation
    model = create_model("resnet50", num_classes=10, freeze_backbone=True)
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(
        f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})"
    )

    # Test a forward pass
    x = torch.randn(4, 3, 224, 224)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    # Test the ensemble
    ensemble = create_model("ensemble_resnet50_efficientnet_b0", num_classes=10)
    print("\nEnsemble:", ensemble)
    y_ensemble = ensemble(x)
    print(f"Ensemble output shape: {y_ensemble.shape}")
