#!/usr/bin/env python
"""
Model fusion for sea turtle re-identification.

This script demonstrates how to combine features from different backbone
architectures to create a more robust re-identification system.
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.model_loader import load_model
from utils.data_utils import prepare_dataloaders


class FeatureExtractor(nn.Module):
    """
    Feature extractor wrapper for pre-trained models.
    """

    def __init__(self, model, model_type):
        """
        Initialize feature extractor.

        Args:
            model: Pre-trained model
            model_type: Type of model ('classification', 'siamese', 'triplet', 'arcface')
        """
        super(FeatureExtractor, self).__init__()
        self.model = model
        self.model_type = model_type

    def forward(self, x):
        """
        Extract features from the model.

        Args:
            x: Input tensor

        Returns:
            Extracted features
        """
        # Extract features based on model type
        if self.model_type in ["siamese", "triplet"]:
            # Use get_embedding for metric learning models
            return self.model.get_embedding(x)
        elif self.model_type == "arcface":
            # ArcFace typically provides normalized embeddings
            _, _, normalized_features = self.model(x)
            return normalized_features
        else:
            # For classification models, remove the last layer
            if hasattr(self.model, "features"):
                # EfficientNet, MobileNet style
                features = self.model.features(x)
                features = F.adaptive_avg_pool2d(features, (1, 1))
                features = torch.flatten(features, 1)
                return features
            else:
                # ResNet style
                # Get all layers except the last one
                modules = list(self.model.children())[:-1]
                feature_extractor = nn.Sequential(*modules)
                features = feature_extractor(x)
                features = torch.flatten(features, 1)
                return features


class FusionModel(nn.Module):
    """
    Model for fusing features from multiple backbones.
    """

    def __init__(
        self,
        feature_extractors,
        input_dims,
        hidden_dim=512,
        num_classes=100,
        fusion_method="concat",
    ):
        """
        Initialize fusion model.

        Args:
            feature_extractors: List of feature extractor models
            input_dims: List of input dimensions for each feature extractor
            hidden_dim: Hidden dimension for fusion layers
            num_classes: Number of classes
            fusion_method: Method to fuse features ('concat', 'sum', 'attention')
        """
        super(FusionModel, self).__init__()
        self.feature_extractors = nn.ModuleList(feature_extractors)
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.fusion_method = fusion_method

        # Create dimension reduction layers
        self.dim_reduction = nn.ModuleList()
        for dim in input_dims:
            self.dim_reduction.append(
                nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                )
            )

        # Calculate output dimension based on fusion method
        if fusion_method == "concat":
            output_dim = hidden_dim * len(feature_extractors)
        elif fusion_method in ["sum", "attention"]:
            output_dim = hidden_dim
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        # Create attention weights if needed
        if fusion_method == "attention":
            self.attention = nn.Parameter(
                torch.ones(len(feature_extractors)) / len(feature_extractors)
            )

        # Create classifier
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.BatchNorm1d(output_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(output_dim // 2, num_classes),
        )

    def forward(self, x):
        """
        Forward pass through the fusion model.

        Args:
            x: Input tensor

        Returns:
            Class logits
        """
        # Extract features from each backbone
        all_features = []
        for i, extractor in enumerate(self.feature_extractors):
            # Extract features
            features = extractor(x)

            # Apply dimension reduction
            reduced_features = self.dim_reduction[i](features)
            all_features.append(reduced_features)

        # Fuse features based on method
        if self.fusion_method == "concat":
            fused_features = torch.cat(all_features, dim=1)
        elif self.fusion_method == "sum":
            fused_features = torch.stack(all_features).sum(dim=0)
        elif self.fusion_method == "attention":
            # Apply softmax to attention weights
            attention_weights = F.softmax(self.attention, dim=0)

            # Weight features
            weighted_features = [
                all_features[i] * attention_weights[i] for i in range(len(all_features))
            ]

            # Sum weighted features
            fused_features = torch.stack(weighted_features).sum(dim=0)

        # Apply classifier
        logits = self.classifier(fused_features)

        return logits

    def get_feature_extractors(self):
        """
        Get the feature extractors.

        Returns:
            List of feature extractors
        """
        return self.feature_extractors


def train_fusion_model(
    model,
    train_loader,
    val_loader,
    num_epochs=10,
    learning_rate=0.001,
    weight_decay=1e-5,
    device="cuda",
):
    """
    Train the fusion model.

    Args:
        model: Fusion model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
        device: Device to train on

    Returns:
        Tuple of (trained model, training history)
    """
    # Move model to device
    model = model.to(device)

    # Set up optimizer and loss function
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    # Training history
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # Training loop
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, targets in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"
        ):
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update metrics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

        # Calculate training metrics
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"
            ):
                # Move data to device
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Update metrics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        # Calculate validation metrics
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        # Update history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Print progress
        print(
            f"Epoch {epoch+1}/{num_epochs}: "
            f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
            f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
        )

    return model, history


def evaluate_model(model, data_loader, device="cuda"):
    """
    Evaluate the model on a dataset.

    Args:
        model: Model to evaluate
        data_loader: Data loader for evaluation
        device: Device to evaluate on

    Returns:
        Tuple of (accuracy, predictions, targets)
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="Evaluating"):
            # Move data to device
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            # Store predictions and targets
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())

    # Calculate accuracy
    accuracy = accuracy_score(all_targets, all_preds)

    return accuracy, all_preds, all_targets


def visualize_training(history, output_path):
    """
    Visualize training history.

    Args:
        history: Training history dictionary
        output_path: Path to save visualization
    """
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_model(model, history, output_path, config):
    """
    Save trained model and training history.

    Args:
        model: Trained model
        history: Training history
        output_path: Path to save model
        config: Model configuration
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save model
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_type": "fusion",
        "config": config,
        "history": history,
    }

    torch.save(checkpoint, output_path)
    print(f"Model saved to {output_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Model fusion for sea turtle re-identification"
    )

    # Required arguments
    parser.add_argument(
        "--backbone1",
        type=str,
        required=True,
        help="First backbone architecture or model path",
    )
    parser.add_argument(
        "--backbone2",
        type=str,
        required=True,
        help="Second backbone architecture or model path",
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Directory containing the data"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save results"
    )

    # Optional arguments
    parser.add_argument(
        "--fusion_method",
        type=str,
        default="concat",
        choices=["concat", "sum", "attention"],
        help="Method to fuse features (default: concat)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=512,
        help="Hidden dimension for fusion layers (default: 512)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of epochs (default: 10)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-5, help="Weight decay (default: 1e-5)"
    )
    parser.add_argument(
        "--device", type=str, default="", help="Device to use (default: auto)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load models
    print("Loading models...")
    model1, info1 = load_model(args.backbone1, device)
    model2, info2 = load_model(args.backbone2, device)

    print(f"Model 1: {info1['model_type']} - {args.backbone1}")
    print(f"Model 2: {info2['model_type']} - {args.backbone2}")

    # Create feature extractors
    feature_extractor1 = FeatureExtractor(model1, info1["model_type"])
    feature_extractor2 = FeatureExtractor(model2, info2["model_type"])

    # Determine input dimensions
    # Run a forward pass with a dummy input to get the feature dimensions
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        feat1 = feature_extractor1(dummy_input)
        feat2 = feature_extractor2(dummy_input)
        input_dim1 = feat1.size(1)
        input_dim2 = feat2.size(1)

    print(f"Feature dimensions: {input_dim1}, {input_dim2}")

    # Create feature extractors and fusion model
    feature_extractors = [feature_extractor1, feature_extractor2]
    input_dims = [input_dim1, input_dim2]

    # Load data
    print("\nLoading data...")
    train_loader, class_names = prepare_dataloaders(
        data_dir=args.data_dir, split="train", batch_size=args.batch_size
    )

    val_loader, _ = prepare_dataloaders(
        data_dir=args.data_dir, split="val", batch_size=args.batch_size
    )

    test_loader, _ = prepare_dataloaders(
        data_dir=args.data_dir, split="test", batch_size=args.batch_size
    )

    print(f"Loaded {len(train_loader.dataset)} training samples")
    print(f"Loaded {len(val_loader.dataset)} validation samples")
    print(f"Loaded {len(test_loader.dataset)} test samples")
    print(f"Number of classes: {len(class_names)}")

    # Create fusion model
    fusion_model = FusionModel(
        feature_extractors=feature_extractors,
        input_dims=input_dims,
        hidden_dim=args.hidden_dim,
        num_classes=len(class_names),
        fusion_method=args.fusion_method,
    )

    # Print model summary
    print("\nFusion model summary:")
    print(f"Feature extractors: {len(feature_extractors)}")
    print(f"Input dimensions: {input_dims}")
    print(f"Hidden dimension: {args.hidden_dim}")
    print(f"Fusion method: {args.fusion_method}")
    print(f"Number of classes: {len(class_names)}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    config = {
        "backbone1": args.backbone1,
        "backbone2": args.backbone2,
        "fusion_method": args.fusion_method,
        "hidden_dim": args.hidden_dim,
        "input_dims": input_dims,
        "num_classes": len(class_names),
        "class_names": class_names,
    }

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # Train fusion model
    print("\nTraining fusion model...")
    trained_model, history = train_fusion_model(
        model=fusion_model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
    )

    # Visualize training
    visualize_training(
        history=history,
        output_path=os.path.join(args.output_dir, "training_history.png"),
    )

    # Save model
    save_model(
        model=trained_model,
        history=history,
        output_path=os.path.join(args.output_dir, "fusion_model.pth"),
        config=config,
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_accuracy, test_preds, test_targets = evaluate_model(
        model=trained_model, data_loader=test_loader, device=device
    )

    print(f"Test accuracy: {test_accuracy:.4f}")

    # Generate classification report
    report = classification_report(
        test_targets, test_preds, target_names=class_names, output_dict=True
    )

    # Save report
    with open(os.path.join(args.output_dir, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=4)

    # Evaluate individual models for comparison
    print("\nEvaluating individual models for comparison...")

    # Create a simple classifier for model 1
    model1_classifier = nn.Sequential(
        feature_extractor1, nn.Linear(input_dim1, len(class_names))
    ).to(device)

    # Create a simple classifier for model 2
    model2_classifier = nn.Sequential(
        feature_extractor2, nn.Linear(input_dim2, len(class_names))
    ).to(device)

    # Train and evaluate model 1
    print("\nTraining classifier for model 1...")
    model1_classifier, model1_history = train_fusion_model(
        model=model1_classifier,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
    )

    # Evaluate model 1
    model1_accuracy, _, _ = evaluate_model(
        model=model1_classifier, data_loader=test_loader, device=device
    )

    # Train and evaluate model 2
    print("\nTraining classifier for model 2...")
    model2_classifier, model2_history = train_fusion_model(
        model=model2_classifier,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
    )

    # Evaluate model 2
    model2_accuracy, _, _ = evaluate_model(
        model=model2_classifier, data_loader=test_loader, device=device
    )

    # Compare results
    print("\nResults comparison:")
    print(f"Model 1 accuracy: {model1_accuracy:.4f}")
    print(f"Model 2 accuracy: {model2_accuracy:.4f}")
    print(f"Fusion model accuracy: {test_accuracy:.4f}")

    # Calculate improvement
    model1_improvement = (test_accuracy - model1_accuracy) / model1_accuracy * 100
    model2_improvement = (test_accuracy - model2_accuracy) / model2_accuracy * 100

    print(f"Improvement over model 1: {model1_improvement:.2f}%")
    print(f"Improvement over model 2: {model2_improvement:.2f}%")

    # Save comparison results
    comparison = {
        "model1_accuracy": float(model1_accuracy),
        "model2_accuracy": float(model2_accuracy),
        "fusion_accuracy": float(test_accuracy),
        "model1_improvement": float(model1_improvement),
        "model2_improvement": float(model2_improvement),
    }

    with open(os.path.join(args.output_dir, "model_comparison.json"), "w") as f:
        json.dump(comparison, f, indent=4)

    print(f"\nAll results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
