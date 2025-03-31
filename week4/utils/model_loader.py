"""
Model loading utilities for ensemble methods.

This module provides functions to load different model types from checkpoints,
including models trained with different architectures and frameworks.
"""

import os
import json
import torch
from typing import Dict, Tuple, Any, Optional, Union


def load_model(
    model_path: str, device: Optional[torch.device] = None
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Load a model from a checkpoint file.

    Args:
        model_path: Path to the model checkpoint
        device: Device to load the model to

    Returns:
        Tuple of (model, model_info)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Extract model info
    model_info = {}

    # Get filename without extension
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    model_info["name"] = model_name

    # Check if checkpoint has metadata
    if "model_type" in checkpoint:
        model_type = checkpoint["model_type"]
    elif "args" in checkpoint and "model_type" in checkpoint["args"]:
        model_type = checkpoint["args"]["model_type"]
    else:
        # Try to infer from filename
        if "siamese" in model_name.lower():
            model_type = "siamese"
        elif "triplet" in model_name.lower():
            model_type = "triplet"
        elif "arcface" in model_name.lower():
            model_type = "arcface"
        else:
            model_type = "classification"  # Default to classification

    model_info["model_type"] = model_type

    # Load the model based on its type
    if model_type == "siamese":
        model = load_siamese_model(checkpoint, device)
        model_info["embedding_dim"] = get_embedding_dim(model)
    elif model_type == "triplet":
        model = load_triplet_model(checkpoint, device)
        model_info["embedding_dim"] = get_embedding_dim(model)
    elif model_type == "arcface":
        model = load_arcface_model(checkpoint, device)
        model_info["embedding_dim"] = get_embedding_dim(model)
    elif model_type == "classification":
        model = load_classification_model(checkpoint, device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Add class info
    model_info["num_classes"] = get_num_classes(model, checkpoint)

    # Add backbone info if available
    if "backbone" in checkpoint:
        model_info["backbone"] = checkpoint["backbone"]
    elif "args" in checkpoint and "backbone" in checkpoint["args"]:
        model_info["backbone"] = checkpoint["args"]["backbone"]

    # Check if model has embedding capability
    model_info["has_embedding"] = hasattr(model, "get_embedding")

    return model, model_info


def load_classification_model(
    checkpoint: Dict[str, Any], device: torch.device
) -> torch.nn.Module:
    """
    Load a classification model from checkpoint.

    Args:
        checkpoint: Model checkpoint
        device: Device to load the model to

    Returns:
        Loaded model
    """
    # Import model definitions when needed to avoid circular imports
    import sys
    import importlib.util

    # Try to infer model architecture from checkpoint
    if "architecture" in checkpoint:
        architecture = checkpoint["architecture"]
    elif "args" in checkpoint and "architecture" in checkpoint["args"]:
        architecture = checkpoint["args"]["architecture"]
    else:
        # Default to a common ResNet model
        from torchvision.models import resnet50

        model = resnet50(pretrained=False)

        # Get number of classes from checkpoint
        num_classes = get_num_classes_from_checkpoint(checkpoint)
        if num_classes is not None:
            # Modify the final layer for the correct number of classes
            in_features = model.fc.in_features
            model.fc = torch.nn.Linear(in_features, num_classes)

        # Load state dict
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        elif "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])

        return model.to(device)

    # Try to import the model architecture
    try:
        # Check if the module is in sys.modules
        if architecture in sys.modules:
            module = sys.modules[architecture]
        else:
            # Try to import the module
            spec = importlib.util.find_spec(architecture)
            if spec is not None:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            else:
                raise ImportError(f"Could not find module: {architecture}")

        # Create model from the module
        if hasattr(module, "create_model"):
            model = module.create_model(checkpoint.get("args", {}))
        else:
            # Assume the architecture is the model class name
            model_class = getattr(module, architecture.split(".")[-1])
            model = model_class()

        # Load state dict
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        elif "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])

        return model.to(device)

    except (ImportError, AttributeError) as e:
        print(f"Warning: Failed to load model using architecture: {e}")

        # Fall back to a standard ResNet model
        from torchvision.models import resnet50

        model = resnet50(pretrained=False)

        # Get number of classes from checkpoint
        num_classes = get_num_classes_from_checkpoint(checkpoint)
        if num_classes is not None:
            # Modify the final layer for the correct number of classes
            in_features = model.fc.in_features
            model.fc = torch.nn.Linear(in_features, num_classes)

        # Load state dict
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        elif "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # Assume the entire checkpoint is the state dict
            model.load_state_dict(checkpoint)

        return model.to(device)


def load_siamese_model(
    checkpoint: Dict[str, Any], device: torch.device
) -> torch.nn.Module:
    """
    Load a siamese network model from checkpoint.

    Args:
        checkpoint: Model checkpoint
        device: Device to load the model to

    Returns:
        Loaded model
    """
    # Try to import the Siamese model
    try:
        # First try to import from week3
        sys.path.append(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        from week3.siamese_network import SiameseNetwork

        # Create model
        backbone = checkpoint.get("backbone", "resnet50")
        embedding_dim = checkpoint.get("embedding_dim", 512)

        model = SiameseNetwork(
            backbone=backbone, embedding_dim=embedding_dim, pretrained=False
        )

        # Load state dict
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        elif "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])

        return model.to(device)

    except ImportError:
        raise ImportError(
            "Could not import SiameseNetwork. Make sure the week3 directory is in the Python path."
        )


def load_triplet_model(
    checkpoint: Dict[str, Any], device: torch.device
) -> torch.nn.Module:
    """
    Load a triplet network model from checkpoint.

    Args:
        checkpoint: Model checkpoint
        device: Device to load the model to

    Returns:
        Loaded model
    """
    # Try to import the Triplet model
    try:
        # First try to import from week3
        sys.path.append(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        from week3.triplet_network import TripletNetwork

        # Create model
        backbone = checkpoint.get("backbone", "resnet50")
        embedding_dim = checkpoint.get("embedding_dim", 512)

        model = TripletNetwork(
            backbone=backbone, embedding_dim=embedding_dim, pretrained=False
        )

        # Load state dict
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        elif "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])

        return model.to(device)

    except ImportError:
        raise ImportError(
            "Could not import TripletNetwork. Make sure the week3 directory is in the Python path."
        )


def load_arcface_model(
    checkpoint: Dict[str, Any], device: torch.device
) -> torch.nn.Module:
    """
    Load an ArcFace model from checkpoint.

    Args:
        checkpoint: Model checkpoint
        device: Device to load the model to

    Returns:
        Loaded model
    """
    # Try to import the ArcFace model
    try:
        # First try to import from week3
        sys.path.append(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        from week3.arcface_model import ArcFaceModel

        # Check if the checkpoint has ArcFace.load classmethod
        if hasattr(ArcFaceModel, "load"):
            # Use the load classmethod
            model = ArcFaceModel.load(checkpoint, device)
        else:
            # Create model manually
            backbone = checkpoint.get("config", {}).get("backbone", "resnet50")
            num_classes = checkpoint.get("num_classes", 100)
            embedding_dim = checkpoint.get("embedding_dim", 512)
            margin = checkpoint.get("config", {}).get("margin", 0.5)
            scale = checkpoint.get("config", {}).get("scale", 30.0)

            model = ArcFaceModel(
                backbone=backbone,
                num_classes=num_classes,
                embedding_dim=embedding_dim,
                margin=margin,
                scale=scale,
            )

            # Load state dict
            if "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            elif "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])

        return model.to(device)

    except ImportError:
        raise ImportError(
            "Could not import ArcFaceModel. Make sure the week3 directory is in the Python path."
        )


def get_embedding_dim(model: torch.nn.Module) -> int:
    """
    Get the embedding dimension of a model.

    Args:
        model: PyTorch model

    Returns:
        Embedding dimension
    """
    # Try different methods to get embedding dimension
    if hasattr(model, "embedding_dim"):
        return model.embedding_dim

    if hasattr(model, "feature_extractor") and hasattr(
        model.feature_extractor, "embedding_dim"
    ):
        return model.feature_extractor.embedding_dim

    # For ArcFace models
    if hasattr(model, "arc_margin") and hasattr(model.arc_margin, "in_features"):
        return model.arc_margin.in_features

    # Default
    return 512


def get_num_classes(model: torch.nn.Module, checkpoint: Dict[str, Any]) -> int:
    """
    Get the number of classes of a model.

    Args:
        model: PyTorch model
        checkpoint: Model checkpoint

    Returns:
        Number of classes
    """
    # Try different methods to get number of classes
    if hasattr(model, "num_classes"):
        return model.num_classes

    # For ArcFace models
    if hasattr(model, "arc_margin") and hasattr(model.arc_margin, "out_features"):
        return model.arc_margin.out_features

    # For classification models
    if hasattr(model, "fc") and hasattr(model.fc, "out_features"):
        return model.fc.out_features

    # Check in checkpoint
    num_classes = get_num_classes_from_checkpoint(checkpoint)
    if num_classes is not None:
        return num_classes

    # Default
    return 100


def get_num_classes_from_checkpoint(checkpoint: Dict[str, Any]) -> Optional[int]:
    """
    Extract number of classes from a checkpoint.

    Args:
        checkpoint: Model checkpoint

    Returns:
        Number of classes if found, None otherwise
    """
    # Check common locations for the number of classes
    if "num_classes" in checkpoint:
        return checkpoint["num_classes"]

    if "args" in checkpoint and "num_classes" in checkpoint["args"]:
        return checkpoint["args"]["num_classes"]

    if "config" in checkpoint and "num_classes" in checkpoint["config"]:
        return checkpoint["config"]["num_classes"]

    # Try to infer from the model's final layer
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        # Assume the entire checkpoint is the state dict
        state_dict = checkpoint

    # Look for common final layer weight keys
    final_layer_keys = [
        "fc.weight",
        "classifier.weight",
        "arcface.weight",
        "arc_margin.weight",
        "head.weight",
        "output_layer.weight",
    ]

    for key in final_layer_keys:
        if key in state_dict:
            return state_dict[key].shape[0]

    return None


if __name__ == "__main__":
    # Example usage
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Test model loading")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to load model on"
    )

    args = parser.parse_args()

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    try:
        model, model_info = load_model(args.model_path, device)

        print("Model loaded successfully!")
        print(f"Model type: {model_info['model_type']}")
        print(f"Number of classes: {model_info['num_classes']}")

        if "embedding_dim" in model_info:
            print(f"Embedding dimension: {model_info['embedding_dim']}")

        if "backbone" in model_info:
            print(f"Backbone: {model_info['backbone']}")

        # Print a summary of the model
        print("\nModel summary:")
        print(model)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback

        traceback.print_exc()
