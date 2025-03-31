"""
Pair visualization utility for Siamese networks.

This module provides utilities for visualizing pairs of sea turtle images
and the predictions made by a Siamese network.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import torchvision.transforms as transforms
from PIL import Image
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
import random

# Try to import the Siamese network
try:
    from siamese_network import SiameseNetwork
except ImportError:
    print(
        "Warning: Could not import SiameseNetwork. Make sure siamese_network.py is in the same directory."
    )


def load_image_pair(
    img1_path: str, img2_path: str, transform: Optional[Callable] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load a pair of images and apply transforms.

    Args:
        img1_path: Path to the first image
        img2_path: Path to the second image
        transform: Optional transform to apply to the images

    Returns:
        Tuple of (transformed_img1, transformed_img2)
    """
    # Load images
    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")

    # Apply transform if provided
    if transform:
        img1 = transform(img1)
        img2 = transform(img2)

    return img1, img2


def visualize_pair(
    img1: Union[torch.Tensor, np.ndarray, Image.Image],
    img2: Union[torch.Tensor, np.ndarray, Image.Image],
    distance: Optional[float] = None,
    label: Optional[int] = None,
    prediction: Optional[int] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 5),
    output_path: Optional[str] = None,
    dpi: int = 300,
) -> Figure:
    """
    Visualize a pair of images with optional distance, label, and prediction.

    Args:
        img1: First image
        img2: Second image
        distance: Optional distance between embeddings
        label: Optional ground truth label (1 if same class, 0 if different)
        prediction: Optional predicted label
        title: Optional title for the figure
        figsize: Figure size
        output_path: Optional path to save the figure
        dpi: DPI for saving the figure

    Returns:
        Matplotlib figure
    """
    # Convert images to appropriate format for display
    if isinstance(img1, torch.Tensor):
        # Handle different tensor shapes
        if img1.dim() == 4:  # Batch of images
            img1 = img1[0]

        # Convert from tensor (C, H, W) to numpy (H, W, C)
        img1 = img1.permute(1, 2, 0).cpu().numpy()

        # Handle normalization
        if img1.min() < 0 or img1.max() > 1:
            # Assume image was normalized with ImageNet stats
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img1 = img1 * std + mean
            img1 = np.clip(img1, 0, 1)

    if isinstance(img2, torch.Tensor):
        # Handle different tensor shapes
        if img2.dim() == 4:  # Batch of images
            img2 = img2[0]

        # Convert from tensor (C, H, W) to numpy (H, W, C)
        img2 = img2.permute(1, 2, 0).cpu().numpy()

        # Handle normalization
        if img2.min() < 0 or img2.max() > 1:
            # Assume image was normalized with ImageNet stats
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img2 = img2 * std + mean
            img2 = np.clip(img2, 0, 1)

    # Convert PIL images to numpy
    if isinstance(img1, Image.Image):
        img1 = np.array(img1) / 255.0

    if isinstance(img2, Image.Image):
        img2 = np.array(img2) / 255.0

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Display images
    axes[0].imshow(img1)
    axes[0].set_title("Image 1")
    axes[0].axis("off")

    axes[1].imshow(img2)
    axes[1].set_title("Image 2")
    axes[1].axis("off")

    # Create a panel for the results
    axes[2].axis("off")

    # Add information text
    info_text = ""

    if distance is not None:
        info_text += f"Distance: {distance:.4f}\n\n"

    if label is not None:
        label_text = "Same Turtle" if label == 1 else "Different Turtles"
        info_text += f"Ground Truth: {label_text}\n\n"

    if prediction is not None:
        pred_text = "Same Turtle" if prediction == 1 else "Different Turtles"
        color = "green" if prediction == label else "red"
        info_text += f"Prediction: {pred_text}"

    # Display info text
    axes[2].text(
        0.5,
        0.5,
        info_text,
        ha="center",
        va="center",
        fontsize=12,
        transform=axes[2].transAxes,
        bbox=dict(facecolor="white", alpha=0.8),
    )

    # Add title
    if title:
        plt.suptitle(title, fontsize=16)

    # Adjust layout
    plt.tight_layout()

    # Save figure if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=dpi)

    return fig


def visualize_pairs_from_model(
    model: nn.Module,
    img_pairs: List[Tuple[str, str, int]],
    output_dir: str,
    threshold: float = 0.5,
    transform: Optional[Callable] = None,
    prefix: str = "",
    device: Optional[torch.device] = None,
) -> None:
    """
    Visualize pairs of images with predictions from a Siamese model.

    Args:
        model: Siamese model
        img_pairs: List of (img1_path, img2_path, label) tuples
        output_dir: Directory to save visualizations
        threshold: Threshold for prediction (distance < threshold -> same class)
        transform: Transform to apply to images (if None, use a default transform)
        prefix: Prefix for output filenames
        device: Device to run inference on
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set model to evaluation mode
    model.eval()

    # Create default transform if not provided
    if transform is None:
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each pair
    for i, (img1_path, img2_path, label) in enumerate(img_pairs):
        # Load and transform images
        img1, img2 = load_image_pair(img1_path, img2_path, transform)

        # Move to device
        img1 = img1.unsqueeze(0).to(device)
        img2 = img2.unsqueeze(0).to(device)

        # Forward pass
        with torch.no_grad():
            embedding1, embedding2, distance = model(img1, img2)

        # Get prediction
        prediction = 1 if distance.item() < threshold else 0

        # Visualize
        output_path = os.path.join(output_dir, f"{prefix}pair_{i+1}.png")

        # Load original images for display
        img1_display = Image.open(img1_path).convert("RGB")
        img2_display = Image.open(img2_path).convert("RGB")

        visualize_pair(
            img1_display,
            img2_display,
            distance=distance.item(),
            label=label,
            prediction=prediction,
            title=f"Pair {i+1}",
            output_path=output_path,
        )

        # Print result
        print(
            f"Pair {i+1}: Distance = {distance.item():.4f}, "
            f"Label = {label}, Prediction = {prediction}"
        )


def visualize_pairs_grid(
    model: nn.Module,
    img_pairs: List[Tuple[str, str, int]],
    output_path: str,
    threshold: float = 0.5,
    transform: Optional[Callable] = None,
    nrows: int = 3,
    ncols: int = 3,
    figsize: Tuple[int, int] = (15, 15),
    device: Optional[torch.device] = None,
    dpi: int = 300,
) -> Figure:
    """
    Visualize a grid of image pairs with predictions from a Siamese model.

    Args:
        model: Siamese model
        img_pairs: List of (img1_path, img2_path, label) tuples
        output_path: Path to save the visualization
        threshold: Threshold for prediction (distance < threshold -> same class)
        transform: Transform to apply to images (if None, use a default transform)
        nrows: Number of rows in the grid
        ncols: Number of columns in the grid
        figsize: Figure size
        device: Device to run inference on
        dpi: DPI for saving the figure

    Returns:
        Matplotlib figure
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set model to evaluation mode
    model.eval()

    # Create default transform if not provided
    if transform is None:
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    # Limit number of pairs
    max_pairs = nrows * ncols
    if len(img_pairs) > max_pairs:
        img_pairs = random.sample(img_pairs, max_pairs)

    # Create figure
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    # Process each pair
    for i, (img1_path, img2_path, label) in enumerate(img_pairs):
        if i >= len(axes):
            break

        # Load and transform images
        img1, img2 = load_image_pair(img1_path, img2_path, transform)

        # Move to device
        img1_tensor = img1.unsqueeze(0).to(device)
        img2_tensor = img2.unsqueeze(0).to(device)

        # Forward pass
        with torch.no_grad():
            embedding1, embedding2, distance = model(img1_tensor, img2_tensor)

        # Get prediction
        prediction = 1 if distance.item() < threshold else 0

        # Load original images for display
        img1_display = Image.open(img1_path).convert("RGB")
        img2_display = Image.open(img2_path).convert("RGB")

        # Create a combined image
        combined_img = np.hstack([np.array(img1_display), np.array(img2_display)])

        # Display combined image
        axes[i].imshow(combined_img)

        # Add text overlay
        result_color = "green" if prediction == label else "red"
        result_text = f"Dist: {distance.item():.2f}, Pred: {'Same' if prediction==1 else 'Diff'}, GT: {'Same' if label==1 else 'Diff'}"

        axes[i].text(
            0.5,
            0.95,
            result_text,
            horizontalalignment="center",
            verticalalignment="top",
            transform=axes[i].transAxes,
            color=result_color,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.7),
        )

        # Remove axis ticks
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    # Hide any unused subplots
    for i in range(len(img_pairs), len(axes)):
        axes[i].axis("off")

    # Add title
    plt.suptitle("Siamese Network Predictions", fontsize=16)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi)

    return fig


def create_pair_dataset_from_directory(
    data_dir: str,
    split: str = "test",
    num_same_pairs: int = 10,
    num_diff_pairs: int = 10,
) -> List[Tuple[str, str, int]]:
    """
    Create a dataset of image pairs from a directory.

    Args:
        data_dir: Directory containing the image data
        split: Dataset split ('train', 'val', or 'test')
        num_same_pairs: Number of pairs with same class
        num_diff_pairs: Number of pairs with different class

    Returns:
        List of (img1_path, img2_path, label) tuples
    """
    # Get the split directory
    split_dir = os.path.join(data_dir, split)
    if not os.path.exists(split_dir):
        raise ValueError(f"Split directory {split_dir} does not exist")

    # Get all classes
    class_dirs = [
        d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))
    ]

    # Create pairs
    pairs = []

    # Create same-class pairs
    for _ in range(num_same_pairs):
        # Choose a random class
        class_dir = random.choice(class_dirs)
        class_path = os.path.join(split_dir, class_dir)

        # Get all images in the class
        images = [
            f
            for f in os.listdir(class_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        # Skip if not enough images
        if len(images) < 2:
            continue

        # Choose two random images
        img1, img2 = random.sample(images, 2)

        # Add to pairs
        pairs.append(
            (
                os.path.join(class_path, img1),
                os.path.join(class_path, img2),
                1,  # Same class
            )
        )

    # Create different-class pairs
    for _ in range(num_diff_pairs):
        # Choose two random classes
        class1, class2 = random.sample(class_dirs, 2)
        class1_path = os.path.join(split_dir, class1)
        class2_path = os.path.join(split_dir, class2)

        # Get images from each class
        images1 = [
            f
            for f in os.listdir(class1_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        images2 = [
            f
            for f in os.listdir(class2_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        # Skip if either class has no images
        if not images1 or not images2:
            continue

        # Choose a random image from each class
        img1 = random.choice(images1)
        img2 = random.choice(images2)

        # Add to pairs
        pairs.append(
            (
                os.path.join(class1_path, img1),
                os.path.join(class2_path, img2),
                0,  # Different class
            )
        )

    return pairs


if __name__ == "__main__":
    # Example usage

    # Create a dummy model (for testing without the actual model)
    class DummySiameseModel(nn.Module):
        def forward(self, x1, x2):
            # Generate a random distance
            distance = torch.rand(x1.size(0))
            return x1, x2, distance

    # Create a dummy dataset
    img_pairs = [
        ("path/to/img1.jpg", "path/to/img2.jpg", 1),  # Same class
        ("path/to/img1.jpg", "path/to/img3.jpg", 0),  # Different class
    ]

    # Create an output directory
    output_dir = "pair_visualization_examples"
    os.makedirs(output_dir, exist_ok=True)

    # Visualize pairs
    model = DummySiameseModel()

    # Note: This example won't actually work without real image paths
    print(
        "This is an example. To use this module, provide real image paths and a trained model."
    )
