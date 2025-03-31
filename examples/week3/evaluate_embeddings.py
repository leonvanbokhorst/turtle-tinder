#!/usr/bin/env python
"""
Utility for evaluating embeddings and re-identification performance.

This script evaluates learned embeddings from different models (Siamese, Triplet, ArcFace)
using appropriate metrics for re-identification tasks.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.manifold import TSNE
from typing import Tuple, Dict, List, Optional, Union, Any

# Try to import the model implementations
try:
    from siamese_network import SiameseNetwork
    from triplet_network import TripletNetwork
    from arcface_model import ArcFaceModel
except ImportError:
    print(
        "Warning: Could not import model implementations. Make sure they are in the same directory."
    )


class TurtleDataset(Dataset):
    """
    Dataset for turtle re-identification evaluation.
    """

    def __init__(self, data_dir: str, split: str = "test", img_size: int = 224):
        """
        Initialize the dataset.

        Args:
            data_dir: Directory containing the image data
            split: Dataset split ('train', 'val', or 'test')
            img_size: Size of the input images
        """
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size

        # Set up transforms
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Load the data
        self.image_paths, self.labels, self.class_to_idx = self._load_dataset(
            data_dir, split
        )

    def _load_dataset(
        self, data_dir: str, split: str
    ) -> Tuple[List[str], List[int], Dict[str, int]]:
        """
        Load dataset from directory.

        Args:
            data_dir: Directory containing the image data
            split: Dataset split ('train', 'val', or 'test')

        Returns:
            Tuple of (image_paths, labels, class_to_idx)
        """
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            raise ValueError(f"Split directory {split_dir} does not exist")

        image_paths = []
        labels = []
        class_to_idx = {}

        # Each subdirectory is a class
        class_dirs = sorted(
            [
                d
                for d in os.listdir(split_dir)
                if os.path.isdir(os.path.join(split_dir, d))
            ]
        )

        for idx, class_dir in enumerate(class_dirs):
            class_to_idx[class_dir] = idx
            class_path = os.path.join(split_dir, class_dir)
            class_images = [
                os.path.join(class_path, f)
                for f in os.listdir(class_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]

            image_paths.extend(class_images)
            labels.extend([idx] * len(class_images))

        return image_paths, labels, class_to_idx

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Get an image, label, and image path.

        Args:
            idx: Index of the image

        Returns:
            Tuple of (image, label, image_path)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        img = Image.open(img_path).convert("RGB")

        # Apply transforms
        img = self.transform(img)

        return img, label, img_path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate embeddings for sea turtle re-identification"
    )

    # Data parameters
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to dataset directory"
    )
    parser.add_argument("--img_size", type=int, default=224, help="Input image size")

    # Model parameters
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["siamese", "triplet", "arcface"],
        default="arcface",
        help="Type of model to evaluate",
    )
    parser.add_argument(
        "--backbone", type=str, default="resnet50", help="Backbone model architecture"
    )
    parser.add_argument(
        "--embedding_dim", type=int, default=512, help="Embedding dimension"
    )

    # Evaluation parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--gallery_split",
        type=str,
        default="train",
        help="Dataset split to use as gallery",
    )
    parser.add_argument(
        "--query_split", type=str, default="test", help="Dataset split to use as query"
    )

    # Output parameters
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Output directory for evaluation results",
    )

    return parser.parse_args()


def load_model(
    model_path: str,
    model_type: str,
    backbone: str,
    embedding_dim: int,
    num_classes: int,
) -> nn.Module:
    """
    Load a trained model from a checkpoint.

    Args:
        model_path: Path to the model checkpoint
        model_type: Type of model ('siamese', 'triplet', or 'arcface')
        backbone: Backbone model architecture
        embedding_dim: Embedding dimension
        num_classes: Number of classes (for ArcFace)

    Returns:
        Loaded model
    """
    if model_type == "siamese":
        model = SiameseNetwork(backbone=backbone, embedding_dim=embedding_dim)
    elif model_type == "triplet":
        model = TripletNetwork(backbone=backbone, embedding_dim=embedding_dim)
    elif model_type == "arcface":
        model = ArcFaceModel(
            backbone=backbone, embedding_dim=embedding_dim, num_classes=num_classes
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load weights
    checkpoint = torch.load(model_path, map_location="cpu")

    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    return model


def extract_embeddings(
    model: nn.Module, data_loader: DataLoader, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Extract embeddings from a dataset.

    Args:
        model: The embedding model
        data_loader: DataLoader for the dataset
        device: Device to run inference on

    Returns:
        Tuple of (embeddings, labels, image_paths)
    """
    model.eval()

    all_embeddings = []
    all_labels = []
    all_paths = []

    with torch.no_grad():
        for images, labels, paths in tqdm(data_loader, desc="Extracting embeddings"):
            images = images.to(device)

            # Get embeddings
            if hasattr(model, "forward_one"):  # Siamese
                embeddings = model.forward_one(images)
            else:  # Triplet or ArcFace
                embeddings = model(images)

            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)
            all_paths.extend(paths)

    # Concatenate all embeddings and labels
    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)

    return embeddings, labels, all_paths


def evaluate_reid_performance(
    query_embeddings: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    gallery_labels: torch.Tensor,
) -> Dict[str, float]:
    """
    Evaluate re-identification performance.

    Args:
        query_embeddings: Embeddings for query images
        query_labels: Labels for query images
        gallery_embeddings: Embeddings for gallery images
        gallery_labels: Labels for gallery images

    Returns:
        Dictionary with evaluation metrics
    """
    # Calculate cosine similarity between query and gallery
    similarity = torch.mm(query_embeddings, gallery_embeddings.t())

    # For each query, rank gallery by similarity
    _, indices = similarity.sort(dim=1, descending=True)

    # Calculate metrics
    metrics = calculate_reid_metrics(
        indices=indices, query_labels=query_labels, gallery_labels=gallery_labels
    )

    return metrics


def calculate_reid_metrics(
    indices: torch.Tensor, query_labels: torch.Tensor, gallery_labels: torch.Tensor
) -> Dict[str, float]:
    """
    Calculate re-identification metrics.

    Args:
        indices: Ranked indices for each query
        query_labels: Labels for query images
        gallery_labels: Labels for gallery images

    Returns:
        Dictionary with metrics
    """
    # Initialize metrics
    rank1 = 0
    rank5 = 0
    rank10 = 0
    ap_sum = 0

    # Get the number of queries
    num_queries = len(query_labels)

    # For each query
    for i in range(num_queries):
        query_label = query_labels[i]

        # Get gallery labels in ranked order for this query
        ranked_labels = gallery_labels[indices[i]]

        # Find positions where gallery label matches query label
        matches = ranked_labels == query_label

        # Compute cmc metrics
        if matches[0]:
            rank1 += 1
        if torch.any(matches[:5]):
            rank5 += 1
        if torch.any(matches[:10]):
            rank10 += 1

        # Compute average precision
        positive_indices = torch.where(matches)[0]

        if len(positive_indices) > 0:
            # Convert to 1-indexed for precision calculation
            positive_positions = positive_indices + 1

            # Calculate precision at each relevant rank
            precisions = (
                torch.arange(1, len(positive_positions) + 1).float()
                / positive_positions.float()
            )

            # Calculate average precision
            ap = torch.mean(precisions)
            ap_sum += ap.item()

    # Normalize metrics
    metrics = {
        "rank1": rank1 / num_queries,
        "rank5": rank5 / num_queries,
        "rank10": rank10 / num_queries,
        "mAP": ap_sum / num_queries,
    }

    return metrics


def visualize_embeddings(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    output_dir: str,
    n_samples: int = 1000,
    perplexity: int = 30,
    method: str = "tsne",
) -> Figure:
    """
    Visualize embeddings using dimensionality reduction.

    Args:
        embeddings: Embeddings to visualize
        labels: Labels for color coding
        output_dir: Output directory for saving the visualization
        n_samples: Number of samples to visualize (TSNE can be slow)
        perplexity: Perplexity parameter for TSNE
        method: Dimensionality reduction method ('tsne' or 'pca')

    Returns:
        Matplotlib figure
    """
    # Convert to numpy
    embeddings = embeddings.numpy()
    labels = labels.numpy()

    # Subsample if needed
    if n_samples < len(embeddings):
        indices = np.random.choice(len(embeddings), n_samples, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]

    # Apply dimensionality reduction
    if method == "tsne":
        print("Applying t-SNE dimensionality reduction...")
        embeddings_2d = TSNE(
            n_components=2, perplexity=perplexity, n_iter=1000, verbose=1
        ).fit_transform(embeddings)
    else:  # PCA
        print("Applying PCA dimensionality reduction...")
        from sklearn.decomposition import PCA

        embeddings_2d = PCA(n_components=2).fit_transform(embeddings)

    # Create visualization
    plt.figure(figsize=(12, 10))

    # Get unique labels
    unique_labels = np.unique(labels)

    # Create a colormap
    cmap = plt.cm.jet

    # Plot each class with a different color
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[cmap(i / len(unique_labels))],
            label=f"Class {label}",
            alpha=0.7,
            s=30,
        )

    # Add legend with smaller point sizes for readability
    if len(unique_labels) <= 20:  # Only show legend if not too many classes
        plt.legend(markerscale=2, fontsize="small")

    plt.title(f"Embedding Visualization with {method.upper()}")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(alpha=0.3)

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"embedding_visualization_{method}.png"))

    return plt.gcf()


def visualize_retrievals(
    query_embeddings: torch.Tensor,
    query_labels: torch.Tensor,
    query_paths: List[str],
    gallery_embeddings: torch.Tensor,
    gallery_labels: torch.Tensor,
    gallery_paths: List[str],
    output_dir: str,
    n_queries: int = 5,
    n_results: int = 5,
):
    """
    Visualize retrieval results for some query images.

    Args:
        query_embeddings: Embeddings for query images
        query_labels: Labels for query images
        query_paths: Paths to query images
        gallery_embeddings: Embeddings for gallery images
        gallery_labels: Labels for gallery images
        gallery_paths: Paths to gallery images
        output_dir: Output directory for saving the visualization
        n_queries: Number of query images to visualize
        n_results: Number of top retrieval results to show
    """
    # Calculate similarity between queries and gallery
    similarity = torch.mm(query_embeddings, gallery_embeddings.t())

    # For each query, rank gallery by similarity
    _, indices = similarity.sort(dim=1, descending=True)

    # Select a subset of queries to visualize
    query_indices = torch.randperm(len(query_labels))[:n_queries]

    # Create a figure for each query
    for i, query_idx in enumerate(query_indices):
        query_label = query_labels[query_idx]
        query_path = query_paths[query_idx]

        # Get top gallery indices for this query
        top_gallery_indices = indices[query_idx][:n_results]

        # Get labels and paths for top gallery images
        top_gallery_labels = gallery_labels[top_gallery_indices]
        top_gallery_paths = [gallery_paths[j] for j in top_gallery_indices]

        # Load images
        query_img = Image.open(query_path).convert("RGB")
        gallery_imgs = [Image.open(path).convert("RGB") for path in top_gallery_paths]

        # Create visualization
        fig, axes = plt.subplots(1, n_results + 1, figsize=(5 * (n_results + 1), 5))

        # Display query image
        axes[0].imshow(query_img)
        axes[0].set_title(f"Query (Class {query_label})")
        axes[0].axis("off")

        # Display gallery images
        for j, (img, label) in enumerate(zip(gallery_imgs, top_gallery_labels)):
            axes[j + 1].imshow(img)

            # Color title based on whether the class matches the query
            color = "green" if label == query_label else "red"
            axes[j + 1].set_title(f"Rank {j + 1} (Class {label})", color=color)
            axes[j + 1].axis("off")

        # Save the figure
        os.makedirs(os.path.join(output_dir, "retrievals"), exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "retrievals", f"query_{i + 1}.png"))
        plt.close()


def main():
    """Main function for evaluating embeddings."""
    # Parse arguments
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create datasets
    gallery_dataset = TurtleDataset(
        data_dir=args.data_dir, split=args.gallery_split, img_size=args.img_size
    )

    query_dataset = TurtleDataset(
        data_dir=args.data_dir, split=args.query_split, img_size=args.img_size
    )

    # Create dataloaders
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    query_loader = DataLoader(
        query_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Print dataset sizes
    print(f"Gallery set size: {len(gallery_dataset)}")
    print(f"Query set size: {len(query_dataset)}")

    # Load model
    model = load_model(
        model_path=args.model_path,
        model_type=args.model_type,
        backbone=args.backbone,
        embedding_dim=args.embedding_dim,
        num_classes=len(gallery_dataset.class_to_idx),
    ).to(device)

    # Extract embeddings
    print("Extracting gallery embeddings...")
    gallery_embeddings, gallery_labels, gallery_paths = extract_embeddings(
        model, gallery_loader, device
    )

    print("Extracting query embeddings...")
    query_embeddings, query_labels, query_paths = extract_embeddings(
        model, query_loader, device
    )

    # Evaluate re-identification performance
    print("Evaluating re-identification performance...")
    metrics = evaluate_reid_performance(
        query_embeddings, query_labels, gallery_embeddings, gallery_labels
    )

    # Print metrics
    print(f"Rank-1 Accuracy: {metrics['rank1']:.4f}")
    print(f"Rank-5 Accuracy: {metrics['rank5']:.4f}")
    print(f"Rank-10 Accuracy: {metrics['rank10']:.4f}")
    print(f"mAP: {metrics['mAP']:.4f}")

    # Save metrics
    with open(os.path.join(args.output_dir, "metrics.txt"), "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")

    # Visualize embeddings
    print("Visualizing embeddings...")
    visualize_embeddings(query_embeddings, query_labels, args.output_dir, method="tsne")

    # Visualize retrievals
    print("Visualizing retrieval results...")
    visualize_retrievals(
        query_embeddings,
        query_labels,
        query_paths,
        gallery_embeddings,
        gallery_labels,
        gallery_paths,
        args.output_dir,
    )

    print(f"Evaluation completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
