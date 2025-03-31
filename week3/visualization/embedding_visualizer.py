"""
Embedding visualization utility for sea turtle re-identification.

This module provides utilities for visualizing embeddings using various
dimensionality reduction techniques (t-SNE, PCA, UMAP) and creating
visualizations of the embedding space.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Any
import seaborn as sns
import json
import cv2
from PIL import Image

# Try to import UMAP (optional dependency)
try:
    from umap import UMAP

    has_umap = True
except ImportError:
    has_umap = False


def reduce_dimensionality(
    embeddings: Union[torch.Tensor, np.ndarray],
    method: str = "tsne",
    n_components: int = 2,
    **kwargs: Any,
) -> np.ndarray:
    """
    Reduce the dimensionality of embeddings for visualization.

    Args:
        embeddings: Embeddings of shape [n_samples, embedding_dim]
        method: Dimensionality reduction method, one of ['tsne', 'pca', 'umap']
        n_components: Number of dimensions to reduce to
        **kwargs: Additional arguments for the dimensionality reduction method

    Returns:
        Reduced embeddings of shape [n_samples, n_components]
    """
    # Convert to numpy if needed
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()

    # Apply dimensionality reduction
    if method == "tsne":
        perplexity = kwargs.get("perplexity", min(30, embeddings.shape[0] // 5))

        # Clip perplexity to avoid TSNE errors with small datasets
        if perplexity < 5:
            perplexity = 5

        reduced = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            n_iter=1000,
            verbose=1,
            random_state=42,
        ).fit_transform(embeddings)
    elif method == "pca":
        reduced = PCA(n_components=n_components, random_state=42).fit_transform(
            embeddings
        )
    elif method == "umap" and has_umap:
        n_neighbors = kwargs.get("n_neighbors", min(15, embeddings.shape[0] // 5))
        min_dist = kwargs.get("min_dist", 0.1)

        reduced = UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42,
        ).fit_transform(embeddings)
    else:
        if method == "umap" and not has_umap:
            print("Warning: UMAP not available. Falling back to t-SNE.")

        # Default to t-SNE
        reduced = TSNE(
            n_components=n_components,
            perplexity=min(30, embeddings.shape[0] // 5),
            n_iter=1000,
            verbose=1,
            random_state=42,
        ).fit_transform(embeddings)

    return reduced


def plot_embeddings(
    embeddings_2d: np.ndarray,
    labels: Union[torch.Tensor, np.ndarray, List[int]],
    output_path: str,
    method: str = "tsne",
    label_names: Optional[Dict[int, str]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    alpha: float = 0.7,
    s: int = 30,
    dpi: int = 300,
) -> Figure:
    """
    Plot embeddings in 2D space.

    Args:
        embeddings_2d: 2D embeddings of shape [n_samples, 2]
        labels: Labels for color coding
        output_path: Path to save the plot
        method: Dimensionality reduction method used (for title)
        label_names: Optional dictionary mapping label indices to names
        title: Optional custom title
        figsize: Figure size
        alpha: Alpha value for points
        s: Size of points
        dpi: DPI for saving the figure

    Returns:
        Matplotlib figure
    """
    # Convert to numpy if needed
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    elif isinstance(labels, list):
        labels = np.array(labels)

    # Create figure
    plt.figure(figsize=figsize)

    # Get unique labels
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)

    # Create a colormap
    if num_classes <= 10:
        # Use a standard colormap for a small number of classes
        cmap = plt.cm.tab10
    elif num_classes <= 20:
        # Use a colormap with more colors
        cmap = plt.cm.tab20
    else:
        # For many classes, create a custom colormap with sufficient colors
        cmap = plt.cm.jet

    # Plot each class with a different color
    for i, label in enumerate(unique_labels):
        mask = labels == label

        # Get label name if provided
        if label_names is not None and label in label_names:
            label_name = label_names[label]
        else:
            label_name = f"Class {label}"

        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[cmap(i / num_classes)],
            label=label_name,
            alpha=alpha,
            s=s,
        )

    # Add title
    if title is None:
        title = f"Embedding Visualization with {method.upper()}"
    plt.title(title)

    # Add axis labels
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")

    # Add grid
    plt.grid(alpha=0.3)

    # Add legend if there are not too many classes
    if num_classes <= 25:
        # Use a smaller marker size in the legend for readability
        plt.legend(markerscale=2, fontsize="small")

    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)

    return plt.gcf()


def plot_embedding_clusters(
    embeddings_2d: np.ndarray,
    labels: Union[torch.Tensor, np.ndarray, List[int]],
    output_path: str,
    method: str = "tsne",
    label_names: Optional[Dict[int, str]] = None,
    plot_centroids: bool = True,
    plot_ellipses: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    alpha: float = 0.7,
    s: int = 30,
    dpi: int = 300,
) -> Figure:
    """
    Plot embeddings in 2D space with additional cluster information (centroids, ellipses).

    Args:
        embeddings_2d: 2D embeddings of shape [n_samples, 2]
        labels: Labels for color coding
        output_path: Path to save the plot
        method: Dimensionality reduction method used (for title)
        label_names: Optional dictionary mapping label indices to names
        plot_centroids: Whether to plot cluster centroids
        plot_ellipses: Whether to plot confidence ellipses
        title: Optional custom title
        figsize: Figure size
        alpha: Alpha value for points
        s: Size of points
        dpi: DPI for saving the figure

    Returns:
        Matplotlib figure
    """
    # Convert to numpy if needed
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    elif isinstance(labels, list):
        labels = np.array(labels)

    # Create figure
    plt.figure(figsize=figsize)

    # Get unique labels
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)

    # Create a colormap
    if num_classes <= 10:
        # Use a standard colormap for a small number of classes
        cmap = plt.cm.tab10
    elif num_classes <= 20:
        # Use a colormap with more colors
        cmap = plt.cm.tab20
    else:
        # For many classes, create a custom colormap with sufficient colors
        cmap = plt.cm.jet

    # Plot each class with a different color
    for i, label in enumerate(unique_labels):
        mask = labels == label

        # Get label name if provided
        if label_names is not None and label in label_names:
            label_name = label_names[label]
        else:
            label_name = f"Class {label}"

        # Get class points
        points = embeddings_2d[mask]

        # Skip if not enough points
        if len(points) < 3:
            plt.scatter(
                points[:, 0],
                points[:, 1],
                c=[cmap(i / num_classes)],
                label=label_name,
                alpha=alpha,
                s=s,
            )
            continue

        # Plot points
        plt.scatter(
            points[:, 0],
            points[:, 1],
            c=[cmap(i / num_classes)],
            label=label_name,
            alpha=alpha,
            s=s,
        )

        # Plot centroid
        if plot_centroids:
            centroid = points.mean(axis=0)
            plt.scatter(
                centroid[0],
                centroid[1],
                c=[cmap(i / num_classes)],
                marker="X",
                s=s * 3,
                edgecolors="black",
            )

        # Plot confidence ellipse
        if plot_ellipses and len(points) >= 5:  # Need at least 5 points for covariance
            from matplotlib.patches import Ellipse
            from scipy.stats import chi2

            # Calculate covariance
            cov = np.cov(points, rowvar=False)

            # Get eigenvalues and eigenvectors
            evals, evecs = np.linalg.eigh(cov)

            # Sort eigenvalues in decreasing order
            idx = np.argsort(evals)[::-1]
            evals = evals[idx]
            evecs = evecs[:, idx]

            # Compute 95% confidence interval for chi-square distribution with 2 degrees of freedom
            chi2_val = chi2.ppf(0.95, 2)

            # Calculate ellipse width and height (semi-major and semi-minor axes)
            width, height = 2 * np.sqrt(chi2_val * evals)

            # Get rotation angle in degrees
            angle = np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0]))

            # Create and add ellipse
            ellipse = Ellipse(
                xy=centroid,
                width=width,
                height=height,
                angle=angle,
                edgecolor=cmap(i / num_classes),
                facecolor="none",
                linestyle="--",
                linewidth=2,
            )
            plt.gca().add_patch(ellipse)

    # Add title
    if title is None:
        title = f"Embedding Clusters with {method.upper()}"
    plt.title(title)

    # Add axis labels
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")

    # Add grid
    plt.grid(alpha=0.3)

    # Add legend if there are not too many classes
    if num_classes <= 25:
        # Use a smaller marker size in the legend for readability
        plt.legend(markerscale=2, fontsize="small")

    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)

    return plt.gcf()


def plot_similarity_matrix(
    embeddings: Union[torch.Tensor, np.ndarray],
    labels: Union[torch.Tensor, np.ndarray, List[int]],
    output_path: str,
    method: str = "cosine",
    label_names: Optional[Dict[int, str]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = "viridis",
    dpi: int = 300,
) -> Figure:
    """
    Plot a similarity matrix for the embeddings.

    Args:
        embeddings: Embeddings of shape [n_samples, embedding_dim]
        labels: Labels for sorting and annotating
        output_path: Path to save the plot
        method: Similarity method, one of ['cosine', 'euclidean', 'dot']
        label_names: Optional dictionary mapping label indices to names
        title: Optional custom title
        figsize: Figure size
        cmap: Colormap for the similarity matrix
        dpi: DPI for saving the figure

    Returns:
        Matplotlib figure
    """
    # Convert to numpy if needed
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    elif isinstance(labels, list):
        labels = np.array(labels)

    # Sort indices by label
    sorted_indices = np.argsort(labels)
    sorted_embeddings = embeddings[sorted_indices]
    sorted_labels = labels[sorted_indices]

    # Calculate similarity matrix
    if method == "cosine":
        # Normalize embeddings
        norms = np.linalg.norm(sorted_embeddings, axis=1, keepdims=True)
        normalized_embeddings = sorted_embeddings / norms
        similarity = np.dot(normalized_embeddings, normalized_embeddings.T)
    elif method == "euclidean":
        # Calculate Euclidean distance (converted to similarity)
        from sklearn.metrics.pairwise import euclidean_distances

        distances = euclidean_distances(sorted_embeddings)
        max_distance = np.max(distances)
        similarity = 1 - distances / max_distance
    elif method == "dot":
        # Dot product
        similarity = np.dot(sorted_embeddings, sorted_embeddings.T)
    else:
        raise ValueError(f"Unknown similarity method: {method}")

    # Create figure
    plt.figure(figsize=figsize)

    # Create similarity matrix plot
    im = plt.imshow(similarity, cmap=cmap)

    # Add colorbar
    plt.colorbar(im, label=f"{method.capitalize()} Similarity")

    # Add class boundaries if there are not too many samples
    if len(similarity) <= 100:
        # Find boundaries between classes
        boundaries = [0]
        for i in range(1, len(sorted_labels)):
            if sorted_labels[i] != sorted_labels[i - 1]:
                boundaries.append(i)
        boundaries.append(len(sorted_labels))

        # Add lines to show class boundaries
        for b in boundaries:
            plt.axhline(y=b - 0.5, color="red", linestyle="-")
            plt.axvline(x=b - 0.5, color="red", linestyle="-")

    # Add title
    if title is None:
        title = f"{method.capitalize()} Similarity Matrix"
    plt.title(title)

    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)

    return plt.gcf()


def create_embedding_visualization(
    embeddings: Union[torch.Tensor, np.ndarray],
    labels: Union[torch.Tensor, np.ndarray, List[int]],
    output_dir: str,
    label_names: Optional[Dict[int, str]] = None,
    sample_images: Optional[List[str]] = None,
    methods: List[str] = ["tsne", "pca"],
    prefix: str = "",
) -> None:
    """
    Create a comprehensive set of embedding visualizations.

    Args:
        embeddings: Embeddings of shape [n_samples, embedding_dim]
        labels: Labels for color coding
        output_dir: Directory to save the visualizations
        label_names: Optional dictionary mapping label indices to names
        sample_images: Optional list of image paths corresponding to each embedding
        methods: List of dimensionality reduction methods to use
        prefix: Optional prefix for output files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Convert to numpy if needed
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    elif isinstance(labels, list):
        labels = np.array(labels)

    # For each method, create visualizations
    for method in methods:
        print(f"Creating visualizations using {method.upper()}...")

        # Reduce dimensionality
        try:
            embeddings_2d = reduce_dimensionality(
                embeddings, method=method, n_components=2
            )

            # Basic embedding plot
            output_path = os.path.join(output_dir, f"{prefix}embeddings_{method}.png")
            plot_embeddings(
                embeddings_2d,
                labels,
                output_path,
                method=method,
                label_names=label_names,
            )

            # Cluster visualization
            output_path = os.path.join(output_dir, f"{prefix}clusters_{method}.png")
            plot_embedding_clusters(
                embeddings_2d,
                labels,
                output_path,
                method=method,
                label_names=label_names,
            )
        except Exception as e:
            print(f"Error creating {method.upper()} visualizations: {e}")

    # Create similarity matrix visualization
    try:
        output_path = os.path.join(output_dir, f"{prefix}similarity_matrix.png")
        plot_similarity_matrix(
            embeddings, labels, output_path, method="cosine", label_names=label_names
        )
    except Exception as e:
        print(f"Error creating similarity matrix: {e}")

    # If sample images are provided, create a visualization with actual images
    if sample_images is not None and len(sample_images) == len(embeddings):
        try:
            # Choose the first method for the image visualization
            method = methods[0]
            embeddings_2d = reduce_dimensionality(
                embeddings, method=method, n_components=2
            )

            # Create scatter plot with images
            output_path = os.path.join(
                output_dir, f"{prefix}image_scatter_{method}.png"
            )
            create_image_scatter_plot(
                embeddings_2d,
                sample_images,
                labels,
                output_path,
                method=method,
                label_names=label_names,
            )
        except Exception as e:
            print(f"Error creating image scatter plot: {e}")


def create_image_scatter_plot(
    embeddings_2d: np.ndarray,
    image_paths: List[str],
    labels: Union[torch.Tensor, np.ndarray, List[int]],
    output_path: str,
    method: str = "tsne",
    label_names: Optional[Dict[int, str]] = None,
    max_images: int = 100,
    figsize: Tuple[int, int] = (16, 14),
    dpi: int = 300,
    thumbnail_size: int = 50,
) -> Figure:
    """
    Create a scatter plot of embeddings with the actual images as thumbnails.

    Args:
        embeddings_2d: 2D embeddings of shape [n_samples, 2]
        image_paths: Paths to images corresponding to each embedding
        labels: Labels for color coding
        output_path: Path to save the plot
        method: Dimensionality reduction method used (for title)
        label_names: Optional dictionary mapping label indices to names
        max_images: Maximum number of images to show (subsamples if needed)
        figsize: Figure size
        dpi: DPI for saving the figure
        thumbnail_size: Size of thumbnail images in pixels

    Returns:
        Matplotlib figure
    """
    # Convert to numpy if needed
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    elif isinstance(labels, list):
        labels = np.array(labels)

    # Subsample if needed
    if len(embeddings_2d) > max_images:
        indices = np.random.choice(len(embeddings_2d), max_images, replace=False)
        embeddings_2d = embeddings_2d[indices]
        labels = labels[indices]
        image_paths = [image_paths[i] for i in indices]

    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    # Get unique labels
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)

    # Create a colormap for borders
    if num_classes <= 10:
        cmap = plt.cm.tab10
    elif num_classes <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.jet

    # Get the minimum and maximum x and y values
    min_x, max_x = np.min(embeddings_2d[:, 0]), np.max(embeddings_2d[:, 0])
    min_y, max_y = np.min(embeddings_2d[:, 1]), np.max(embeddings_2d[:, 1])

    # Add some margin
    margin_x = 0.05 * (max_x - min_x)
    margin_y = 0.05 * (max_y - min_y)
    min_x -= margin_x
    max_x += margin_x
    min_y -= margin_y
    max_y += margin_y

    # Set limits
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    # Calculate the size of a thumbnail in data units
    thumbnail_size_x = (max_x - min_x) * thumbnail_size / (figsize[0] * dpi)
    thumbnail_size_y = (max_y - min_y) * thumbnail_size / (figsize[1] * dpi)

    # Draw the thumbnails
    for i, (x, y) in enumerate(embeddings_2d):
        # Load image
        img = plt.imread(image_paths[i])

        # Get label
        label = labels[i]
        label_idx = np.where(unique_labels == label)[0][0]

        # Create a border color based on the label
        border_color = cmap(label_idx / num_classes)

        # Create inset axis for the image
        ins = ax.inset_axes(
            [
                x - thumbnail_size_x / 2,
                y - thumbnail_size_y / 2,
                thumbnail_size_x,
                thumbnail_size_y,
            ],
            transform=ax.transData,
        )

        # Display image in the inset axis
        ins.imshow(img)
        ins.axis("off")

        # Add a colored border
        for spine in ins.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2)

    # Add a legend for the colors
    for i, label_idx in enumerate(unique_labels):
        color = cmap(i / num_classes)

        # Get label name if provided
        if label_names is not None and label_idx in label_names:
            label_name = label_names[label_idx]
        else:
            label_name = f"Class {label_idx}"

        # Add a proxy artist for the legend
        ax.plot([], [], color=color, linewidth=5, label=label_name)

    ax.legend(loc="best")

    # Add title
    ax.set_title(f"Image Embedding Visualization with {method.upper()}")

    # Add axis labels
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")

    # Add grid
    ax.grid(alpha=0.3)

    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)

    return fig


if __name__ == "__main__":
    # Example usage

    # Create random embeddings
    n_samples = 100
    embedding_dim = 128
    n_classes = 5

    # Create embeddings with each class forming a cluster
    embeddings = np.random.randn(n_samples, embedding_dim)

    # Create labels: [0, 0, 0, ..., 1, 1, 1, ..., 2, 2, 2, ...]
    labels = np.array([i // (n_samples // n_classes) for i in range(n_samples)])

    # Create an output directory
    output_dir = "embedding_visualization_examples"
    os.makedirs(output_dir, exist_ok=True)

    # Create label names
    label_names = {i: f"Turtle {i+1}" for i in range(n_classes)}

    # Create visualizations
    create_embedding_visualization(
        embeddings,
        labels,
        output_dir,
        label_names=label_names,
        methods=["tsne", "pca"],
        prefix="example_",
    )

    print(f"Visualizations saved to {output_dir}")
