"""
Data utilities for Week 4 examples.

This module provides functions for loading and preparing datasets for
ensemble methods and model evaluation.
"""

import os
import json
from typing import Tuple, List, Dict, Optional, Union, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image


class TurtleDataset(Dataset):
    """
    Dataset for sea turtle re-identification.
    """

    def __init__(
        self, data_dir: str, split: str = "test", transform: Optional[Any] = None
    ):
        """
        Initialize the dataset.

        Args:
            data_dir: Directory containing the dataset
            split: Dataset split ('train', 'val', 'test')
            transform: Transforms to apply to images
        """
        self.data_dir = data_dir
        self.split = split

        # Set up transforms
        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform = transform

        # Load the dataset
        self.samples, self.class_names = self._load_dataset()

    def _load_dataset(self) -> Tuple[List[Tuple[str, int]], List[str]]:
        """
        Load the dataset from disk.

        Returns:
            Tuple of (samples, class_names)
        """
        split_dir = os.path.join(self.data_dir, self.split)

        if not os.path.exists(split_dir):
            raise ValueError(f"Split directory not found: {split_dir}")

        # Get class directories
        class_dirs = sorted(
            [
                d
                for d in os.listdir(split_dir)
                if os.path.isdir(os.path.join(split_dir, d))
            ]
        )

        if not class_dirs:
            raise ValueError(f"No class directories found in {split_dir}")

        # Create class mapping
        class_to_idx = {class_name: i for i, class_name in enumerate(class_dirs)}

        # Collect samples
        samples = []
        for class_name in class_dirs:
            class_dir = os.path.join(split_dir, class_name)
            class_idx = class_to_idx[class_name]

            # Get all images
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(class_dir, img_name)
                    samples.append((img_path, class_idx))

        return samples, class_dirs

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image, label)
        """
        img_path, label = self.samples[idx]

        # Load image
        with open(img_path, "rb") as f:
            img = Image.open(f).convert("RGB")

        # Apply transforms
        if self.transform is not None:
            img = self.transform(img)

        return img, label


def load_dataset(
    data_dir: str, split: str = "test", transform: Optional[Any] = None
) -> TurtleDataset:
    """
    Load a dataset for evaluation.

    Args:
        data_dir: Directory containing the dataset
        split: Dataset split ('train', 'val', 'test')
        transform: Transforms to apply to images

    Returns:
        Dataset object
    """
    return TurtleDataset(data_dir, split, transform)


def prepare_dataloaders(
    data_dir: str,
    split: str = "test",
    batch_size: int = 32,
    num_workers: int = 4,
    transform: Optional[Any] = None,
) -> Tuple[DataLoader, List[str]]:
    """
    Prepare dataloaders for evaluation.

    Args:
        data_dir: Directory containing the dataset
        split: Dataset split ('train', 'val', 'test')
        batch_size: Batch size
        num_workers: Number of workers for data loading
        transform: Transforms to apply to images

    Returns:
        Tuple of (dataloader, class_names)
    """
    # Load dataset
    dataset = load_dataset(data_dir, split, transform)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader, dataset.class_names


def get_class_distribution(dataloader: DataLoader) -> Dict[int, int]:
    """
    Get the class distribution in a dataset.

    Args:
        dataloader: DataLoader object

    Returns:
        Dictionary mapping class index to count
    """
    class_counts = {}

    for _, labels in dataloader:
        for label in labels.numpy():
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1

    return class_counts


def visualize_samples(
    dataloader: DataLoader, class_names: List[str], num_samples: int = 5
) -> None:
    """
    Visualize samples from a dataset.

    Args:
        dataloader: DataLoader object
        class_names: List of class names
        num_samples: Number of samples to visualize per class
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Get samples per class
    samples_by_class = {}

    for images, labels in dataloader:
        for i, label in enumerate(labels.numpy()):
            if label not in samples_by_class:
                samples_by_class[label] = []

            if len(samples_by_class[label]) < num_samples:
                # Convert tensor to numpy
                img = images[i].numpy().transpose(1, 2, 0)

                # Denormalize
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)

                samples_by_class[label].append(img)

        # Check if we have enough samples
        if all(len(samples) >= num_samples for samples in samples_by_class.values()):
            break

    # Plot samples
    num_classes = len(class_names)
    fig, axes = plt.subplots(
        num_classes, num_samples, figsize=(num_samples * 2, num_classes * 2)
    )

    for i, class_idx in enumerate(sorted(samples_by_class.keys())):
        for j, img in enumerate(samples_by_class[class_idx]):
            if num_classes > 1:
                ax = axes[i, j]
            else:
                ax = axes[j]

            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])

            if j == 0:
                ax.set_ylabel(class_names[class_idx])

    plt.tight_layout()
    plt.show()


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """
    Save results to JSON or NPZ file.

    Args:
        results: Results dictionary
        output_path: Path to save results
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if output_path.endswith(".json"):
        # Convert numpy arrays to lists
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                results[key] = value.tolist()

        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)

    elif output_path.endswith(".npz"):
        np.savez(output_path, **results)

    else:
        raise ValueError(f"Unsupported file extension for {output_path}")


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Data utilities for Week 4")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Directory containing the dataset"
    )
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--visualize", action="store_true", help="Visualize dataset samples"
    )

    args = parser.parse_args()

    # Prepare dataloaders
    dataloader, class_names = prepare_dataloaders(
        data_dir=args.data_dir, split=args.split, batch_size=args.batch_size
    )

    print(f"Loaded {len(dataloader.dataset)} samples from {args.split} split")
    print(f"Found {len(class_names)} classes: {class_names}")

    # Get class distribution
    class_counts = get_class_distribution(dataloader)

    print("\nClass distribution:")
    for label, count in sorted(class_counts.items()):
        print(f"  {class_names[label]}: {count} samples")

    # Visualize samples if requested
    if args.visualize:
        visualize_samples(dataloader, class_names)
