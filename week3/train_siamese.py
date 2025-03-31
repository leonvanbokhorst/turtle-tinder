#!/usr/bin/env python
"""
Training script for Siamese Network for sea turtle re-identification.
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import pandas as pd
from typing import Tuple, Dict, List, Optional

# Import Siamese model
from siamese_network import SiameseNetwork, contrastive_loss


class SiamesePairDataset(Dataset):
    """
    Dataset for training Siamese networks with image pairs.
    Each item contains a pair of images and a label (1 if same class, 0 if different).
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        img_size: int = 224,
        pairs_per_class: int = 10,
    ):
        """
        Initialize the dataset.

        Args:
            data_dir: Directory containing the image data
            split: Dataset split ('train', 'val', or 'test')
            img_size: Size of the input images
            pairs_per_class: Number of positive/negative pairs to generate per class
        """
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size
        self.pairs_per_class = pairs_per_class

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

        # If training, add augmentations
        if split == "train":
            self.transform = transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ColorJitter(
                        brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        # Load the data
        self.image_paths, self.labels = self._load_dataset(data_dir, split)

        # Group images by class
        self.class_to_images = self._group_by_class()

        # Generate pairs
        self.pairs = self._generate_pairs()

    def _load_dataset(self, data_dir: str, split: str) -> Tuple[List[str], List[int]]:
        """
        Load dataset from directory.

        Args:
            data_dir: Directory containing the image data
            split: Dataset split ('train', 'val', or 'test')

        Returns:
            Tuple of (image_paths, labels)
        """
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            raise ValueError(f"Split directory {split_dir} does not exist")

        image_paths = []
        labels = []

        # Each subdirectory is a class
        class_dirs = sorted(
            [
                d
                for d in os.listdir(split_dir)
                if os.path.isdir(os.path.join(split_dir, d))
            ]
        )

        for label, class_dir in enumerate(class_dirs):
            class_path = os.path.join(split_dir, class_dir)
            class_images = [
                os.path.join(class_path, f)
                for f in os.listdir(class_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]

            image_paths.extend(class_images)
            labels.extend([label] * len(class_images))

        return image_paths, labels

    def _group_by_class(self) -> Dict[int, List[int]]:
        """
        Group image indices by class.

        Returns:
            Dictionary mapping class labels to lists of image indices
        """
        class_to_images = {}
        for idx, label in enumerate(self.labels):
            if label not in class_to_images:
                class_to_images[label] = []
            class_to_images[label].append(idx)
        return class_to_images

    def _generate_pairs(self) -> List[Tuple[int, int, int]]:
        """
        Generate pairs of images for training.
        Each pair consists of (img1_idx, img2_idx, label) where label is 1 if same class, 0 if different.

        Returns:
            List of (img1_idx, img2_idx, label) tuples
        """
        pairs = []
        num_classes = len(self.class_to_images)

        # For each class
        for class_idx in self.class_to_images:
            class_images = self.class_to_images[class_idx]

            # Skip classes with only one image
            if len(class_images) < 2:
                continue

            # Generate positive pairs (same class)
            for _ in range(self.pairs_per_class):
                # Randomly select two images from the same class
                idx1, idx2 = random.sample(class_images, 2)
                pairs.append((idx1, idx2, 1))  # 1 = same class

            # Generate negative pairs (different classes)
            for _ in range(self.pairs_per_class):
                # Randomly select one image from this class
                idx1 = random.choice(class_images)

                # Randomly select a different class
                negative_class = random.choice(
                    [c for c in self.class_to_images.keys() if c != class_idx]
                )

                # Randomly select one image from the negative class
                idx2 = random.choice(self.class_to_images[negative_class])

                pairs.append((idx1, idx2, 0))  # 0 = different class

        random.shuffle(pairs)
        return pairs

    def __len__(self) -> int:
        """Return the number of image pairs in the dataset."""
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get an image pair and label.

        Args:
            idx: Index of the pair

        Returns:
            Tuple of (image1, image2, label)
        """
        img1_idx, img2_idx, label = self.pairs[idx]

        # Load images
        img1_path = self.image_paths[img1_idx]
        img2_path = self.image_paths[img2_idx]

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        # Apply transforms
        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, torch.tensor(label)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Siamese network for sea turtle re-identification"
    )

    # Data parameters
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to dataset directory"
    )
    parser.add_argument("--img_size", type=int, default=224, help="Input image size")
    parser.add_argument(
        "--pairs_per_class", type=int, default=20, help="Number of pairs per class"
    )

    # Model parameters
    parser.add_argument(
        "--backbone", type=str, default="resnet50", help="Backbone model architecture"
    )
    parser.add_argument(
        "--embedding_dim", type=int, default=128, help="Embedding dimension"
    )
    parser.add_argument(
        "--pretrained", action="store_true", help="Use pretrained weights"
    )
    parser.add_argument(
        "--margin", type=float, default=1.0, help="Margin for contrastive loss"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.5, help="Dropout probability"
    )

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of workers for data loading"
    )

    # Output parameters
    parser.add_argument(
        "--output_dir", type=str, default="./outputs", help="Output directory"
    )

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_dataloaders(args) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create dataloaders for training, validation, and testing."""
    train_dataset = SiamesePairDataset(
        data_dir=args.data_dir,
        split="train",
        img_size=args.img_size,
        pairs_per_class=args.pairs_per_class,
    )

    val_dataset = SiamesePairDataset(
        data_dir=args.data_dir,
        split="val",
        img_size=args.img_size,
        pairs_per_class=args.pairs_per_class // 2,  # Fewer pairs for validation
    )

    test_dataset = SiamesePairDataset(
        data_dir=args.data_dir,
        split="test",
        img_size=args.img_size,
        pairs_per_class=args.pairs_per_class // 2,  # Fewer pairs for testing
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Train the model for one epoch.

    Args:
        model: The model to train
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run training on

    Returns:
        Tuple of (average loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training")
    for img1, img2, labels in pbar:
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        embedding1, embedding2, distances = model(img1, img2)

        # Calculate loss
        loss = criterion(embedding1, embedding2, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Update metrics
        running_loss += loss.item() * img1.size(0)

        # Calculate accuracy (threshold distance at 0.5)
        predictions = (distances < 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        # Update progress bar
        pbar.set_postfix({"loss": loss.item(), "acc": correct / total})

    # Calculate epoch metrics
    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def validate(
    model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float]:
    """
    Validate the model.

    Args:
        model: The model to validate
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to run validation on

    Returns:
        Tuple of (average loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for img1, img2, labels in pbar:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            # Forward pass
            embedding1, embedding2, distances = model(img1, img2)

            # Calculate loss
            loss = criterion(embedding1, embedding2, labels)

            # Update metrics
            running_loss += loss.item() * img1.size(0)

            # Calculate accuracy (threshold distance at 0.5)
            predictions = (distances < 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # Update progress bar
            pbar.set_postfix({"loss": loss.item(), "acc": correct / total})

    # Calculate epoch metrics
    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def visualize_results(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: str,
    num_examples: int = 10,
):
    """
    Visualize some example pairs and model predictions.

    Args:
        model: The trained model
        test_loader: DataLoader for test data
        device: Device to run inference on
        output_dir: Output directory for saving visualizations
        num_examples: Number of examples to visualize
    """
    model.eval()

    # Get a batch of examples
    img1_list, img2_list, labels_list, distances_list = [], [], [], []

    with torch.no_grad():
        for img1, img2, labels in test_loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            # Forward pass
            _, _, distances = model(img1, img2)

            # Add to lists
            img1_list.append(img1.cpu())
            img2_list.append(img2.cpu())
            labels_list.append(labels.cpu())
            distances_list.append(distances.cpu())

            # Break if we have enough examples
            if sum(len(x) for x in img1_list) >= num_examples:
                break

    # Concatenate all examples
    img1_all = torch.cat(img1_list)
    img2_all = torch.cat(img2_list)
    labels_all = torch.cat(labels_list)
    distances_all = torch.cat(distances_list)

    # Select first num_examples
    img1_all = img1_all[:num_examples]
    img2_all = img2_all[:num_examples]
    labels_all = labels_all[:num_examples]
    distances_all = distances_all[:num_examples]

    # Denormalize images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    img1_all = img1_all * std + mean
    img2_all = img2_all * std + mean

    # Clamp to [0, 1]
    img1_all = torch.clamp(img1_all, 0, 1)
    img2_all = torch.clamp(img2_all, 0, 1)

    # Classify based on distance threshold
    predictions = (distances_all < 0.5).float()

    # Create visualization
    fig, axes = plt.subplots(num_examples, 3, figsize=(12, 2 * num_examples))

    for i in range(num_examples):
        # Display images
        axes[i, 0].imshow(img1_all[i].permute(1, 2, 0))
        axes[i, 1].imshow(img2_all[i].permute(1, 2, 0))

        # Create a blank placeholder for the third column
        axes[i, 2].axis("off")

        # Add information
        title = f"True: {'Same' if labels_all[i]==1 else 'Different'}, "
        title += f"Pred: {'Same' if predictions[i]==1 else 'Different'}, "
        title += f"Dist: {distances_all[i]:.2f}"

        # Color code based on correctness
        if predictions[i] == labels_all[i]:
            color = "green"
        else:
            color = "red"

        axes[i, 2].text(
            0.5, 0.5, title, ha="center", va="center", color=color, fontsize=10
        )

        # Remove axis ticks
        for j in range(3):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    # Add column titles
    axes[0, 0].set_title("Image 1")
    axes[0, 1].set_title("Image 2")
    axes[0, 2].set_title("Prediction")

    # Save figure
    plt.tight_layout()
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    plt.savefig(os.path.join(output_dir, "visualizations", "pair_predictions.png"))
    plt.close()


def plot_training_curve(train_losses, val_losses, train_accs, val_accs, output_dir):
    """
    Plot training and validation curves.

    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accs: List of training accuracies
        val_accs: List of validation accuracies
        output_dir: Output directory for saving plots
    """
    # Create output directory
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "plots", "loss_curve.png"))
    plt.close()

    # Plot accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(train_accs, label="Training Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "plots", "accuracy_curve.png"))
    plt.close()

    # Save data as CSV
    metrics_df = pd.DataFrame(
        {
            "epoch": list(range(1, len(train_losses) + 1)),
            "train_loss": train_losses,
            "val_loss": val_losses,
            "train_acc": train_accs,
            "val_acc": val_accs,
        }
    )
    metrics_df.to_csv(
        os.path.join(output_dir, "plots", "training_metrics.csv"), index=False
    )


def main():
    """Main function for training the Siamese network."""
    # Parse arguments
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"siamese_{args.backbone}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Save arguments
    with open(os.path.join(output_dir, "args.txt"), "w") as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    model = SiameseNetwork(
        backbone=args.backbone,
        embedding_dim=args.embedding_dim,
        pretrained=args.pretrained,
        dropout=args.dropout,
    ).to(device)

    # Create loss function
    criterion = lambda e1, e2, l: contrastive_loss(e1, e2, l, margin=args.margin)

    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(args)

    # Print dataset sizes
    print(f"Train set size: {len(train_loader.dataset)}")
    print(f"Validation set size: {len(val_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")

    # Training loop
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"New best validation accuracy: {best_val_acc:.4f}")

            # Save model
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            }
            torch.save(checkpoint, os.path.join(output_dir, "best_model.pth"))

    # Plot training curves
    plot_training_curve(train_losses, val_losses, train_accs, val_accs, output_dir)

    # Load best model
    checkpoint = torch.load(os.path.join(output_dir, "best_model.pth"))
    model.load_state_dict(checkpoint["model_state_dict"])

    # Evaluate on test set
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    # Save test metrics
    with open(os.path.join(output_dir, "test_metrics.txt"), "w") as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")

    # Visualize results
    visualize_results(model, test_loader, device, output_dir)

    print(f"Training completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
