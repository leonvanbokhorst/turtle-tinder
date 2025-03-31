#!/usr/bin/env python3
# Sea Turtle Re-Identification - Week 1
# Training script for the baseline CNN model

import os
import argparse
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision

# Import our custom modules
from baseline_cnn import BaselineCNN

# If we had utilities to import
# from utils.data_loader import AlbumentationsDataset


class AlbumentationsDataset(torch.utils.data.Dataset):
    """Custom Dataset class for using Albumentations transforms with PyTorch"""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # List all class folders (turtle IDs)
        self.classes = sorted(
            [
                d
                for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            ]
        )

        # Map class names to indices
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # Collect all samples (image paths and their classes)
        self.samples = []
        for cls in self.classes:
            class_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Read image
        from PIL import Image

        image = np.array(Image.open(img_path).convert("RGB"))

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a baseline CNN for turtle identification"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing the processed dataset with train/val/test splits",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models",
        help="Directory to save model checkpoints and results",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of epochs to train for"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Initial learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay (L2 regularization)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to use (cuda, cpu, or leave empty for auto)",
    )
    return parser.parse_args()


def create_transforms():
    """
    Create data transforms for training and validation.

    Training transforms include augmentations for underwater images.
    Validation transforms only include resize and normalization.
    """
    # Training transforms with augmentation
    train_transform = A.Compose(
        [
            # Geometric transformations
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5
            ),
            # Blur to simulate water effects
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=7, p=0.5),
                    A.GaussianBlur(blur_limit=7, p=0.5),
                    A.MedianBlur(blur_limit=5, p=0.5),
                ],
                p=0.7,
            ),
            # Color transformations to simulate underwater lighting
            A.OneOf(
                [
                    A.ColorJitter(
                        brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=0.5
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=10,
                        sat_shift_limit=30,
                        val_shift_limit=20,
                        p=0.5,
                    ),
                ],
                p=0.8,
            ),
            # Underwater color shifts (more blue/green)
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            # Add noise to simulate underwater particles
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            # Resize and normalize (ImageNet values) for neural networks
            A.Resize(height=224, width=224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    # Validation transforms (no augmentation)
    val_transform = A.Compose(
        [
            A.Resize(height=224, width=224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    return train_transform, val_transform


def create_dataloaders(data_dir, train_transform, val_transform, batch_size):
    """
    Create training and validation data loaders.

    Args:
        data_dir: Root directory containing train/val/test splits
        train_transform: Transforms for training data
        val_transform: Transforms for validation data
        batch_size: Batch size for training

    Returns:
        train_loader, val_loader, num_classes
    """
    # Create datasets
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_dataset = AlbumentationsDataset(train_dir, transform=train_transform)
    val_dataset = AlbumentationsDataset(val_dir, transform=val_transform)

    # Get number of classes
    num_classes = len(train_dataset.classes)
    print(f"Dataset contains {num_classes} different turtle IDs")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, num_classes


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.

    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer for updating weights
        device: Device to run training on

    Returns:
        Average training loss and accuracy for the epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """
    Validate the model on the validation set.

    Args:
        model: The neural network model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to run validation on

    Returns:
        Validation loss and accuracy
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Statistics
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = val_loss / len(val_loader.dataset)
    val_acc = correct / total

    return val_loss, val_acc


def save_metrics(train_losses, train_accs, val_losses, val_accs, output_dir):
    """Save training and validation metrics as plots"""
    plt.figure(figsize=(12, 5))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training and Validation Accuracy")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_metrics.png"))
    plt.close()

    # Also save as CSV
    import pandas as pd

    metrics = pd.DataFrame(
        {
            "epoch": list(range(1, len(train_losses) + 1)),
            "train_loss": train_losses,
            "train_acc": train_accs,
            "val_loss": val_losses,
            "val_acc": val_accs,
        }
    )
    metrics.to_csv(os.path.join(output_dir, "training_metrics.csv"), index=False)


def main():
    args = parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create transforms and data loaders
    train_transform, val_transform = create_transforms()
    train_loader, val_loader, num_classes = create_dataloaders(
        args.data_dir, train_transform, val_transform, args.batch_size
    )

    print(f"Training set: {len(train_loader.dataset)} images")
    print(f"Validation set: {len(val_loader.dataset)} images")

    # Create the model
    model = BaselineCNN(num_classes=num_classes)
    model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # Initialize lists to track metrics
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_acc = 0.0

    # Training loop
    print("Starting training...")
    start_time = time.time()

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step(val_loss)

        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                model.state_dict(), os.path.join(args.output_dir, "best_model.pth")
            )
            print(f"Saved new best model with validation accuracy: {val_acc:.4f}")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                },
                os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pth"),
            )

    # Training complete
    training_time = time.time() - start_time
    print(
        f"Training completed in {str(datetime.timedelta(seconds=int(training_time)))}"
    )

    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model.pth"))

    # Save training metrics
    save_metrics(train_losses, train_accs, val_losses, val_accs, args.output_dir)

    # Print best validation accuracy
    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
