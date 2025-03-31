#!/usr/bin/env python3
# Sea Turtle Re-Identification - Week 2
# Transfer Learning with Frozen Features

import os
import argparse
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import sys

# Add the parent directory to the path to import from week1
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from week1.utils.data_loader import TurtleDataset, create_underwater_transforms


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a model with transfer learning for turtle identification"
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
        "--model",
        type=str,
        default="resnet50",
        choices=[
            "resnet18",
            "resnet50",
            "resnet101",
            "efficientnet_b0",
            "efficientnet_b3",
            "mobilenet_v2",
        ],
        help="Pre-trained model architecture to use",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of epochs to train for"
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
        "--freeze_backbone",
        action="store_true",
        help="Whether to freeze the backbone and only train the new layers",
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
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker threads for data loading",
    )
    return parser.parse_args()


def create_model(model_name, num_classes, freeze_backbone=True):
    """
    Create a pre-trained model with a new classification head.

    Args:
        model_name: Name of the pre-trained model architecture
        num_classes: Number of classes to classify
        freeze_backbone: Whether to freeze the backbone weights

    Returns:
        The model with a new classification head
    """
    model = None

    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

    elif model_name == "resnet50":
        model = models.resnet50(pretrained=True)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

    elif model_name == "resnet101":
        model = models.resnet101(pretrained=True)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=True)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.2, inplace=True), nn.Linear(num_features, num_classes)
        )

    elif model_name == "efficientnet_b3":
        model = models.efficientnet_b3(pretrained=True)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3, inplace=True), nn.Linear(num_features, num_classes)
        )

    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=True)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(num_features, num_classes)
        )

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


def create_dataloaders(data_dir, batch_size=32, num_workers=4):
    """
    Create data loaders for training, validation, and testing.

    Args:
        data_dir: Directory containing train, val, and test subdirectories
        batch_size: Batch size for training
        num_workers: Number of worker threads for data loading

    Returns:
        train_loader, val_loader, test_loader, and the number of classes
    """
    # Create transforms
    train_transform = create_underwater_transforms(train=True)
    eval_transform = create_underwater_transforms(train=False)

    # Create datasets
    train_dataset = TurtleDataset(
        root_dir=os.path.join(data_dir, "train"), transform=train_transform
    )

    val_dataset = TurtleDataset(
        root_dir=os.path.join(data_dir, "val"), transform=eval_transform
    )

    test_dataset = TurtleDataset(
        root_dir=os.path.join(data_dir, "test"), transform=eval_transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    num_classes = len(train_dataset.classes)

    return train_loader, val_loader, test_loader, num_classes, train_dataset.classes


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


def test(model, test_loader, device):
    """
    Test the model on the test set.

    Args:
        model: The neural network model
        test_loader: DataLoader for test data
        device: Device to run testing on

    Returns:
        Test accuracy and predictions
    """
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Statistics
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Save predictions and labels
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = correct / total

    return test_acc, all_predictions, all_labels


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


def count_trainable_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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

    # Create data loaders
    train_loader, val_loader, test_loader, num_classes, class_names = (
        create_dataloaders(args.data_dir, args.batch_size, args.workers)
    )

    print(f"Dataset contains {num_classes} turtle classes")
    print(f"Training set: {len(train_loader.dataset)} images")
    print(f"Validation set: {len(val_loader.dataset)} images")
    print(f"Test set: {len(test_loader.dataset)} images")

    # Create the model
    model = create_model(args.model, num_classes, args.freeze_backbone)
    model.to(device)

    # Count trainable parameters
    trainable_params = count_trainable_parameters(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model}")
    print(f"Total parameters: {total_params:,}")
    print(
        f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})"
    )

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    # Only optimize trainable parameters
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    # Initialize lists to track metrics
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_acc = 0.0

    # Training loop
    print(
        f"\nStarting training with {'frozen' if args.freeze_backbone else 'unfrozen'} backbone..."
    )
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

    # Load best model for evaluation
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pth")))

    # Evaluate on test set
    test_acc, predictions, true_labels = test(model, test_loader, device)
    print(f"\nTest accuracy: {test_acc:.4f}")

    # Save model metadata and results
    metadata = {
        "model_name": args.model,
        "num_classes": num_classes,
        "freeze_backbone": args.freeze_backbone,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "training_time": str(datetime.timedelta(seconds=int(training_time))),
        "class_names": class_names,
    }

    # Save metadata as JSON
    import json

    with open(os.path.join(args.output_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    # Create confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns

    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "confusion_matrix.png"))

    # Generate classification report
    report = classification_report(true_labels, predictions, target_names=class_names)
    print("\nClassification Report:")
    print(report)

    with open(os.path.join(args.output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    print(f"\nAll results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
