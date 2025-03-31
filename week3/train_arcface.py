#!/usr/bin/env python
"""
Training script for ArcFace model for sea turtle re-identification.

This script implements training with the ArcFace loss function, which enhances
class separability by adding an angular margin penalty to the target logit.
"""

import os
import argparse
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Import local modules
from utils.data_loader import TurtleDataset, create_dataloaders
from arcface_model import ArcFaceModel
from utils.metrics import evaluate_embeddings, compute_embedding_statistics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train ArcFace model for turtle re-identification"
    )

    # Data parameters
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to dataset directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/arcface",
        help="Directory to save outputs",
    )

    # Model parameters
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        choices=[
            "resnet18",
            "resnet34",
            "resnet50",
            "efficientnet_b0",
            "efficientnet_b3",
        ],
        help="Backbone architecture",
    )
    parser.add_argument(
        "--embedding_dim", type=int, default=512, help="Embedding dimension"
    )
    parser.add_argument(
        "--pretrained", action="store_true", help="Use pretrained backbone"
    )

    # ArcFace parameters
    parser.add_argument("--margin", type=float, default=0.5, help="ArcFace margin")
    parser.add_argument(
        "--scale", type=float, default=30.0, help="ArcFace scale factor"
    )

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        choices=["step", "cosine", "plateau"],
        help="Learning rate scheduler",
    )
    parser.add_argument(
        "--early_stopping",
        type=int,
        default=10,
        help="Early stopping patience (0 to disable)",
    )

    # System parameters
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument(
        "--device", type=str, default="", help="Device to use (empty for auto)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def set_seed(seed):
    """Set all random seeds."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train model for one epoch."""
    model.train()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for images, labels in tqdm(train_loader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        logits, _, _ = model(images, labels)

        # Compute loss
        loss = criterion(logits, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Compute accuracy
        _, predicted = torch.max(logits.data, 1)
        total_samples += labels.size(0)
        running_correct += (predicted == labels).sum().item()
        running_loss += loss.item() * images.size(0)

    # Calculate metrics
    train_loss = running_loss / total_samples
    train_acc = running_correct / total_samples

    return train_loss, train_acc


def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    # Lists to store feature embeddings and labels for evaluation
    embeddings = []
    labels = []

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validation"):
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            logits, features, norm_features = model(images, targets)

            # Compute loss
            loss = criterion(logits, targets)

            # Compute accuracy
            _, predicted = torch.max(logits.data, 1)
            total_samples += targets.size(0)
            running_correct += (predicted == targets).sum().item()
            running_loss += loss.item() * images.size(0)

            # Store embeddings and labels for metric evaluation
            embeddings.append(norm_features.cpu())
            labels.append(targets.cpu())

    # Calculate metrics
    val_loss = running_loss / total_samples
    val_acc = running_correct / total_samples

    # Concatenate all embeddings and labels
    all_embeddings = torch.cat(embeddings, dim=0)
    all_labels = torch.cat(labels, dim=0)

    return val_loss, val_acc, all_embeddings, all_labels


def test(model, test_loader, device):
    """Test model and return embeddings and labels."""
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass (extract embeddings only)
            _, _, norm_features = model(images)

            # Store embeddings and labels
            embeddings.append(norm_features.cpu())
            labels.append(targets.cpu())

    # Concatenate all embeddings and labels
    all_embeddings = torch.cat(embeddings, dim=0)
    all_labels = torch.cat(labels, dim=0)

    return all_embeddings, all_labels


def create_scheduler(optimizer, args, num_training_steps):
    """Create learning rate scheduler."""
    if args.lr_scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=num_training_steps // 3, gamma=0.1
        )
    elif args.lr_scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_training_steps
        )
    elif args.lr_scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=5, verbose=True
        )
    else:
        raise ValueError(f"Unknown scheduler: {args.lr_scheduler}")

    return scheduler


def plot_training_curves(metrics, output_dir):
    """Plot and save training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot losses
    ax1.plot(metrics["train_losses"], label="Train Loss")
    ax1.plot(metrics["val_losses"], label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.set_title("Training and Validation Loss")
    ax1.grid(True)

    # Plot accuracies
    ax2.plot(metrics["train_accs"], label="Train Accuracy")
    ax2.plot(metrics["val_accs"], label="Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.set_title("Training and Validation Accuracy")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"))
    plt.close(fig)


def save_model(model, optimizer, metrics, args, epoch, output_dir, is_best=False):
    """Save model checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "args": vars(args),
    }

    # Save the latest checkpoint
    torch.save(checkpoint, os.path.join(output_dir, "latest_checkpoint.pth"))

    # Save the best checkpoint
    if is_best:
        torch.save(checkpoint, os.path.join(output_dir, "best_model.pth"))


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.backbone}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Save arguments
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir,
        args.batch_size,
        dataset_type="standard",
        num_workers=args.workers,
    )

    # Get number of classes from dataset
    num_classes = len(train_loader.dataset.base_dataset.class_to_idx)
    print(f"Number of classes: {num_classes}")

    # Create model
    model = ArcFaceModel(
        backbone=args.backbone,
        num_classes=num_classes,
        embedding_dim=args.embedding_dim,
        margin=args.margin,
        scale=args.scale,
        pretrained=args.pretrained,
    )
    model = model.to(device)

    # Print model summary
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {num_params:,} trainable parameters")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Create scheduler
    scheduler = create_scheduler(
        optimizer, args, num_training_steps=len(train_loader) * args.epochs
    )

    # Initialize training metrics
    metrics = {
        "train_losses": [],
        "train_accs": [],
        "val_losses": [],
        "val_accs": [],
        "best_val_loss": float("inf"),
        "best_val_acc": 0.0,
        "best_epoch": 0,
    }

    # Training loop
    patience_counter = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # Train for one epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc, val_embeddings, val_labels = validate(
            model, val_loader, criterion, device
        )

        # Update learning rate if using 'plateau' scheduler
        if args.lr_scheduler == "plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # Update metrics
        metrics["train_losses"].append(train_loss)
        metrics["train_accs"].append(train_acc)
        metrics["val_losses"].append(val_loss)
        metrics["val_accs"].append(val_acc)

        # Check if this is the best model
        is_best = val_acc > metrics["best_val_acc"]
        if is_best:
            metrics["best_val_acc"] = val_acc
            metrics["best_val_loss"] = val_loss
            metrics["best_epoch"] = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        # Save model
        save_model(model, optimizer, metrics, args, epoch, output_dir, is_best)

        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Early stopping
        if args.early_stopping > 0 and patience_counter >= args.early_stopping:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

        # Plot and save training curves
        plot_training_curves(metrics, output_dir)

    # Load best model for testing
    best_checkpoint = torch.load(os.path.join(output_dir, "best_model.pth"))
    model.load_state_dict(best_checkpoint["model_state_dict"])
    print(f"\nLoaded best model from epoch {best_checkpoint['epoch'] + 1}")

    # Test the model
    test_embeddings, test_labels = test(model, test_loader, device)

    # Split test set into query and gallery for evaluation
    # Randomly select 25% of samples per class for query
    unique_labels = torch.unique(test_labels)
    query_indices = []
    gallery_indices = []

    for label in unique_labels:
        label_indices = torch.where(test_labels == label)[0]
        n_query = max(1, int(0.25 * len(label_indices)))

        # Randomly select query indices
        perm = torch.randperm(len(label_indices))
        label_query = label_indices[perm[:n_query]]
        label_gallery = label_indices[perm[n_query:]]

        query_indices.append(label_query)
        gallery_indices.append(label_gallery)

    query_indices = torch.cat(query_indices)
    gallery_indices = torch.cat(gallery_indices)

    query_embeddings = test_embeddings[query_indices]
    query_labels = test_labels[query_indices]
    gallery_embeddings = test_embeddings[gallery_indices]
    gallery_labels = test_labels[gallery_indices]

    # Evaluate embeddings
    from utils.metrics import evaluate_embeddings

    metrics_results = evaluate_embeddings(
        query_embeddings, gallery_embeddings, query_labels, gallery_labels
    )

    # Compute embedding statistics
    embed_stats = compute_embedding_statistics(test_embeddings)

    # Merge all results and save
    results = {
        "train_time": time.time() - start_time,
        "best_epoch": metrics["best_epoch"],
        "best_val_acc": metrics["best_val_acc"],
        "best_val_loss": metrics["best_val_loss"],
        "final_epoch": len(metrics["train_losses"]) - 1,
        "embedding_stats": embed_stats,
        **metrics_results,
    }

    # Save evaluation results
    with open(os.path.join(output_dir, "evaluation_results.json"), "w") as f:
        # Convert numpy values to Python types for JSON serialization
        json_results = {}
        for k, v in results.items():
            if isinstance(v, np.ndarray):
                json_results[k] = v.tolist()
            elif isinstance(v, np.float32) or isinstance(v, np.float64):
                json_results[k] = float(v)
            elif isinstance(v, np.int64) or isinstance(v, np.int32):
                json_results[k] = int(v)
            else:
                json_results[k] = v

        json.dump(json_results, f, indent=4)

    # Print final results
    print("\nTraining completed!")
    print(
        f"Best validation accuracy: {metrics['best_val_acc']:.4f} at epoch {metrics['best_epoch'] + 1}"
    )
    print(f"Time elapsed: {time.time() - start_time:.2f} seconds")
    print(f"Rank-1 Accuracy: {metrics_results['rank-1']:.4f}")
    print(f"mAP: {metrics_results['mAP']:.4f}")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
