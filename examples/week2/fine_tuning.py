#!/usr/bin/env python3
# Sea Turtle Re-Identification - Week 2
# Progressive Fine-Tuning of Pre-trained Models

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
import sys

# Add the parent directory to the path to import from week1
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from week1.utils.data_loader import TurtleDataset, create_underwater_transforms


def parse_args():
    parser = argparse.ArgumentParser(
        description="Progressive fine-tuning of models for turtle identification"
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
        default="./fine_tuned_models",
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
        "--learning_rate",
        type=float,
        default=0.0001,
        help="Initial learning rate (should be lower for fine-tuning)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay (L2 regularization)",
    )
    parser.add_argument(
        "--epochs_per_stage",
        type=int,
        default=5,
        help="Number of epochs to train for each fine-tuning stage",
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


def create_model(model_name, num_classes):
    """
    Create a pre-trained model with a new classification head.

    Args:
        model_name: Name of the pre-trained model architecture
        num_classes: Number of classes to classify

    Returns:
        The model with a new classification head
    """
    model = None

    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

    elif model_name == "resnet50":
        model = models.resnet50(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

    elif model_name == "resnet101":
        model = models.resnet101(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=True)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.2, inplace=True), nn.Linear(num_features, num_classes)
        )

    elif model_name == "efficientnet_b3":
        model = models.efficientnet_b3(pretrained=True)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3, inplace=True), nn.Linear(num_features, num_classes)
        )

    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=True)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(num_features, num_classes)
        )

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


def get_fine_tuning_stages(model_name, model):
    """
    Define the layers to unfreeze at each stage of fine-tuning.

    Args:
        model_name: Name of the pre-trained model architecture
        model: The model to fine-tune

    Returns:
        List of stages, where each stage is a list of layer names to unfreeze
    """
    if "resnet" in model_name:
        # ResNet: fine-tune in stages, starting with the FC layer, then layer4, layer3, etc.
        return [
            ["fc"],  # Stage 1: Only the classifier
            ["layer4"],  # Stage 2: Last convolutional block
            ["layer3"],  # Stage 3: Third block
            ["layer2"],  # Stage 4: Second block
            ["layer1"],  # Stage 5: First block
        ]

    elif "efficientnet" in model_name:
        # For EfficientNet, we first train the classifier, then the features in reverse order
        # The deeper part of EfficientNet has more complex structures
        return [
            ["classifier"],  # Stage 1: Only the classifier
            ["features.7", "features.8"],  # Stage 2: Last features blocks
            ["features.6", "features.5"],  # Stage 3: Later middle features
            ["features.4", "features.3"],  # Stage 4: Early middle features
            ["features.2", "features.1", "features.0"],  # Stage 5: Early features
        ]

    elif model_name == "mobilenet_v2":
        # For MobileNetV2, similar approach but with their specific layer names
        return [
            ["classifier"],  # Stage 1: Only the classifier
            ["features.18", "features.17"],  # Stage 2: Last bottleneck blocks
            ["features.16", "features.15", "features.14"],  # Stage 3
            ["features.13", "features.12", "features.11"],  # Stage 4
            ["features.10", "features.9", "features.8"],  # Stage 5
            ["features.7", "features.6", "features.5"],  # Stage 6
            [
                "features.4",
                "features.3",
                "features.2",
                "features.1",
                "features.0",
            ],  # Stage 7
        ]

    else:
        # Default: just train everything at once after the initial classifier fine-tuning
        return [
            ["classifier", "fc"],  # Stage 1: Only the classifier
            [".*"],  # Stage 2: All layers
        ]


def freeze_all_layers(model):
    """
    Freeze all layers in the model.

    Args:
        model: The model to freeze
    """
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_layers(model, layer_names):
    """
    Unfreeze specified layers in the model.

    Args:
        model: The model to modify
        layer_names: List of layer name patterns to unfreeze (supports regex patterns)
    """
    import re

    for name, param in model.named_parameters():
        for layer_pattern in layer_names:
            if re.search(layer_pattern, name):
                param.requires_grad = True
                break


def count_trainable_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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


def save_metrics(metrics, output_dir):
    """Save training and validation metrics as plots for all stages"""
    plt.figure(figsize=(14, 10))

    # Plot losses
    plt.subplot(2, 1, 1)
    for stage, stage_metrics in enumerate(metrics):
        plt.plot(stage_metrics["train_loss"], label=f"Stage {stage+1} Train Loss")
        plt.plot(stage_metrics["val_loss"], label=f"Stage {stage+1} Val Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss Across Fine-tuning Stages")

    # Plot accuracies
    plt.subplot(2, 1, 2)
    for stage, stage_metrics in enumerate(metrics):
        plt.plot(stage_metrics["train_acc"], label=f"Stage {stage+1} Train Acc")
        plt.plot(stage_metrics["val_acc"], label=f"Stage {stage+1} Val Acc")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training and Validation Accuracy Across Fine-tuning Stages")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_metrics_all_stages.png"))
    plt.close()

    # Also save as CSV
    all_metrics = []
    for stage, stage_metrics in enumerate(metrics):
        for epoch in range(len(stage_metrics["train_loss"])):
            all_metrics.append(
                {
                    "stage": stage + 1,
                    "epoch": epoch + 1,
                    "train_loss": stage_metrics["train_loss"][epoch],
                    "train_acc": stage_metrics["train_acc"][epoch],
                    "val_loss": stage_metrics["val_loss"][epoch],
                    "val_acc": stage_metrics["val_acc"][epoch],
                }
            )

    df = pd.DataFrame(all_metrics)
    df.to_csv(os.path.join(output_dir, "training_metrics_all_stages.csv"), index=False)


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
    model = create_model(args.model, num_classes)

    # Get the fine-tuning stages for this model
    stages = get_fine_tuning_stages(args.model, model)
    print(f"Progressive fine-tuning with {len(stages)} stages")

    # Initially freeze all layers
    freeze_all_layers(model)

    # Move model to device
    model.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Initialize metrics storage
    all_stage_metrics = []
    best_val_acc = 0.0

    # Start timer for overall training
    start_time = time.time()

    # Fine-tune in stages
    for stage_idx, layer_names in enumerate(stages):
        print(f"\n{'='*80}")
        print(f"Stage {stage_idx+1}/{len(stages)}: Unfreezing {', '.join(layer_names)}")
        print(f"{'='*80}")

        # Unfreeze the layers for this stage
        unfreeze_layers(model, layer_names)

        # Count trainable parameters
        trainable_params = count_trainable_parameters(model)
        total_params = sum(p.numel() for p in model.parameters())
        print(
            f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})"
        )

        # Create optimizer for this stage (only update trainable parameters)
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, verbose=True
        )

        # Track metrics for this stage
        stage_train_losses = []
        stage_train_accs = []
        stage_val_losses = []
        stage_val_accs = []
        stage_best_acc = 0.0

        # Train for specified number of epochs
        for epoch in range(args.epochs_per_stage):
            print(f"Epoch {epoch+1}/{args.epochs_per_stage}")

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
            stage_train_losses.append(train_loss)
            stage_train_accs.append(train_acc)
            stage_val_losses.append(val_loss)
            stage_val_accs.append(val_acc)

            # Save best stage model
            if val_acc > stage_best_acc:
                stage_best_acc = val_acc
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        args.output_dir, f"best_model_stage_{stage_idx+1}.pth"
                    ),
                )
                print(
                    f"Saved new best model for stage {stage_idx+1} with accuracy: {val_acc:.4f}"
                )

            # Update overall best accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    model.state_dict(),
                    os.path.join(args.output_dir, "best_model_overall.pth"),
                )
                print(f"New overall best model with accuracy: {val_acc:.4f}")

        # Save stage metrics
        all_stage_metrics.append(
            {
                "train_loss": stage_train_losses,
                "train_acc": stage_train_accs,
                "val_loss": stage_val_losses,
                "val_acc": stage_val_accs,
                "best_acc": stage_best_acc,
                "trainable_params": trainable_params,
            }
        )

        # Save checkpoint for this stage
        torch.save(
            {
                "stage": stage_idx + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": stage_best_acc,
            },
            os.path.join(args.output_dir, f"checkpoint_stage_{stage_idx+1}.pth"),
        )

    # Training complete
    training_time = time.time() - start_time
    print(
        f"\nTraining completed in {str(datetime.timedelta(seconds=int(training_time)))}"
    )

    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model.pth"))

    # Save all metrics
    save_metrics(all_stage_metrics, args.output_dir)

    # Load best overall model for evaluation
    model.load_state_dict(
        torch.load(os.path.join(args.output_dir, "best_model_overall.pth"))
    )

    # Evaluate on test set
    test_acc, predictions, true_labels = test(model, test_loader, device)
    print(f"\nTest accuracy with best model: {test_acc:.4f}")

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

    # Save a summary of each stage
    stage_summary = []
    for i, stage_metrics in enumerate(all_stage_metrics):
        stage_summary.append(
            {
                "stage": i + 1,
                "trainable_params": stage_metrics["trainable_params"],
                "best_val_acc": stage_metrics["best_acc"],
                "final_train_acc": stage_metrics["train_acc"][-1],
                "final_val_acc": stage_metrics["val_acc"][-1],
            }
        )

    # Save stage summary
    stage_summary_df = pd.DataFrame(stage_summary)
    stage_summary_df.to_csv(
        os.path.join(args.output_dir, "stage_summary.csv"), index=False
    )

    # Create a plot showing improvement across stages
    plt.figure(figsize=(10, 6))
    plt.plot(
        [s["stage"] for s in stage_summary],
        [s["best_val_acc"] for s in stage_summary],
        "o-",
        label="Best Validation Accuracy",
    )
    plt.xlabel("Fine-tuning Stage")
    plt.ylabel("Validation Accuracy")
    plt.title("Accuracy Improvement Across Fine-tuning Stages")
    plt.xticks([s["stage"] for s in stage_summary])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "accuracy_by_stage.png"))

    print(f"\nAll results saved to {args.output_dir}")
    print("Stage summary:")
    print(stage_summary_df)
    print(f"\nBest overall validation accuracy: {best_val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
