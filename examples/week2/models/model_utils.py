#!/usr/bin/env python3
# Sea Turtle Re-Identification - Week 2
# Model utility functions

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import time
import datetime
from sklearn.metrics import confusion_matrix, classification_report


def count_parameters(model: nn.Module) -> Tuple[int, int, float]:
    """
    Count the total and trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Tuple of (total_params, trainable_params, trainable_percentage)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_percentage = (
        100.0 * trainable_params / total_params if total_params > 0 else 0.0
    )

    return total_params, trainable_params, trainable_percentage


def create_optimizer(
    model: nn.Module,
    optimizer_name: str = "adam",
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    momentum: float = 0.9,  # For SGD
    beta1: float = 0.9,  # For Adam
    beta2: float = 0.999,  # For Adam
) -> optim.Optimizer:
    """
    Create an optimizer for the trainable parameters of a model.

    Args:
        model: PyTorch model
        optimizer_name: Name of optimizer ('adam', 'sgd', etc.)
        learning_rate: Learning rate
        weight_decay: L2 regularization strength
        momentum: Momentum for SGD
        beta1: First beta parameter for Adam
        beta2: Second beta parameter for Adam

    Returns:
        PyTorch optimizer
    """
    # Get only trainable parameters
    params = filter(lambda p: p.requires_grad, model.parameters())

    if optimizer_name.lower() == "adam":
        return optim.Adam(
            params, lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2)
        )
    elif optimizer_name.lower() == "sgd":
        return optim.SGD(
            params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay
        )
    elif optimizer_name.lower() == "adamw":
        return optim.AdamW(
            params, lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2)
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_name: str = "plateau",
    mode: str = "min",
    factor: float = 0.5,
    patience: int = 5,
    min_lr: float = 1e-6,
    epochs: int = 30,
    warmup_epochs: int = 5,
) -> Any:
    """
    Create a learning rate scheduler.

    Args:
        optimizer: PyTorch optimizer
        scheduler_name: Name of the scheduler
        mode: 'min' for loss, 'max' for accuracy (for ReduceLROnPlateau)
        factor: Factor to reduce learning rate by
        patience: Number of epochs with no improvement before reducing LR
        min_lr: Minimum learning rate
        epochs: Total number of epochs (for cosine annealing)
        warmup_epochs: Number of warmup epochs (for warmup schedulers)

    Returns:
        Learning rate scheduler
    """
    if scheduler_name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            verbose=True,
        )
    elif scheduler_name == "step":
        step_size = epochs // 3  # Reduce LR three times during training
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=factor)
    elif scheduler_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=min_lr
        )
    elif scheduler_name == "warmup_cosine":
        # Custom lambda function for linear warmup followed by cosine annealing
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Linear warmup
                return epoch / warmup_epochs
            else:
                # Cosine annealing
                progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def save_model(
    model: nn.Module,
    epoch: int,
    optimizer: optim.Optimizer,
    scheduler: Any,
    train_loss: float,
    val_loss: float,
    train_acc: float,
    val_acc: float,
    path: str,
    is_best: bool = False,
    metadata: Dict[str, Any] = None,
) -> None:
    """
    Save a model checkpoint.

    Args:
        model: PyTorch model
        epoch: Current epoch
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        train_loss: Training loss
        val_loss: Validation loss
        train_acc: Training accuracy
        val_acc: Validation accuracy
        path: Path to save the checkpoint
        is_best: Whether this is the best model so far
        metadata: Additional metadata to save
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict() if hasattr(scheduler, "state_dict") else None
        ),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "metadata": metadata or {},
    }

    # Save the regular checkpoint
    torch.save(checkpoint, path)

    # If this is the best model, save a copy
    if is_best:
        best_path = os.path.join(os.path.dirname(path), "best_model.pth")
        torch.save(checkpoint, best_path)


def load_model(
    model: nn.Module,
    path: str,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: torch.device = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load a model checkpoint.

    Args:
        model: PyTorch model structure (uninitialized)
        path: Path to the checkpoint
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load the model onto

    Returns:
        Tuple of (loaded_model, checkpoint_dict)
    """
    # Load the checkpoint
    if device is None:
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location=device)

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer state if provided
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state if provided
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return model, checkpoint


def freeze_layers(model: nn.Module, layer_names: List[str] = None) -> None:
    """
    Freeze specific layers in the model or all layers if none specified.

    Args:
        model: PyTorch model
        layer_names: List of layer name patterns to freeze
    """
    if layer_names is None:
        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False
    else:
        import re

        # Freeze only specified layers
        for name, param in model.named_parameters():
            for layer_pattern in layer_names:
                if re.search(layer_pattern, name):
                    param.requires_grad = False
                    break


def unfreeze_layers(model: nn.Module, layer_names: List[str]) -> None:
    """
    Unfreeze specific layers in the model.

    Args:
        model: PyTorch model
        layer_names: List of layer name patterns to unfreeze
    """
    import re

    for name, param in model.named_parameters():
        for layer_pattern in layer_names:
            if re.search(layer_pattern, name):
                param.requires_grad = True
                break


def plot_learning_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    title: str = "Training Progress",
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot learning curves for training and validation.

    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accs: List of training accuracies
        val_accs: List of validation accuracies
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=figsize)
    plt.suptitle(title)

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, "-o", label="Training Loss")
    plt.plot(epochs, val_losses, "-o", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.grid(alpha=0.3)

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, "-o", label="Training Accuracy")
    plt.plot(epochs, val_accs, "-o", label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training and Validation Accuracy")
    plt.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "Blues",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot a confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        title: Plot title
        figsize: Figure size
        cmap: Colormap
        save_path: Path to save the figure
    """
    import seaborn as sns

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create figure
    plt.figure(figsize=figsize)

    # Plot confusion matrix
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def get_classification_metrics(
    y_true: List[int], y_pred: List[int], class_names: List[str]
) -> Tuple[Dict[str, float], str]:
    """
    Calculate classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names

    Returns:
        Tuple of (metrics_dict, report_string)
    """
    # Calculate classification report
    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=3, output_dict=True
    )

    # Convert to string format for printing
    report_string = classification_report(
        y_true, y_pred, target_names=class_names, digits=3
    )

    # Extract overall metrics
    metrics = {
        "accuracy": report["accuracy"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_precision": report["weighted avg"]["precision"],
        "weighted_recall": report["weighted avg"]["recall"],
        "weighted_f1": report["weighted avg"]["f1-score"],
    }

    return metrics, report_string


def time_execution(func: Callable) -> Callable:
    """
    Decorator to time the execution of a function.

    Args:
        func: Function to time

    Returns:
        Wrapped function that prints execution time
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time

        # Format time based on duration
        if execution_time < 60:
            time_str = f"{execution_time:.2f} seconds"
        else:
            time_str = str(datetime.timedelta(seconds=int(execution_time)))

        print(f"Execution of {func.__name__} completed in {time_str}")
        return result

    return wrapper


def visualize_model_predictions(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    class_names: List[str],
    device: torch.device,
    num_images: int = 16,
    figsize: Tuple[int, int] = (15, 15),
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize model predictions on sample images.

    Args:
        model: Trained PyTorch model
        dataloader: DataLoader with images
        class_names: List of class names
        device: Device to run inference on
        num_images: Number of images to display
        figsize: Figure size
        save_path: Path to save the figure
    """
    import math
    import numpy as np

    def denormalize(img):
        """Denormalize an image tensor to numpy array for display"""
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        # Convert to numpy and transpose from [C, H, W] to [H, W, C]
        img = img.cpu().numpy().transpose(1, 2, 0)

        # Denormalize
        img = std * img + mean
        img = np.clip(img, 0, 1)

        return img

    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=figsize)

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1

                # Calculate grid position
                ax = plt.subplot(math.ceil(num_images / 4), 4, images_so_far)
                ax.axis("off")

                # Display image
                image = denormalize(inputs[j])
                ax.imshow(image)

                # Add color-coded title
                color = "green" if preds[j] == labels[j] else "red"
                ax.set_title(
                    f"True: {class_names[labels[j]]}\nPred: {class_names[preds[j]]}",
                    color=color,
                )

                if images_so_far == num_images:
                    if save_path:
                        plt.savefig(save_path)
                        plt.close()
                    else:
                        plt.tight_layout()
                        plt.show()
                    return

        # In case we didn't get enough images from the dataloader
        if images_so_far > 0:
            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.tight_layout()
                plt.show()


def get_layer_groups(model_name: str) -> List[List[str]]:
    """
    Get layer groups for a specific model architecture.
    Useful for progressive fine-tuning.

    Args:
        model_name: Model architecture name

    Returns:
        List of layer groups, each group containing regex patterns
    """
    if "resnet" in model_name:
        # ResNet layer groups, from shallow to deep
        return [
            ["fc"],  # Classifier only
            ["layer4"],  # Final ResNet block
            ["layer3"],  # Third block
            ["layer2"],  # Second block
            ["layer1"],  # First block
            ["conv1", "bn1"],  # Initial convolution
        ]

    elif "efficientnet" in model_name:
        # EfficientNet layer groups
        return [
            ["classifier"],  # Classifier only
            ["features.8", "features.7"],  # Final features
            ["features.6", "features.5"],  # Later middle features
            ["features.4", "features.3"],  # Early middle features
            ["features.2", "features.1", "features.0"],  # Initial features
        ]

    elif "mobilenet" in model_name:
        # MobileNet layer groups
        return [
            ["classifier"],  # Classifier only
            ["features.18", "features.17", "features.16"],  # Final features
            ["features.15", "features.14", "features.13"],  # Later middle
            ["features.12", "features.11", "features.10"],  # Middle
            ["features.9", "features.8", "features.7"],  # Early middle
            ["features.6", "features.5", "features.4"],  # Early
            ["features.3", "features.2", "features.1", "features.0"],  # Initial
        ]

    else:
        # Default generic pattern
        return [["classifier", "fc"], [".*"]]  # Classifier only  # All layers


def apply_progressive_freezing(
    model: nn.Module,
    model_name: str,
    current_stage: int,
    freeze_before_layer: Optional[str] = None,
) -> nn.Module:
    """
    Apply progressive freezing/unfreezing for fine-tuning.

    Args:
        model: PyTorch model
        model_name: Model architecture name
        current_stage: Current stage of fine-tuning (0-based index)
        freeze_before_layer: If provided, freeze all layers before this pattern

    Returns:
        Modified model with appropriate layers frozen/unfrozen
    """
    # Get layer groups for this architecture
    layer_groups = get_layer_groups(model_name)

    if current_stage >= len(layer_groups):
        # If stage exceeds available groups, unfreeze everything
        for param in model.parameters():
            param.requires_grad = True
        return model

    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Then unfreeze layers up to the current stage
    layers_to_unfreeze = []
    for i in range(current_stage + 1):
        layers_to_unfreeze.extend(layer_groups[i])

    # Unfreeze the selected layers
    unfreeze_layers(model, layers_to_unfreeze)

    # If freeze_before_layer is specified, freeze all layers before it
    if freeze_before_layer:
        import re

        layer_found = False

        for name, param in model.named_parameters():
            if re.search(freeze_before_layer, name):
                layer_found = True
                continue

            if not layer_found:
                param.requires_grad = False

    return model


def get_lr_range_test(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    start_lr: float = 1e-7,
    end_lr: float = 10,
    num_iterations: int = 100,
    save_path: Optional[str] = None,
) -> Tuple[List[float], List[float]]:
    """
    Perform a learning rate range test to find optimal learning rate.

    Args:
        model: PyTorch model
        train_loader: DataLoader with training data
        criterion: Loss function
        device: Device to run the test on
        start_lr: Starting learning rate
        end_lr: Ending learning rate
        num_iterations: Number of iterations for the test
        save_path: Path to save the resulting plot

    Returns:
        Tuple of (learning_rates, losses)
    """
    # Create a copy of the model to avoid modifying the original
    model_copy = type(model)()
    model_copy.load_state_dict(model.state_dict())
    model_copy.to(device)
    model_copy.train()

    # Create an optimizer with very low learning rate
    optimizer = optim.Adam(model_copy.parameters(), lr=start_lr)

    # Calculate the multiplier to increase lr each step
    gamma = (end_lr / start_lr) ** (1 / num_iterations)

    # Create a learning rate scheduler
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    # Lists to store learning rates and losses
    learning_rates = []
    losses = []

    # Get a fixed batch for the test
    it = iter(train_loader)
    batch = next(it)

    # Run the test
    for i in range(num_iterations):
        # Get the current learning rate
        lr = optimizer.param_groups[0]["lr"]
        learning_rates.append(lr)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model_copy(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Record the loss
        losses.append(loss.item())

        # Update weights and learning rate
        optimizer.step()
        scheduler.step()

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(learning_rates, losses)
    plt.xscale("log")
    plt.xlabel("Learning Rate (log scale)")
    plt.ylabel("Loss")
    plt.title("Learning Rate Range Test")
    plt.grid(True, alpha=0.3)

    # Find and mark the point of steepest descent
    min_gradient_idx = np.argmin(np.gradient(losses))
    min_loss_idx = np.argmin(losses)

    plt.scatter(
        learning_rates[min_gradient_idx],
        losses[min_gradient_idx],
        color="red",
        s=75,
        marker="o",
    )
    plt.annotate(
        f"Steepest: {learning_rates[min_gradient_idx]:.1e}",
        (learning_rates[min_gradient_idx], losses[min_gradient_idx]),
        xytext=(10, 10),
        textcoords="offset points",
    )

    plt.scatter(
        learning_rates[min_loss_idx],
        losses[min_loss_idx],
        color="green",
        s=75,
        marker="o",
    )
    plt.annotate(
        f"Min Loss: {learning_rates[min_loss_idx]:.1e}",
        (learning_rates[min_loss_idx], losses[min_loss_idx]),
        xytext=(10, -15),
        textcoords="offset points",
    )

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

    return learning_rates, losses
