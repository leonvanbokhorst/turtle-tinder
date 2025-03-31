#!/usr/bin/env python3
# Sea Turtle Re-Identification - Week 1
# Visualization Utilities

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd


def plot_images(
    images,
    labels=None,
    predictions=None,
    class_names=None,
    num_images=16,
    figsize=(12, 12),
    title=None,
):
    """
    Plot a grid of images with optional labels and predictions.

    Args:
        images: List of images or tensor of shape [N, C, H, W]
        labels: Optional ground truth labels
        predictions: Optional model predictions
        class_names: Optional list of class names for labels
        num_images: Number of images to display
        figsize: Figure size
        title: Optional title for the figure
    """
    # Convert tensor to numpy if needed
    if torch.is_tensor(images):
        images = images.cpu().numpy()

    # If we have a tensor of shape [N, C, H, W], convert to [N, H, W, C]
    if len(images.shape) == 4 and images.shape[1] == 3:
        images = np.transpose(images, (0, 2, 3, 1))

    # Limit the number of images to display
    num_images = min(num_images, len(images))

    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_images)))

    # Create figure
    plt.figure(figsize=figsize)

    # Add overall title if provided
    if title:
        plt.suptitle(title, fontsize=16)

    # Plot each image
    for i in range(num_images):
        plt.subplot(grid_size, grid_size, i + 1)

        # Get the image
        img = images[i]

        # If normalized, denormalize for display
        if img.max() <= 1.0:
            plt.imshow(img)
        else:
            plt.imshow(img.astype(np.uint8))

        # Add label/prediction information if provided
        if labels is not None or predictions is not None:
            label_text = ""

            # Add ground truth label
            if labels is not None:
                label = labels[i]
                if class_names and label < len(class_names):
                    label_text += f"True: {class_names[label]}"
                else:
                    label_text += f"True: {label}"

            # Add prediction
            if predictions is not None:
                pred = predictions[i]
                if class_names and pred < len(class_names):
                    label_text += f"\nPred: {class_names[pred]}"
                else:
                    label_text += f"\nPred: {pred}"

                # Color code based on correctness
                if labels is not None:
                    if pred == labels[i]:
                        plt.title(label_text, color="green")
                    else:
                        plt.title(label_text, color="red")
                else:
                    plt.title(label_text)
            else:
                plt.title(label_text)

        plt.axis("off")

    plt.tight_layout()

    return plt.gcf()  # Return the figure for further customization


def denormalize_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize an image that was normalized with the given mean and std.

    Args:
        image: Normalized image tensor or numpy array [C, H, W] or [H, W, C]
        mean: Mean values used for normalization
        std: Standard deviation values used for normalization

    Returns:
        Denormalized image in range [0, 1]
    """
    # Convert tensor to numpy if needed
    if torch.is_tensor(image):
        image = image.cpu().numpy()

    # Handle [C, H, W] vs [H, W, C] format
    if image.shape[0] == 3 and len(image.shape) == 3:  # [C, H, W]
        # Convert to [H, W, C]
        image = np.transpose(image, (1, 2, 0))

        # Denormalize
        image = image * np.array(std) + np.array(mean)
    else:  # Already [H, W, C]
        image = image * np.array(std) + np.array(mean)

    # Clip values to [0, 1]
    image = np.clip(image, 0, 1)

    return image


def visualize_augmentations(
    original_image, transforms, num_examples=5, figsize=(15, 10)
):
    """
    Apply and visualize multiple augmentations to an image.

    Args:
        original_image: PIL Image or numpy array
        transforms: Albumentations transform function
        num_examples: Number of augmented examples to generate
        figsize: Figure size

    Returns:
        Matplotlib figure with original and augmented images
    """
    # Convert PIL to numpy if needed
    if isinstance(original_image, Image.Image):
        original_image = np.array(original_image)

    plt.figure(figsize=figsize)

    # Show original image
    plt.subplot(3, 3, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis("off")

    # Apply transforms multiple times
    for i in range(num_examples):
        # Apply transform
        augmented = transforms(image=original_image)
        aug_image = augmented["image"]

        # If normalized, denormalize for display
        if "Normalize" in str(transforms):
            aug_image = denormalize_image(aug_image)

        # Display
        plt.subplot(3, 3, i + 2)
        plt.imshow(aug_image)
        plt.title(f"Augmentation {i+1}")
        plt.axis("off")

    plt.tight_layout()

    return plt.gcf()


def plot_confusion_matrix(
    y_true, y_pred, class_names=None, figsize=(10, 8), normalize=True
):
    """
    Plot a confusion matrix for classification results.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        figsize: Figure size
        normalize: Whether to normalize by row (true label)

    Returns:
        Matplotlib figure with confusion matrix
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Normalize if requested
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
        title = "Normalized Confusion Matrix"
    else:
        fmt = "d"
        title = "Confusion Matrix"

    # Create figure
    plt.figure(figsize=figsize)

    # Plot confusion matrix
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names if class_names else "auto",
        yticklabels=class_names if class_names else "auto",
    )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title(title)
    plt.tight_layout()

    return plt.gcf()


def plot_training_history(history, figsize=(12, 6)):
    """
    Plot training and validation metrics from a history dictionary or DataFrame.

    Args:
        history: Dict or DataFrame containing 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        figsize: Figure size

    Returns:
        Matplotlib figure with training history plots
    """
    # Convert dict to DataFrame if needed
    if isinstance(history, dict):
        history = pd.DataFrame(history)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot loss
    axes[0].plot(history["train_loss"], label="Training Loss")
    axes[0].plot(history["val_loss"], label="Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].set_title("Training and Validation Loss")

    # Plot accuracy
    axes[1].plot(history["train_acc"], label="Training Accuracy")
    axes[1].plot(history["val_acc"], label="Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].set_title("Training and Validation Accuracy")

    plt.tight_layout()

    return fig


def visualize_model_predictions(model, dataloader, class_names, device, num_images=16):
    """
    Visualize model predictions on sample images from a dataloader.

    Args:
        model: PyTorch model
        dataloader: DataLoader to get sample images
        class_names: List of class names
        device: Device to run model on
        num_images: Number of images to display

    Returns:
        Matplotlib figure with predictions
    """
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(15, 15))

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 4 + 1, 4, images_so_far)
                ax.axis("off")

                # Denormalize image for display
                image = denormalize_image(inputs.cpu()[j])

                # Display image
                ax.imshow(image)

                # Color code based on prediction correctness
                if preds[j] == labels[j]:
                    ax.set_title(
                        f"True: {class_names[labels[j]]}\nPred: {class_names[preds[j]]}",
                        color="green",
                    )
                else:
                    ax.set_title(
                        f"True: {class_names[labels[j]]}\nPred: {class_names[preds[j]]}",
                        color="red",
                    )

                if images_so_far == num_images:
                    return fig

    return fig


def print_classification_report(y_true, y_pred, class_names=None):
    """
    Print a classification report with precision, recall, f1-score.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    report = classification_report(
        y_true, y_pred, target_names=class_names if class_names else None, digits=3
    )

    print("\nClassification Report:")
    print(report)

    return report


# Example usage when script is run directly
if __name__ == "__main__":
    print("Visualization utilities loaded. Import this module to use these functions.")
    print("Example functions:")
    print(
        "- plot_images: Display a grid of images with optional labels and predictions"
    )
    print("- denormalize_image: Convert normalized images back to viewable format")
    print("- visualize_augmentations: Show examples of data augmentations")
    print(
        "- plot_confusion_matrix: Visualize model performance with a confusion matrix"
    )
    print("- plot_training_history: Track training progress over epochs")
    print("- visualize_model_predictions: See how the model performs on sample images")
    print("- print_classification_report: Detailed metrics on model performance")
