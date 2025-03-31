#!/usr/bin/env python3
# Sea Turtle Re-Identification - Week 1
# Image Preprocessing and Augmentation Pipeline

import os
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torchvision


def parse_args():
    parser = argparse.ArgumentParser(description="Sea Turtle Image Preprocessing Demo")
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to a sample turtle image"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./augmentation_examples",
        help="Directory to save augmentation examples",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=10,
        help="Number of augmentation examples to generate",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    return parser.parse_args()


def create_underwater_augmentation_pipeline():
    """
    Create an augmentation pipeline tailored for underwater turtle images.

    This pipeline addresses common challenges in underwater photography:
    - Variable lighting conditions
    - Color shifts (blue/green tint)
    - Blur and distortion from water
    - Different viewing angles
    """
    transform = A.Compose(
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
            # Occasional minor distortions
            A.OneOf(
                [
                    A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.5),
                    A.GridDistortion(num_steps=5, distort_limit=0.05, p=0.5),
                    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
                ],
                p=0.3,
            ),
            # Resize and normalize (ImageNet values) for neural networks
            A.Resize(height=224, width=224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return transform


def create_basic_transform():
    """Basic resize and normalize without augmentation"""
    return A.Compose(
        [
            A.Resize(height=224, width=224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def denormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize an image for visualization.

    Args:
        image: Normalized tensor image [C, H, W]
        mean: Mean values used for normalization
        std: Std values used for normalization

    Returns:
        Denormalized numpy image [H, W, C] with values in [0, 1]
    """
    # Convert to numpy if tensor
    if isinstance(image, torch.Tensor):
        image = image.numpy()

    # If single channel, repeat to make RGB
    if image.shape[0] == 1:
        image = np.repeat(image, 3, axis=0)

    # Transpose from [C, H, W] to [H, W, C]
    image = image.transpose(1, 2, 0)

    # Denormalize
    image = image * np.array(std) + np.array(mean)

    # Clip values to [0, 1]
    image = np.clip(image, 0, 1)

    return image


def visualize_augmentations(image_path, transform, num_examples, output_dir):
    """
    Apply augmentations to an image and visualize the results.

    Args:
        image_path: Path to the input image
        transform: Albumentations transform pipeline
        num_examples: Number of augmented examples to generate
        output_dir: Directory to save visualization
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the image
    image = np.array(Image.open(image_path))

    # Create a figure
    plt.figure(figsize=(15, 15))

    # Show original image
    plt.subplot(4, 4, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    # Apply basic transform (resize + normalize)
    basic_transform = create_basic_transform()
    basic_result = basic_transform(image=image)["image"]
    plt.subplot(4, 4, 2)
    plt.imshow(denormalize(basic_result.transpose(2, 0, 1)))
    plt.title("Basic Transform")
    plt.axis("off")

    # Generate and display augmented examples
    for i in range(num_examples):
        # Apply augmentation
        augmented = transform(image=image)
        aug_image = augmented["image"]

        # Denormalize for visualization
        denorm_image = denormalize(aug_image.transpose(2, 0, 1))

        # Display
        plt.subplot(4, 4, i + 3)
        plt.imshow(denorm_image)
        plt.title(f"Augmentation {i+1}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "augmentation_examples.png"))
    plt.close()

    print(
        f"Saved augmentation examples to {os.path.join(output_dir, 'augmentation_examples.png')}"
    )


def create_dataloaders_example(data_dir, transform, batch_size=32):
    """
    Example of creating PyTorch dataloaders with the augmentation pipeline.
    This is a demonstration function showing how to integrate the augmentations
    into a complete training pipeline.

    Args:
        data_dir: Directory with images organized in class folders
        transform: Albumentations transform pipeline
        batch_size: Batch size for the dataloader

    Returns:
        Tuple of (train_loader, val_loader)
    """

    # Custom Dataset class for Albumentations
    class AlbumentationsDataset(torch.utils.data.Dataset):
        def __init__(self, root_dir, transform=None):
            self.root_dir = root_dir
            self.transform = transform
            self.classes = sorted(os.listdir(root_dir))
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

            self.samples = []
            for cls in self.classes:
                class_dir = os.path.join(root_dir, cls)
                if not os.path.isdir(class_dir):
                    continue

                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                        self.samples.append(
                            (os.path.join(class_dir, img_name), self.class_to_idx[cls])
                        )

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            img_path, label = self.samples[idx]
            image = np.array(Image.open(img_path).convert("RGB"))

            if self.transform:
                transformed = self.transform(image=image)
                image = transformed["image"]

                # Convert to tensor if not already
                if not isinstance(image, torch.Tensor):
                    image = torch.from_numpy(image.transpose(2, 0, 1)).float()

            return image, label

    # Create datasets with the transform
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    # Create validation transform without heavy augmentations
    val_transform = A.Compose(
        [
            A.Resize(height=224, width=224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    # Ensure the transform includes ToTensorV2 at the end
    if "ToTensorV2" not in str(transform):
        transform = A.Compose([*transform, ToTensorV2()])

    train_dataset = AlbumentationsDataset(train_dir, transform=transform)
    val_dataset = AlbumentationsDataset(val_dir, transform=val_transform)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return train_loader, val_loader


def main():
    args = parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create the augmentation pipeline
    transform = create_underwater_augmentation_pipeline()

    # Apply and visualize augmentations
    visualize_augmentations(
        args.image_path, transform, args.num_examples, args.output_dir
    )

    # Example usage in PyTorch (for reference)
    print("\nExample code for creating dataloaders:")
    print(
        """
    # Create augmentation transform
    transform = create_underwater_augmentation_pipeline()
    
    # Add ToTensorV2 to convert to PyTorch tensors at the end
    transform = A.Compose([*transform, ToTensorV2()])
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders_example(
        data_dir='./processed_dataset', 
        transform=transform,
        batch_size=32
    )
    
    # Use in training loop
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            # Forward pass
            # ...
    """
    )


if __name__ == "__main__":
    main()
