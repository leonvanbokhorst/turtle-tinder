#!/usr/bin/env python3
# Sea Turtle Re-Identification - Week 1
# Data Loading Utilities

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class TurtleDataset(Dataset):
    """
    Dataset class for loading turtle images with labels.

    The dataset assumes a directory structure where:
    - Each turtle ID has its own directory
    - Images for that turtle are inside that directory

    For example:
    data_dir/
        turtle_001/
            image1.jpg
            image2.jpg
        turtle_002/
            image1.jpg
            ...
    """

    def __init__(self, root_dir, transform=None):
        """
        Initialize the TurtleDataset.

        Args:
            root_dir: Root directory of the dataset with subdirectories for each turtle
            transform: Optional transform to apply to images
        """
        self.root_dir = root_dir
        self.transform = transform

        # Find all classes (turtle IDs)
        self.classes = sorted(
            [
                d
                for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            ]
        )

        # Map class names to numeric indices
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # Collect all sample images with their labels
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[class_name]))

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (image, label) where label is the class index
        """
        img_path, label = self.samples[idx]

        # Load image
        image = np.array(Image.open(img_path).convert("RGB"))

        # Apply transforms if provided
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label


def create_underwater_transforms(train=True):
    """
    Create transforms for underwater turtle images.

    Args:
        train: Whether to create transforms for training (with augmentation)
               or validation/testing (without augmentation)

    Returns:
        Albumentations transform object
    """
    # Common operations for both train and validation
    transforms_list = [
        A.Resize(height=224, width=224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]

    # Additional augmentations for training
    if train:
        train_transforms = [
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
        ]

        # Insert augmentations at the beginning of the list
        transforms_list = train_transforms + transforms_list

    return A.Compose(transforms_list)


def create_data_loaders(data_dir, batch_size=32, num_workers=4):
    """
    Create data loaders for training, validation, and test sets.

    Args:
        data_dir: Root directory containing train, val, and test subdirectories
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading

    Returns:
        Dictionary with train_loader, val_loader, test_loader, and class_names
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

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "class_names": train_dataset.classes,
    }


# Example usage
if __name__ == "__main__":
    print("Data loading utilities loaded. Import this module to use these functions.")
    print("Example usage:")
    print(
        """
    from utils.data_loader import create_data_loaders
    
    # Create data loaders for all sets
    loaders = create_data_loaders(
        data_dir='./processed_dataset',
        batch_size=32,
        num_workers=4
    )
    
    # Access individual loaders
    train_loader = loaders['train_loader']
    val_loader = loaders['val_loader']
    test_loader = loaders['test_loader']
    
    # Get class names (turtle IDs)
    class_names = loaders['class_names']
    
    # Example training loop
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            # Training step
            ...
    """
    )
