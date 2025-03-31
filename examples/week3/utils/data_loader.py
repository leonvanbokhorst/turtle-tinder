"""
Data loader utilities for metric learning in sea turtle re-identification.

This module provides specialized dataset classes for training Siamese,
Triplet, and ArcFace networks.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from typing import List, Dict, Tuple, Optional, Union, Callable


class TurtleDataset(Dataset):
    """
    Base dataset for sea turtle re-identification.
    """

    def __init__(
        self, data_dir: str, split: str = "train", transform: Optional[Callable] = None
    ):
        """
        Initialize the dataset.

        Args:
            data_dir: Directory containing the image data
            split: Dataset split ('train', 'val', or 'test')
            transform: Transforms to apply to the images
        """
        self.data_dir = data_dir
        self.split = split

        # Set up transforms
        if transform is None:
            # Default transform
            if split == "train":
                # More augmentations for training
                self.transform = transforms.Compose(
                    [
                        transforms.Resize((256, 256)),
                        transforms.RandomCrop(224),
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
            else:
                # Minimal transforms for validation/testing
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

        # Load the data
        self.image_paths, self.labels, self.class_to_idx = self._load_dataset(
            data_dir, split
        )

        # Create a mapping from label to list of indices
        self.label_to_indices = self._group_by_label()

    def _load_dataset(
        self, data_dir: str, split: str
    ) -> Tuple[List[str], List[int], Dict[str, int]]:
        """
        Load dataset from directory.

        Args:
            data_dir: Directory containing the image data
            split: Dataset split ('train', 'val', or 'test')

        Returns:
            Tuple of (image_paths, labels, class_to_idx)
        """
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            raise ValueError(f"Split directory {split_dir} does not exist")

        image_paths = []
        labels = []
        class_to_idx = {}

        # Each subdirectory is a class
        class_dirs = sorted(
            [
                d
                for d in os.listdir(split_dir)
                if os.path.isdir(os.path.join(split_dir, d))
            ]
        )

        for idx, class_dir in enumerate(class_dirs):
            class_to_idx[class_dir] = idx
            class_path = os.path.join(split_dir, class_dir)
            class_images = [
                os.path.join(class_path, f)
                for f in os.listdir(class_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]

            image_paths.extend(class_images)
            labels.extend([idx] * len(class_images))

        return image_paths, labels, class_to_idx

    def _group_by_label(self) -> Dict[int, List[int]]:
        """
        Group image indices by label.

        Returns:
            Dictionary mapping labels to lists of image indices
        """
        label_to_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(idx)
        return label_to_indices

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get an image and its label.

        Args:
            idx: Index of the image

        Returns:
            Tuple of (image, label)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        img = Image.open(img_path).convert("RGB")

        # Apply transforms
        if self.transform:
            img = self.transform(img)

        return img, label


class SiamesePairDataset(Dataset):
    """
    Dataset for training Siamese networks with pairs of images.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        pairs_per_class: int = 10,
    ):
        """
        Initialize the dataset.

        Args:
            data_dir: Directory containing the image data
            split: Dataset split ('train', 'val', or 'test')
            transform: Transforms to apply to the images
            pairs_per_class: Number of positive and negative pairs to generate per class
        """
        # Load base dataset
        self.base_dataset = TurtleDataset(data_dir, split, transform)
        self.transform = self.base_dataset.transform
        self.pairs_per_class = pairs_per_class

        # Generate pairs
        self.pairs = self._generate_pairs()

    def _generate_pairs(self) -> List[Tuple[int, int, int]]:
        """
        Generate pairs of images for training.
        Each pair consists of (idx1, idx2, label) where label is 1 if same class, 0 if different.

        Returns:
            List of (idx1, idx2, label) tuples
        """
        pairs = []
        label_to_indices = self.base_dataset.label_to_indices

        # For each class
        for class_idx in label_to_indices:
            class_indices = label_to_indices[class_idx]

            # Skip classes with only one image
            if len(class_indices) < 2:
                continue

            # Generate positive pairs (same class)
            for _ in range(self.pairs_per_class):
                # Randomly select two images from the same class
                idx1, idx2 = random.sample(class_indices, 2)
                pairs.append((idx1, idx2, 1))  # 1 = same class

            # Generate negative pairs (different classes)
            for _ in range(self.pairs_per_class):
                # Randomly select one image from this class
                idx1 = random.choice(class_indices)

                # Randomly select a different class
                negative_class = random.choice(
                    [c for c in label_to_indices.keys() if c != class_idx]
                )

                # Randomly select one image from the negative class
                idx2 = random.choice(label_to_indices[negative_class])

                pairs.append((idx1, idx2, 0))  # 0 = different class

        random.shuffle(pairs)
        return pairs

    def __len__(self) -> int:
        """Return the number of pairs in the dataset."""
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a pair of images and a label.

        Args:
            idx: Index of the pair

        Returns:
            Tuple of (image1, image2, label)
        """
        idx1, idx2, label = self.pairs[idx]

        # Get images
        img1_path = self.base_dataset.image_paths[idx1]
        img2_path = self.base_dataset.image_paths[idx2]

        # Load images
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        # Apply transforms
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label)


class TripletDataset(Dataset):
    """
    Dataset for training Triplet networks with triplets of images.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        triplets_per_class: int = 10,
    ):
        """
        Initialize the dataset.

        Args:
            data_dir: Directory containing the image data
            split: Dataset split ('train', 'val', or 'test')
            transform: Transforms to apply to the images
            triplets_per_class: Number of triplets to generate per class
        """
        # Load base dataset
        self.base_dataset = TurtleDataset(data_dir, split, transform)
        self.transform = self.base_dataset.transform
        self.triplets_per_class = triplets_per_class

        # Generate triplets
        self.triplets = self._generate_triplets()

    def _generate_triplets(self) -> List[Tuple[int, int, int]]:
        """
        Generate triplets of images for training.
        Each triplet consists of (anchor_idx, pos_idx, neg_idx).

        Returns:
            List of (anchor_idx, pos_idx, neg_idx) tuples
        """
        triplets = []
        label_to_indices = self.base_dataset.label_to_indices

        # For each class
        for class_idx in label_to_indices:
            class_indices = label_to_indices[class_idx]

            # Skip classes with only one image
            if len(class_indices) < 2:
                continue

            # Generate triplets for this class
            for _ in range(self.triplets_per_class):
                # Randomly select anchor and positive (same class)
                anchor_idx, pos_idx = random.sample(class_indices, 2)

                # Randomly select a different class
                negative_class = random.choice(
                    [c for c in label_to_indices.keys() if c != class_idx]
                )

                # Randomly select negative from the different class
                neg_idx = random.choice(label_to_indices[negative_class])

                triplets.append((anchor_idx, pos_idx, neg_idx))

        random.shuffle(triplets)
        return triplets

    def __len__(self) -> int:
        """Return the number of triplets in the dataset."""
        return len(self.triplets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a triplet of images.

        Args:
            idx: Index of the triplet

        Returns:
            Tuple of (anchor, positive, negative)
        """
        anchor_idx, pos_idx, neg_idx = self.triplets[idx]

        # Get images
        anchor_path = self.base_dataset.image_paths[anchor_idx]
        pos_path = self.base_dataset.image_paths[pos_idx]
        neg_path = self.base_dataset.image_paths[neg_idx]

        # Load images
        anchor = Image.open(anchor_path).convert("RGB")
        positive = Image.open(pos_path).convert("RGB")
        negative = Image.open(neg_path).convert("RGB")

        # Apply transforms
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        # Get labels
        anchor_label = self.base_dataset.labels[anchor_idx]
        pos_label = self.base_dataset.labels[pos_idx]
        neg_label = self.base_dataset.labels[neg_idx]

        # Ensure correct labels
        assert anchor_label == pos_label
        assert anchor_label != neg_label

        return anchor, positive, negative


class OnlineTripletDataset(Dataset):
    """
    Dataset for online triplet mining.
    Returns individual samples with labels, and triplets are mined during training.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        samples_per_class: int = 4,
    ):
        """
        Initialize the dataset.

        Args:
            data_dir: Directory containing the image data
            split: Dataset split ('train', 'val', or 'test')
            transform: Transforms to apply to the images
            samples_per_class: Number of samples per class in each batch
        """
        # Load base dataset
        self.base_dataset = TurtleDataset(data_dir, split, transform)
        self.transform = self.base_dataset.transform
        self.samples_per_class = samples_per_class

        # Create a balanced sampling strategy
        self.indices = self._create_balanced_sampling()

    def _create_balanced_sampling(self) -> List[int]:
        """
        Create a balanced sampling strategy where each class has roughly
        the same number of samples in each epoch.

        Returns:
            List of indices to sample
        """
        label_to_indices = self.base_dataset.label_to_indices

        # Create balanced indices
        indices = []

        # For each class
        for class_idx in label_to_indices:
            class_indices = label_to_indices[class_idx]

            # Sample with replacement if needed
            if len(class_indices) < self.samples_per_class:
                sampled_indices = random.choices(
                    class_indices, k=self.samples_per_class
                )
            else:
                sampled_indices = random.sample(class_indices, self.samples_per_class)

            indices.extend(sampled_indices)

        random.shuffle(indices)
        return indices

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get an image and its label.

        Args:
            idx: Index in the balanced sampling

        Returns:
            Tuple of (image, label)
        """
        true_idx = self.indices[idx]

        # Get image and label
        img_path = self.base_dataset.image_paths[true_idx]
        label = self.base_dataset.labels[true_idx]

        # Load image
        img = Image.open(img_path).convert("RGB")

        # Apply transforms
        if self.transform:
            img = self.transform(img)

        return img, label


def create_dataloaders(
    data_dir: str,
    batch_size: int,
    dataset_type: str = "triplet",
    num_workers: int = 4,
    **kwargs,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders for training, validation, and testing.

    Args:
        data_dir: Directory containing the image data
        batch_size: Batch size
        dataset_type: Type of dataset ('standard', 'siamese', 'triplet', or 'online_triplet')
        num_workers: Number of workers for data loading
        **kwargs: Additional arguments for the dataset

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Set up transforms
    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create datasets based on type
    if dataset_type == "standard":
        train_dataset = TurtleDataset(data_dir, "train", train_transform)
        val_dataset = TurtleDataset(data_dir, "val", val_transform)
        test_dataset = TurtleDataset(data_dir, "test", val_transform)

    elif dataset_type == "siamese":
        train_dataset = SiamesePairDataset(data_dir, "train", train_transform, **kwargs)
        val_dataset = SiamesePairDataset(data_dir, "val", val_transform, **kwargs)
        test_dataset = SiamesePairDataset(data_dir, "test", val_transform, **kwargs)

    elif dataset_type == "triplet":
        train_dataset = TripletDataset(data_dir, "train", train_transform, **kwargs)
        val_dataset = TripletDataset(data_dir, "val", val_transform, **kwargs)
        test_dataset = TripletDataset(data_dir, "test", val_transform, **kwargs)

    elif dataset_type == "online_triplet":
        train_dataset = OnlineTripletDataset(
            data_dir, "train", train_transform, **kwargs
        )
        val_dataset = OnlineTripletDataset(data_dir, "val", val_transform, **kwargs)
        test_dataset = OnlineTripletDataset(data_dir, "test", val_transform, **kwargs)

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # Create dataloaders
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

    return train_loader, val_loader, test_loader


class UnderWaterAugmentation:
    """
    Custom transformation for underwater image augmentation.
    Simulates underwater effects like color attenuation, blur, and particles.
    """

    def __init__(
        self,
        prob: float = 0.5,
        blue_tint_strength: float = 0.3,
        haze_strength: float = 0.2,
        blur_strength: float = 0.5,
    ):
        """
        Initialize the underwater augmentation.

        Args:
            prob: Probability of applying the augmentation
            blue_tint_strength: Strength of blue color shift (0-1)
            haze_strength: Strength of underwater haze effect (0-1)
            blur_strength: Strength of underwater blur (0-1)
        """
        self.prob = prob
        self.blue_tint_strength = blue_tint_strength
        self.haze_strength = haze_strength
        self.blur_strength = blur_strength

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Apply underwater augmentation to an image.

        Args:
            img: Input image

        Returns:
            Augmented image
        """
        if random.random() > self.prob:
            return img

        # Convert to numpy array
        img_np = np.array(img).astype(np.float32) / 255.0

        # Apply blue tint (attenuate red and green channels)
        if random.random() < self.blue_tint_strength:
            strength = random.uniform(0.1, 0.4)
            img_np[:, :, 0] *= 1 - strength  # Reduce red channel
            img_np[:, :, 1] *= 1 - strength * 0.7  # Reduce green channel less

        # Apply underwater haze
        if random.random() < self.haze_strength:
            haze = np.ones_like(img_np) * np.array([0.1, 0.2, 0.4])  # Blue-ish haze
            strength = random.uniform(0.05, 0.2)
            img_np = img_np * (1 - strength) + haze * strength

        # Apply blur
        if random.random() < self.blur_strength:
            from scipy.ndimage import gaussian_filter

            sigma = random.uniform(0.5, 1.5)
            for i in range(3):
                img_np[:, :, i] = gaussian_filter(img_np[:, :, i], sigma=sigma)

        # Clip and convert back to uint8
        img_np = np.clip(img_np, 0, 1) * 255
        img_np = img_np.astype(np.uint8)

        return Image.fromarray(img_np)


if __name__ == "__main__":
    # Example usage

    # Create datasets
    data_dir = "/path/to/sea_turtle_dataset"

    # Standard dataset
    standard_dataset = TurtleDataset(data_dir, "train")
    print(f"Standard dataset size: {len(standard_dataset)}")

    # Siamese dataset
    siamese_dataset = SiamesePairDataset(data_dir, "train")
    print(f"Siamese dataset size: {len(siamese_dataset)}")

    # Triplet dataset
    triplet_dataset = TripletDataset(data_dir, "train")
    print(f"Triplet dataset size: {len(triplet_dataset)}")

    # Online triplet dataset
    online_dataset = OnlineTripletDataset(data_dir, "train")
    print(f"Online triplet dataset size: {len(online_dataset)}")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir, batch_size=32, dataset_type="triplet"
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
