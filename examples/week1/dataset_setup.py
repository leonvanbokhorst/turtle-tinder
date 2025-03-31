#!/usr/bin/env python3
# Sea Turtle Re-Identification - Week 1
# Dataset Setup Script

import os
import argparse
import random
import shutil
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Sea Turtle Dataset Setup")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to directory containing turtle images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./processed_dataset",
        help="Path to output the processed dataset",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Ratio of images to use for training",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Ratio of images to use for validation",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    return parser.parse_args()


def create_dataset_structure(output_dir):
    """Create the necessary directory structure for the dataset."""
    os.makedirs(output_dir, exist_ok=True)

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)


def collect_images(data_dir):
    """
    Collect all image paths and organize them by turtle ID.

    Assumes each turtle's images are in a separate directory named with the turtle's ID.
    """
    turtle_images = defaultdict(list)

    # List all directories in the data_dir (each should be a turtle ID)
    turtle_ids = [
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    ]

    print(f"Found {len(turtle_ids)} unique turtle IDs")

    for turtle_id in turtle_ids:
        turtle_dir = os.path.join(data_dir, turtle_id)

        # List all image files for this turtle
        image_files = [
            f
            for f in os.listdir(turtle_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        for img_file in image_files:
            img_path = os.path.join(turtle_dir, img_file)
            turtle_images[turtle_id].append(img_path)

    return turtle_images


def split_dataset(turtle_images, train_ratio, val_ratio, seed=42):
    """
    Split the dataset into train, validation, and test sets.

    For each turtle, we split its images according to the specified ratios.
    This ensures each set has examples of all turtles.
    """
    random.seed(seed)

    train_images = []
    val_images = []
    test_images = []

    # Calculate test ratio
    test_ratio = 1.0 - train_ratio - val_ratio

    for turtle_id, image_paths in turtle_images.items():
        # Shuffle the images for this turtle
        random.shuffle(image_paths)

        # Calculate split indices
        n_images = len(image_paths)
        n_train = int(n_images * train_ratio)
        n_val = int(n_images * val_ratio)

        # Split the images
        train_subset = image_paths[:n_train]
        val_subset = image_paths[n_train : n_train + n_val]
        test_subset = image_paths[n_train + n_val :]

        # Add to the corresponding sets with turtle ID
        train_images.extend([(path, turtle_id) for path in train_subset])
        val_images.extend([(path, turtle_id) for path in val_subset])
        test_images.extend([(path, turtle_id) for path in test_subset])

    return train_images, val_images, test_images


def copy_images_to_split(images, output_dir, split):
    """
    Copy images to the appropriate split directory.

    Args:
        images: List of (image_path, turtle_id) tuples
        output_dir: Base output directory
        split: Split name ('train', 'val', or 'test')
    """
    split_dir = os.path.join(output_dir, split)

    # Ensure the split directory exists
    os.makedirs(split_dir, exist_ok=True)

    # Create a directory for each turtle ID in this split
    turtle_ids = set(turtle_id for _, turtle_id in images)
    for turtle_id in turtle_ids:
        os.makedirs(os.path.join(split_dir, turtle_id), exist_ok=True)

    # Copy each image to its appropriate location
    for img_path, turtle_id in images:
        # Get the filename of the image
        filename = os.path.basename(img_path)

        # Define the destination path
        dest_path = os.path.join(split_dir, turtle_id, filename)

        # Copy the image
        shutil.copy2(img_path, dest_path)

    print(f"Copied {len(images)} images to {split} set")


def visualize_dataset_stats(turtle_images, output_dir):
    """
    Create visualizations of the dataset statistics.
    """
    # 1. Count images per turtle
    turtle_counts = {
        turtle_id: len(images) for turtle_id, images in turtle_images.items()
    }

    # 2. Create a histogram of images per turtle
    plt.figure(figsize=(12, 6))
    plt.bar(turtle_counts.keys(), turtle_counts.values())
    plt.xticks(rotation=90)
    plt.xlabel("Turtle ID")
    plt.ylabel("Number of Images")
    plt.title("Number of Images per Turtle")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "images_per_turtle.png"))

    # 3. Create a summary dataframe
    summary = pd.DataFrame(
        list(turtle_counts.items()), columns=["TurtleID", "ImageCount"]
    )
    summary = summary.sort_values("ImageCount", ascending=False)

    # 4. Save summary statistics
    with open(os.path.join(output_dir, "dataset_stats.txt"), "w") as f:
        f.write(f"Total number of turtles: {len(turtle_images)}\n")
        f.write(f"Total number of images: {sum(turtle_counts.values())}\n")
        f.write(
            f"Average images per turtle: {sum(turtle_counts.values()) / len(turtle_images):.2f}\n"
        )
        f.write(f"Minimum images for a turtle: {min(turtle_counts.values())}\n")
        f.write(f"Maximum images for a turtle: {max(turtle_counts.values())}\n")
        f.write("\nTop 10 turtles by image count:\n")
        for i, (turtle_id, count) in enumerate(
            summary.iloc[:10].itertuples(index=False)
        ):
            f.write(f"{i+1}. Turtle {turtle_id}: {count} images\n")

    print(f"Saved dataset statistics to {output_dir}")


def display_sample_images(turtle_images, output_dir, num_turtles=5, num_images=4):
    """
    Display sample images from random turtles.
    """
    # Select random turtles
    turtle_ids = list(turtle_images.keys())
    if len(turtle_ids) > num_turtles:
        selected_turtles = random.sample(turtle_ids, num_turtles)
    else:
        selected_turtles = turtle_ids

    # Create a figure
    fig, axes = plt.subplots(num_turtles, num_images, figsize=(15, 3 * num_turtles))

    # Display images for each selected turtle
    for i, turtle_id in enumerate(selected_turtles):
        images = turtle_images[turtle_id]

        # Select random images for this turtle
        if len(images) > num_images:
            selected_images = random.sample(images, num_images)
        else:
            selected_images = images

        # Display each selected image
        for j, img_path in enumerate(selected_images):
            try:
                img = Image.open(img_path)
                if i < len(selected_turtles) and j < num_images:
                    if num_turtles > 1:
                        ax = axes[i, j]
                    else:
                        ax = axes[j]
                    ax.imshow(img)
                    ax.set_title(f"Turtle {turtle_id}")
                    ax.axis("off")
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sample_images.png"))
    print(f"Saved sample images to {output_dir}")


def main():
    args = parse_args()

    # Create the output directory structure
    create_dataset_structure(args.output_dir)

    # Collect all turtle images
    turtle_images = collect_images(args.data_dir)

    if not turtle_images:
        print("No turtle images found! Please check your data directory structure.")
        return

    # Visualize dataset statistics
    visualize_dataset_stats(turtle_images, args.output_dir)

    # Display sample images
    display_sample_images(turtle_images, args.output_dir)

    # Split the dataset
    train_images, val_images, test_images = split_dataset(
        turtle_images, args.train_ratio, args.val_ratio, args.seed
    )

    # Copy images to their respective splits
    copy_images_to_split(train_images, args.output_dir, "train")
    copy_images_to_split(val_images, args.output_dir, "val")
    copy_images_to_split(test_images, args.output_dir, "test")

    print("Dataset setup complete!")
    print(f"Training set: {len(train_images)} images")
    print(f"Validation set: {len(val_images)} images")
    print(f"Test set: {len(test_images)} images")


if __name__ == "__main__":
    main()
