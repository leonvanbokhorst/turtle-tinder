#!/usr/bin/env python3
# Sea Turtle Re-Identification - Week 1
# Baseline CNN Model

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineCNN(nn.Module):
    """
    A simple baseline CNN architecture for turtle identification.

    This model consists of 4 convolutional blocks followed by
    a classification head. Each block has:
    - Convolutional layer
    - Batch normalization
    - ReLU activation
    - Max pooling

    The classification head has:
    - Dropout for regularization
    - Two fully connected layers
    """

    def __init__(self, num_classes, input_channels=3):
        """
        Initialize the baseline CNN model.

        Args:
            num_classes: Number of turtle IDs to classify
            input_channels: Number of input channels (3 for RGB images)
        """
        super(BaselineCNN, self).__init__()

        # Feature extractor
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Fourth convolutional block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Calculate size after convolutions and pooling
        # Input: 224x224 -> After 4 max-pooling (2x2): 14x14
        self.feature_size = 256 * 14 * 14

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
               Expected size is [batch_size, 3, 224, 224]

        Returns:
            Logits for each class, shape [batch_size, num_classes]
        """
        # Extract features
        x = self.features(x)

        # Flatten
        x = torch.flatten(x, 1)

        # Classification
        x = self.classifier(x)

        return x


class TurtleIDSmallCNN(nn.Module):
    """
    An even smaller CNN architecture that might be useful for quick experiments
    or limited computational resources.
    """

    def __init__(self, num_classes, input_channels=3):
        super(TurtleIDSmallCNN, self).__init__()

        # Feature extraction
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # 224x224 input -> 28x28 after three strides of 2
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        # Flatten and classify
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# Example usage:
if __name__ == "__main__":
    # Create a model with 10 classes (10 different turtles)
    model = BaselineCNN(num_classes=10)

    # Print model architecture and parameters
    print(model)

    # Calculate number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Test forward pass with a random input
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    output = model(input_tensor)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")

    # Also demonstrate the smaller model
    small_model = TurtleIDSmallCNN(num_classes=10)
    small_output = small_model(input_tensor)
    small_params = sum(p.numel() for p in small_model.parameters())

    print(f"\nSmall model parameters: {small_params:,}")
    print(f"Small model output shape: {small_output.shape}")
