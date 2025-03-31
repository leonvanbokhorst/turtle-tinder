# Week 1: Data Preparation and Baseline CNN Models

Welcome to Week 1 of our Sea Turtle Re-Identification System! This week focuses on:

- Understanding the problem domain
- Setting up and exploring your dataset
- Implementing preprocessing and augmentation pipelines
- Building a baseline CNN classifier

## Folder Structure

```
week1/
├── README.md                   # This guide
├── dataset_setup.py            # Script for organizing and exploring the dataset
├── preprocessing_pipeline.py   # Image preprocessing and augmentation demo
├── baseline_cnn.py             # Simple CNN model implementation
├── train_baseline.py           # Training script for the baseline model
└── utils/
    ├── visualization.py        # Helper functions for visualizing images/results
    └── data_loader.py          # Data loading utilities
```

## Exercise 1: Dataset Setup

In this exercise, you'll organize your turtle images and explore their characteristics.

### Running the Script

```bash
python dataset_setup.py --data_dir /path/to/your/turtle/images --output_dir ./processed_dataset
```

The script will:

1. Create a structured dataset with train/val/test splits
2. Generate statistics about image count per turtle ID
3. Display sample images from different individuals

### Key Concepts:

- **Dataset Structure**: Images organized by individual turtle ID
- **Train/Val/Test Split**: Properly stratified to ensure all individuals appear in all splits
- **Data Exploration**: Understanding distribution and characteristics of your dataset

## Exercise 2: Preprocessing Pipeline

This exercise helps you create a robust pipeline for preprocessing turtle images and applying augmentations.

### Running the Script

```bash
python preprocessing_pipeline.py --image_path examples/sample_turtle.jpg
```

### Key Concepts:

- **Image Normalization**: Scaling pixel values and standardizing dimensions
- **Data Augmentation**: Simulating underwater conditions like:
  - Brightness/contrast changes (underwater lighting)
  - Color distortion (blue/green shifts in water)
  - Rotation/flips (different viewing angles)
  - Noise and blur (turbidity and water effects)

### Example Augmentations:

```python
# Key augmentations for underwater turtle images
transforms = A.Compose([
    A.RandomRotate90(p=0.5),
    A.Flip(p=0.5),
    A.OneOf([
        A.MotionBlur(blur_limit=7, p=0.5),
        A.GaussianBlur(blur_limit=7, p=0.5),
    ], p=0.7),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.8),
    A.OneOf([
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
    ], p=0.7),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
    A.Resize(height=224, width=224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

## Exercise 3: Build and Train a CNN

In this exercise, you'll implement a simple CNN classifier as your baseline.

### Running the Script

```bash
python train_baseline.py --data_dir ./processed_dataset --epochs 10 --batch_size 32 --output_dir ./models
```

### Key Concepts:

- **CNN Architecture**: Basic convolutional neural network design
- **Training Loop**: Including learning rate, optimizer, and loss function
- **Evaluation**: Monitoring accuracy and loss on validation set

### Example CNN Architecture:

```python
class BaselineCNN(nn.Module):
    def __init__(self, num_classes):
        super(BaselineCNN, self).__init__()
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
```

## Expected Outcome

By the end of this week, you should have:

1. A well-organized dataset with proper train/validation/test splits
2. A visualization of your data with statistics on individual turtles
3. A preprocessing pipeline with augmentations specific to underwater turtle images
4. A baseline CNN model trained to classify individual turtles
5. Quantitative results (accuracy, loss curves) to serve as a benchmark

## Tips for Success

- Start with a manageable subset if your dataset is very large
- Visualize augmentations to ensure they're realistic for underwater conditions
- Monitor training vs. validation accuracy to detect overfitting
- Save your model checkpoints regularly
- Document your baseline performance metrics for comparison in Week 2

Next week, we'll improve on this baseline using transfer learning and fine-tuning pretrained models!
