# Week 2: Transfer Learning and Fine-Tuning Advanced CNNs

Welcome to Week 2 of our Sea Turtle Re-Identification System! This week builds on the baseline CNN from Week 1 and introduces transfer learning to significantly boost performance.

## Folder Structure

```
week2/
├── README.md                     # This guide
├── transfer_learning.py          # Main script for transfer learning
├── fine_tuning.py                # Incremental fine-tuning script
├── models/
│   ├── model_architectures.py    # Different model architectures to try
│   └── model_utils.py            # Helper functions for models
├── compare_architectures.py      # Experiment script to compare different backbones
└── utils/
    ├── visualization.py          # Visualization utilities (extended from Week 1)
    └── training.py               # Training helpers, learning rate scheduling, etc.
```

## Week 2 Overview

This week focuses on:

1. Using pre-trained models as feature extractors
2. Fine-tuning strategies for transfer learning
3. Comparing different model architectures (ResNet, EfficientNet)
4. Implementing best practices for training deep networks

## Exercise 1: Transfer Learning with Frozen Features

In this exercise, you'll leverage a pre-trained CNN as a fixed feature extractor, adding only a new classification layer.

### Running the Script

```bash
python transfer_learning.py --data_dir path/to/dataset --model resnet50 --freeze_backbone --output_dir ./models
```

### Key Concepts:

- **Pre-trained Models**: Using models already trained on ImageNet, which have learned useful feature representations
- **Feature Extraction**: Keeping the pre-trained layers frozen while training only new layers
- **Initialization Strategy**: Proper initialization of the new classification layer

### Implementation Details:

The script lets you choose from several pre-trained models:

- `resnet18`, `resnet50` (good balance of speed/accuracy)
- `efficientnet_b0`, `efficientnet_b3` (more efficient architectures)
- `mobilenet_v2` (lightweight, for potential deployment)

```python
# Example of loading a pre-trained model with frozen weights
def create_model(model_name, num_classes, freeze_backbone=True):
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)

        # Freeze all parameters
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False

        # Replace the final fully connected layer
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

    # Similar for other architectures...

    return model
```

## Exercise 2: Progressive Fine-tuning

This exercise demonstrates how to gradually unfreeze and fine-tune layers of a pre-trained network.

### Running the Script

```bash
python fine_tuning.py --data_dir path/to/dataset --model efficientnet_b0 --learning_rate 0.0001 --output_dir ./models
```

### Key Concepts:

- **Layer-by-Layer Unfreezing**: Starting with the final layers and gradually unfreezing earlier layers
- **Layer Groups**: Understanding how to group layers (e.g., ResNet blocks, EfficientNet stages)
- **Lower Learning Rates**: Using smaller learning rates for pre-trained weights
- **Layer-wise Learning Rates**: Optionally applying different learning rates to different parts of the network

### Implementation Example:

```python
# Progressive unfreezing example for ResNet
def unfreeze_model_layers(model, unfreeze_layers):
    """
    Unfreeze selected layers of a model.

    Args:
        model: The model to unfreeze layers from
        unfreeze_layers: List of layer names to unfreeze
    """
    # First freeze all parameters
    for name, param in model.named_parameters():
        param.requires_grad = False

    # Then unfreeze the specified layers
    for name, param in model.named_parameters():
        for layer_name in unfreeze_layers:
            if layer_name in name:
                param.requires_grad = True
                break
```

## Exercise 3: Comparing Model Architectures

This exercise helps you compare different model architectures to find the best backbone for turtle re-identification.

### Running the Script

```bash
python compare_architectures.py --data_dir path/to/dataset --output_dir ./comparisons
```

### Key Concepts:

- **Architecture Comparison**: Evaluating trade-offs between different models
- **Performance Metrics**: Accuracy, training time, and model size
- **Decision Factors**: Understanding when to choose different architectures

### Models to Compare:

The script evaluates several architectures:

- `ResNet-18`: Lightweight, fast to train
- `ResNet-50`: More powerful, still reasonable training time
- `EfficientNet-B0`: Very efficient, good for deployment
- `EfficientNet-B3`: More powerful but larger

### Example Output:

```
Model Comparison Results:
--------------------------
ResNet-18:
  - Parameters: 11.7M
  - Training time: 45 minutes
  - Validation accuracy: 87.2%

ResNet-50:
  - Parameters: 25.6M
  - Training time: 92 minutes
  - Validation accuracy: 91.3%

EfficientNet-B0:
  - Parameters: 5.3M
  - Training time: 63 minutes
  - Validation accuracy: 89.5%

EfficientNet-B3:
  - Parameters: 12.2M
  - Training time: 108 minutes
  - Validation accuracy: 93.1%
```

## Exercise 4: Training Best Practices

This exercise demonstrates best practices for training deep neural networks effectively.

### Key Concepts:

- **Learning Rate Scheduling**: Implementing learning rate warmup, step decay, and cosine annealing
- **Early Stopping**: Preventing overfitting by monitoring validation performance
- **Regularization**: Applying techniques like weight decay and dropout
- **Mixed Precision Training**: Using FP16 for faster training on supported hardware

### Example Schedule Implementation:

```python
# Learning rate scheduler with warmup and cosine annealing
def create_lr_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Warmup phase: linear increase
            return epoch / warmup_epochs
        else:
            # Cosine annealing
            return 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

## Expected Outcome

By the end of this week, you should have:

1. A significantly more accurate model than your Week 1 baseline
2. Understanding of transfer learning and fine-tuning techniques
3. Data on which architectures work best for turtle re-identification
4. A well-trained model to use as a foundation for Week 3's metric learning

## Tips for Success

- **Start Simple**: Begin with a frozen backbone before trying fine-tuning
- **Resource Management**: If GPU memory is limited, use smaller batch sizes or smaller models
- **Be Patient**: Fine-tuning can take time; EfficientNet models are often slower to train but more accurate
- **Save Checkpoints**: Save intermediate models in case training crashes
- **Monitor Hardware**: Watch GPU memory usage and temperature during longer training runs

Next week, we'll move beyond classification and explore metric learning approaches to develop a system that can compare turtle identities!
