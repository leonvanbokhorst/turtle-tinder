#!/usr/bin/env python
"""
Ensemble inference module for sea turtle re-identification.

This script provides functionality to perform inference using multiple models
and combine their predictions for improved accuracy. It implements various
ensemble techniques like averaging, weighted averaging, and majority voting.
"""

import os
import sys
import json
import argparse
import time
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    confusion_matrix,
    classification_report,
)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.model_loader import load_model
from utils.data_utils import load_dataset, prepare_dataloaders


class EnsembleInference:
    """
    Class for performing inference with an ensemble of models.
    """

    def __init__(
        self,
        model_paths: List[str],
        model_weights: Optional[List[float]] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the ensemble inference.

        Args:
            model_paths: List of paths to model checkpoints
            model_weights: Optional weights for each model (must sum to 1)
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
        """
        self.model_paths = model_paths
        self.num_models = len(model_paths)

        # Set up device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Set up model weights for weighted averaging
        if model_weights is None:
            # Equal weights
            self.model_weights = [1.0 / self.num_models] * self.num_models
        else:
            # Validate weights
            if len(model_weights) != self.num_models:
                raise ValueError(
                    f"Number of weights ({len(model_weights)}) must match number of models ({self.num_models})"
                )
            if abs(sum(model_weights) - 1.0) > 1e-5:
                raise ValueError(
                    f"Model weights must sum to 1, got {sum(model_weights)}"
                )

            self.model_weights = model_weights

        # Load models
        self.models = []
        self.model_info = []

        print(f"Loading {self.num_models} models...")
        for i, path in enumerate(model_paths):
            model, info = load_model(path, self.device)
            self.models.append(model)
            self.model_info.append(info)
            print(
                f"  Model {i+1}: {info['model_type']} (weight: {self.model_weights[i]:.3f})"
            )

        # Check if models are compatible for ensembling
        self._validate_models()

    def _validate_models(self):
        """
        Validate that models are compatible for ensembling.
        """
        # Check if all models have the same number of classes
        num_classes = self.model_info[0]["num_classes"]
        for i, info in enumerate(self.model_info[1:], 1):
            if info["num_classes"] != num_classes:
                raise ValueError(
                    f"Models have different number of classes: "
                    f"{self.model_info[0]['num_classes']} vs {info['num_classes']}"
                )

        self.num_classes = num_classes
        print(f"All models have {self.num_classes} classes")

    def predict_single(
        self,
        image: torch.Tensor,
        method: str = "weighted_avg",
        return_all: bool = False,
    ) -> Union[Tuple[int, float], Tuple[np.ndarray, List[Tuple[int, float]]]]:
        """
        Predict class for a single image using the ensemble.

        Args:
            image: Input image tensor [1, C, H, W]
            method: Ensemble method ('avg', 'weighted_avg', 'max_confidence')
            return_all: Whether to return all model predictions

        Returns:
            If return_all=False:
                Tuple of (predicted_class, confidence)
            If return_all=True:
                Tuple of (ensemble_probs, list of (predicted_class, confidence) for each model)
        """
        # Ensure image has batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)

        # Move to device
        image = image.to(self.device)

        # Get predictions from each model
        probs_list = []
        predictions = []

        with torch.no_grad():
            for i, model in enumerate(self.models):
                # Set model to evaluation mode
                model.eval()

                # Get model output (depends on model type)
                if hasattr(model, "get_embedding"):
                    # For feature-based models, extract features and perform inference
                    embeddings = model.get_embedding(image)
                    # Additional logic for classification from embeddings would go here
                    logits = model.classify(
                        embeddings
                    )  # Assuming model has classify method
                else:
                    # For classification models, forward pass directly
                    logits = model(image)

                # Convert to probabilities
                probs = F.softmax(logits, dim=1)
                probs_list.append(probs.cpu().numpy())

                # Get prediction and confidence
                pred_class = torch.argmax(probs, dim=1).item()
                confidence = probs[0, pred_class].item()
                predictions.append((pred_class, confidence))

        # Combine predictions based on method
        probs_array = np.array(probs_list)

        if method == "avg":
            # Simple averaging
            ensemble_probs = np.mean(probs_array, axis=0)
        elif method == "weighted_avg":
            # Weighted averaging
            ensemble_probs = np.average(probs_array, axis=0, weights=self.model_weights)
        elif method == "max_confidence":
            # Take prediction with highest confidence
            ensemble_probs = probs_array[np.argmax([p[1] for p in predictions])]
        else:
            raise ValueError(f"Unknown ensemble method: {method}")

        # Get ensemble prediction
        ensemble_class = np.argmax(ensemble_probs)
        ensemble_confidence = ensemble_probs[0, ensemble_class]

        if return_all:
            return ensemble_probs, predictions
        else:
            return ensemble_class, ensemble_confidence

    def predict_batch(
        self, dataloader: DataLoader, method: str = "weighted_avg"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict classes for a batch of images using the ensemble.

        Args:
            dataloader: DataLoader for images
            method: Ensemble method ('avg', 'weighted_avg', 'max_confidence')

        Returns:
            Tuple of (y_true, y_pred, confidences)
        """
        y_true = []
        y_pred = []
        confidences = []

        # Store predictions from each model
        all_model_preds = [[] for _ in range(self.num_models)]

        for images, labels in tqdm(dataloader, desc="Ensemble inference"):
            # Move to device
            images = images.to(self.device)

            # Get predictions from each model
            batch_probs_list = []

            with torch.no_grad():
                for i, model in enumerate(self.models):
                    # Set model to evaluation mode
                    model.eval()

                    # Get model output
                    if hasattr(model, "get_embedding"):
                        embeddings = model.get_embedding(images)
                        logits = model.classify(embeddings)
                    else:
                        logits = model(images)

                    # Convert to probabilities
                    probs = F.softmax(logits, dim=1)
                    batch_probs_list.append(probs.cpu().numpy())

                    # Store individual model predictions
                    model_preds = torch.argmax(probs, dim=1).cpu().numpy()
                    all_model_preds[i].extend(model_preds)

            # Combine predictions
            batch_probs_array = np.array(batch_probs_list)

            if method == "avg":
                ensemble_probs = np.mean(batch_probs_array, axis=0)
            elif method == "weighted_avg":
                ensemble_probs = np.average(
                    batch_probs_array, axis=0, weights=self.model_weights
                )
            elif method == "max_confidence":
                # Take prediction with highest confidence for each sample
                max_indices = np.argmax(np.max(batch_probs_array, axis=2), axis=0)
                ensemble_probs = np.zeros_like(batch_probs_array[0])
                for i in range(len(max_indices)):
                    ensemble_probs[i] = batch_probs_array[max_indices[i], i]
            else:
                raise ValueError(f"Unknown ensemble method: {method}")

            # Get ensemble predictions and confidences
            batch_preds = np.argmax(ensemble_probs, axis=1)
            batch_confidences = np.max(ensemble_probs, axis=1)

            # Store results
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(batch_preds)
            confidences.extend(batch_confidences)

        # Calculate model agreement
        agreement = self._calculate_model_agreement(all_model_preds, y_true)
        print(f"Model agreement: {agreement:.2f}%")

        return np.array(y_true), np.array(y_pred), np.array(confidences)

    def _calculate_model_agreement(
        self, model_predictions: List[List[int]], ground_truth: List[int]
    ) -> float:
        """
        Calculate the percentage of samples where all models agree on the prediction.

        Args:
            model_predictions: List of predictions for each model
            ground_truth: Ground truth labels

        Returns:
            Agreement percentage
        """
        num_samples = len(ground_truth)
        all_agree = 0
        correct_when_agree = 0

        # Convert to numpy for easier manipulation
        preds_array = np.array(model_predictions)

        for i in range(num_samples):
            # Check if all models predict the same class
            if len(set(preds_array[:, i])) == 1:
                all_agree += 1
                # Check if the agreed prediction is correct
                if preds_array[0, i] == ground_truth[i]:
                    correct_when_agree += 1

        agreement_pct = 100 * all_agree / num_samples
        if all_agree > 0:
            accuracy_when_agree = 100 * correct_when_agree / all_agree
            print(f"Accuracy when all models agree: {accuracy_when_agree:.2f}%")

        return agreement_pct


def evaluate_ensemble(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidences: np.ndarray,
    class_names: List[str],
    output_dir: str,
) -> Dict[str, float]:
    """
    Evaluate ensemble performance and generate visualizations.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        confidences: Prediction confidences
        class_names: List of class names
        output_dir: Directory to save visualizations

    Returns:
        Dictionary of evaluation metrics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Convert labels to one-hot encoding for precision-recall curves
    n_classes = len(class_names)
    y_true_onehot = np.zeros((len(y_true), n_classes))
    for i, label in enumerate(y_true):
        y_true_onehot[i, label] = 1

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot and save confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    # Generate classification report
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )

    # Save report as JSON
    with open(os.path.join(output_dir, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=4)

    # Plot confidence distribution
    plt.figure(figsize=(10, 6))
    plt.hist(confidences, bins=20, alpha=0.7)
    plt.title("Confidence Distribution")
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "confidence_distribution.png"), dpi=300)
    plt.close()

    # Plot confidence for correct vs. incorrect predictions
    correct = y_pred == y_true
    plt.figure(figsize=(10, 6))
    plt.hist(
        confidences[correct],
        bins=20,
        alpha=0.7,
        label="Correct predictions",
        color="green",
    )
    plt.hist(
        confidences[~correct],
        bins=20,
        alpha=0.7,
        label="Incorrect predictions",
        color="red",
    )
    plt.title("Confidence Distribution: Correct vs. Incorrect Predictions")
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "confidence_by_correctness.png"), dpi=300)
    plt.close()

    # Calculate and return metrics
    accuracy = np.mean(y_pred == y_true)
    avg_confidence = np.mean(confidences)
    avg_confidence_correct = np.mean(confidences[correct]) if np.sum(correct) > 0 else 0
    avg_confidence_incorrect = (
        np.mean(confidences[~correct]) if np.sum(~correct) > 0 else 0
    )

    metrics = {
        "accuracy": accuracy,
        "avg_confidence": avg_confidence,
        "avg_confidence_correct": avg_confidence_correct,
        "avg_confidence_incorrect": avg_confidence_incorrect,
        "model_calibration": avg_confidence_correct - avg_confidence_incorrect,
    }

    # Save metrics
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics


def compare_ensemble_methods(
    ensemble: EnsembleInference,
    dataloader: DataLoader,
    class_names: List[str],
    output_dir: str,
) -> Dict[str, Dict[str, float]]:
    """
    Compare different ensemble methods on the same data.

    Args:
        ensemble: EnsembleInference object
        dataloader: DataLoader for evaluation
        class_names: List of class names
        output_dir: Directory to save results

    Returns:
        Dictionary of metrics for each method
    """
    methods = ["avg", "weighted_avg", "max_confidence"]
    results = {}

    for method in methods:
        print(f"\nEvaluating ensemble method: {method}")

        # Predict with current method
        y_true, y_pred, confidences = ensemble.predict_batch(dataloader, method=method)

        # Evaluate and save results
        method_output_dir = os.path.join(output_dir, f"method_{method}")
        metrics = evaluate_ensemble(
            y_true, y_pred, confidences, class_names, method_output_dir
        )

        results[method] = metrics
        print(f"Accuracy ({method}): {metrics['accuracy']:.4f}")

    # Compare methods
    method_comparison = {
        "method": list(results.keys()),
        "accuracy": [results[m]["accuracy"] for m in results],
        "avg_confidence": [results[m]["avg_confidence"] for m in results],
        "model_calibration": [results[m]["model_calibration"] for m in results],
    }

    # Plot method comparison
    plt.figure(figsize=(12, 6))
    x = np.arange(len(methods))
    width = 0.25

    plt.bar(x - width, method_comparison["accuracy"], width, label="Accuracy")
    plt.bar(x, method_comparison["avg_confidence"], width, label="Avg Confidence")
    plt.bar(
        x + width, method_comparison["model_calibration"], width, label="Calibration"
    )

    plt.xlabel("Ensemble Method")
    plt.ylabel("Score")
    plt.title("Comparison of Ensemble Methods")
    plt.xticks(x, methods)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "method_comparison.png"), dpi=300)
    plt.close()

    # Save comparison results
    with open(os.path.join(output_dir, "method_comparison.json"), "w") as f:
        json.dump(results, f, indent=4)

    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Ensemble inference for sea turtle re-identification"
    )

    # Required arguments
    parser.add_argument(
        "--models_dir",
        type=str,
        required=True,
        help="Directory containing model checkpoints",
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Directory containing test data"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save results"
    )

    # Optional arguments
    parser.add_argument(
        "--model_pattern",
        type=str,
        default="*.pth",
        help="Pattern to match model files (default: *.pth)",
    )
    parser.add_argument(
        "--weights_file",
        type=str,
        default=None,
        help="JSON file containing model weights",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for inference (default: auto)",
    )
    parser.add_argument(
        "--ensemble_method",
        type=str,
        default="weighted_avg",
        choices=["avg", "weighted_avg", "max_confidence"],
        help="Method for combining predictions (default: weighted_avg)",
    )
    parser.add_argument(
        "--compare_methods",
        action="store_true",
        help="Compare different ensemble methods",
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Find model files
    import glob

    model_paths = sorted(glob.glob(os.path.join(args.models_dir, args.model_pattern)))

    if not model_paths:
        raise ValueError(
            f"No model files found in {args.models_dir} matching {args.model_pattern}"
        )

    print(f"Found {len(model_paths)} model files")

    # Load model weights if provided
    model_weights = None
    if args.weights_file is not None:
        with open(args.weights_file, "r") as f:
            weights_data = json.load(f)

        if isinstance(weights_data, dict):
            # Map weights to model paths
            model_weights = [
                weights_data.get(os.path.basename(path), 1.0) for path in model_paths
            ]
        elif isinstance(weights_data, list):
            # Direct list of weights
            model_weights = weights_data

        # Normalize weights
        if model_weights:
            weight_sum = sum(model_weights)
            model_weights = [w / weight_sum for w in model_weights]

    # Create ensemble
    ensemble = EnsembleInference(
        model_paths=model_paths, model_weights=model_weights, device=args.device
    )

    # Load test data
    test_loader, class_names = prepare_dataloaders(
        data_dir=args.data_dir, split="test", batch_size=args.batch_size
    )

    print(
        f"Loaded test data with {len(test_loader.dataset)} samples and {len(class_names)} classes"
    )

    # Save configuration
    config = {
        "models": model_paths,
        "weights": ensemble.model_weights,
        "ensemble_method": args.ensemble_method,
        "num_test_samples": len(test_loader.dataset),
        "num_classes": len(class_names),
        "class_names": class_names,
    }

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        # Convert paths to relative
        config["models"] = [os.path.basename(p) for p in config["models"]]
        json.dump(config, f, indent=4)

    # Evaluate ensemble
    if args.compare_methods:
        # Compare different ensemble methods
        results = compare_ensemble_methods(
            ensemble=ensemble,
            dataloader=test_loader,
            class_names=class_names,
            output_dir=args.output_dir,
        )

        # Print comparison
        print("\nEnsemble Method Comparison:")
        for method, metrics in results.items():
            print(
                f"  {method}: Accuracy={metrics['accuracy']:.4f}, "
                f"Avg Confidence={metrics['avg_confidence']:.4f}, "
                f"Calibration={metrics['model_calibration']:.4f}"
            )
    else:
        # Evaluate with the specified method
        print(f"\nEvaluating ensemble with method: {args.ensemble_method}")
        y_true, y_pred, confidences = ensemble.predict_batch(
            test_loader, method=args.ensemble_method
        )

        metrics = evaluate_ensemble(
            y_true=y_true,
            y_pred=y_pred,
            confidences=confidences,
            class_names=class_names,
            output_dir=args.output_dir,
        )

        # Print results
        print("\nEnsemble Performance:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Average Confidence: {metrics['avg_confidence']:.4f}")
        print(f"  Model Calibration: {metrics['model_calibration']:.4f}")

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
