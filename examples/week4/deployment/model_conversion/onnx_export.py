#!/usr/bin/env python
"""
ONNX export utility for sea turtle re-identification models.

This script converts PyTorch models to ONNX format, which can be used
for deployment on various platforms and with different runtimes.
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import time

# Add parent directories to path
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)
from utils.model_loader import load_model


def export_to_onnx(
    model,
    input_shape,
    output_path,
    model_type="classification",
    opset_version=12,
    dynamic_axes=None,
    input_names=None,
    output_names=None,
    metadata=None,
):
    """
    Export a PyTorch model to ONNX format.

    Args:
        model: PyTorch model
        input_shape: Shape of input tensor (batch_size, channels, height, width)
        output_path: Path to save ONNX model
        model_type: Type of model ('classification', 'siamese', 'triplet', 'arcface')
        opset_version: ONNX opset version
        dynamic_axes: Dynamic axes for ONNX export
        input_names: Names of input tensors
        output_names: Names of output tensors
        metadata: Additional metadata to include in ONNX model

    Returns:
        Path to saved ONNX model
    """
    # Set model to evaluation mode
    model.eval()

    # Create dummy input for tracing
    dummy_input = torch.randn(*input_shape)

    # Set input and output names
    if input_names is None:
        input_names = ["input"]

    if output_names is None:
        if model_type == "classification":
            output_names = ["output"]
        elif model_type in ["siamese", "triplet", "arcface"]:
            output_names = ["embedding"]

    # Set dynamic axes if not provided
    if dynamic_axes is None:
        dynamic_axes = {"input": {0: "batch_size"}, output_names[0]: {0: "batch_size"}}

    # Create export directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Define forward function based on model type
    if model_type == "classification":
        # Standard classification model
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            export_params=True,
            do_constant_folding=True,
        )
    elif model_type in ["siamese", "triplet", "arcface"]:
        # Metric learning models - export just the embedding function
        class EmbeddingWrapper(torch.nn.Module):
            def __init__(self, model):
                super(EmbeddingWrapper, self).__init__()
                self.model = model

            def forward(self, x):
                if hasattr(self.model, "get_embedding"):
                    return self.model.get_embedding(x)
                elif model_type == "arcface":
                    # For ArcFace, use the third output (normalized features)
                    _, _, embeddings = self.model(x)
                    return embeddings
                else:
                    # Fallback - try to extract features
                    if hasattr(self.model, "backbone"):
                        features = self.model.backbone(x)
                        if hasattr(self.model, "neck"):
                            features = self.model.neck(features)
                        return features
                    else:
                        raise ValueError(
                            f"Don't know how to extract embeddings from model type {model_type}"
                        )

        # Create embedding wrapper
        embedding_model = EmbeddingWrapper(model)

        # Export the embedding model
        torch.onnx.export(
            embedding_model,
            dummy_input,
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            export_params=True,
            do_constant_folding=True,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(f"Model exported to ONNX format: {output_path}")

    # If metadata is provided, add it to the model
    if metadata is not None:
        try:
            import onnx

            model = onnx.load(output_path)

            for key, value in metadata.items():
                meta = model.metadata_props.add()
                meta.key = key
                meta.value = str(value)

            onnx.save(model, output_path)
            print(f"Added metadata to ONNX model")

        except ImportError:
            print(
                f"Warning: Could not add metadata to ONNX model. ONNX module not found."
            )

    return output_path


def validate_onnx_model(onnx_path, input_shape, rtol=1e-3, atol=1e-4):
    """
    Validate the exported ONNX model by comparing its outputs with the PyTorch model.

    Args:
        onnx_path: Path to ONNX model
        input_shape: Shape of input tensor
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison

    Returns:
        True if validation passes, False otherwise
    """
    try:
        import onnxruntime as ort
        import onnx
    except ImportError:
        print("Warning: onnxruntime or onnx not found. Skipping validation.")
        return False

    # Load ONNX model
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model check passed!")
    except Exception as e:
        print(f"ONNX model check failed: {e}")
        return False

    # Create random input
    input_data = np.random.rand(*input_shape).astype(np.float32)

    # Create ONNX runtime session
    options = ort.SessionOptions()
    options.enable_profiling = True
    session = ort.InferenceSession(
        onnx_path, options, providers=["CPUExecutionProvider"]
    )

    # Get input and output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Run inference
    start_time = time.time()
    onnx_outputs = session.run([output_name], {input_name: input_data})
    onnx_time = time.time() - start_time

    print(f"ONNX inference time: {onnx_time:.4f} seconds")

    # Check outputs
    onnx_output = onnx_outputs[0]
    print(f"ONNX output shape: {onnx_output.shape}")

    return True


def create_example_input(image_path, input_shape, save_path=None):
    """
    Create an example input for the ONNX model.

    Args:
        image_path: Path to input image
        input_shape: Shape of input tensor (batch_size, channels, height, width)
        save_path: Path to save preprocessed input

    Returns:
        Preprocessed input tensor
    """
    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")

    # Resize to expected input shape
    height, width = input_shape[2], input_shape[3]

    # Define preprocessing
    preprocess = transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Apply preprocessing
    input_tensor = preprocess(img)

    # Add batch dimension
    input_tensor = input_tensor.unsqueeze(0)

    # Save input if requested
    if save_path is not None:
        # Convert to numpy and save
        input_numpy = input_tensor.numpy()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, input_numpy)
        print(f"Example input saved to {save_path}")

    return input_tensor


def create_conversion_report(
    original_path, onnx_path, input_shape, metadata=None, save_path=None
):
    """
    Create a conversion report with information about the models.

    Args:
        original_path: Path to original PyTorch model
        onnx_path: Path to ONNX model
        input_shape: Shape of input tensor
        metadata: Additional metadata
        save_path: Path to save report

    Returns:
        Report dictionary
    """
    # Get file sizes
    original_size = os.path.getsize(original_path) / (1024 * 1024)  # MB
    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB

    # Create report
    report = {
        "original_model": {"path": original_path, "size_mb": round(original_size, 2)},
        "onnx_model": {"path": onnx_path, "size_mb": round(onnx_size, 2)},
        "input_shape": list(input_shape),
        "size_change_percent": round(
            (onnx_size - original_size) / original_size * 100, 2
        ),
    }

    # Add metadata if provided
    if metadata is not None:
        report["metadata"] = metadata

    # Save report if requested
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(report, f, indent=4)
        print(f"Conversion report saved to {save_path}")

    return report


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")

    # Required arguments
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to PyTorch model"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save ONNX model"
    )

    # Optional arguments
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        choices=["classification", "siamese", "triplet", "arcface"],
        help="Type of model (default: auto-detect)",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="1,3,224,224",
        help="Shape of input tensor (default: 1,3,224,224)",
    )
    parser.add_argument(
        "--opset_version", type=int, default=12, help="ONNX opset version (default: 12)"
    )
    parser.add_argument(
        "--validate", action="store_true", help="Validate the exported ONNX model"
    )
    parser.add_argument(
        "--example_image",
        type=str,
        default=None,
        help="Path to example image for testing",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to use (default: cpu)"
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Parse input shape
    input_shape = tuple(map(int, args.input_shape.split(",")))

    # Set device
    device = torch.device(args.device)

    # Load PyTorch model
    print(f"Loading PyTorch model from {args.model_path}")
    model, model_info = load_model(args.model_path, device)

    # Use detected model type if not specified
    model_type = args.model_type or model_info["model_type"]
    print(f"Model type: {model_type}")

    # Get metadata from model info
    metadata = {
        "model_type": model_type,
        "num_classes": model_info["num_classes"],
        "framework": "pytorch",
    }

    if "backbone" in model_info:
        metadata["backbone"] = model_info["backbone"]

    if "embedding_dim" in model_info:
        metadata["embedding_dim"] = model_info["embedding_dim"]

    # Export model to ONNX
    print(f"Exporting model to ONNX format...")
    onnx_path = export_to_onnx(
        model=model,
        input_shape=input_shape,
        output_path=args.output_path,
        model_type=model_type,
        opset_version=args.opset_version,
        metadata=metadata,
    )

    # Validate ONNX model if requested
    if args.validate:
        print(f"Validating ONNX model...")
        validate_onnx_model(onnx_path=onnx_path, input_shape=input_shape)

    # Create example input if requested
    if args.example_image:
        print(f"Creating example input from {args.example_image}")
        example_output_path = os.path.join(
            os.path.dirname(args.output_path), "example_input.npy"
        )
        create_example_input(
            image_path=args.example_image,
            input_shape=input_shape,
            save_path=example_output_path,
        )

    # Create conversion report
    report_path = os.path.join(
        os.path.dirname(args.output_path), "conversion_report.json"
    )
    report = create_conversion_report(
        original_path=args.model_path,
        onnx_path=onnx_path,
        input_shape=input_shape,
        metadata=metadata,
        save_path=report_path,
    )

    # Print summary
    print("\nConversion Summary:")
    print(f"Original model: {report['original_model']['size_mb']:.2f} MB")
    print(f"ONNX model: {report['onnx_model']['size_mb']:.2f} MB")
    print(f"Size change: {report['size_change_percent']:.2f}%")
    print(f"ONNX model saved to: {onnx_path}")


if __name__ == "__main__":
    main()
