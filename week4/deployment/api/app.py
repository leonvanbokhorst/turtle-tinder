#!/usr/bin/env python
"""
FastAPI application for sea turtle re-identification.

This script creates a web API for identifying sea turtles from images.
"""

import os
import sys
import json
import time
import argparse
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io
import torch
import torchvision.transforms as transforms

# Try to import onnxruntime
try:
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: onnxruntime not found. Using PyTorch for inference.")

# Add parent directories to path for importing utils
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from utils.model_loader import load_model


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("turtle_api")


class IdentificationRequest(BaseModel):
    """Request model for identification."""

    top_k: int = 3
    threshold: float = 0.0


class IdentificationResult(BaseModel):
    """Result model for identification."""

    class_id: int
    class_name: str
    confidence: float


class IdentificationResponse(BaseModel):
    """Response model for identification."""

    results: List[IdentificationResult]
    processing_time: float


class TurtleIdentifier:
    """Class for sea turtle identification."""

    def __init__(
        self,
        model_path: str,
        class_names_path: Optional[str] = None,
        device: str = "cpu",
        input_size: tuple = (224, 224),
        use_onnx: bool = None,
    ):
        """
        Initialize the identifier.

        Args:
            model_path: Path to the model
            class_names_path: Path to class names JSON file
            device: Device to use for inference
            input_size: Input image size
            use_onnx: Whether to use ONNX runtime (default: auto-detect)
        """
        self.model_path = model_path
        self.device = device
        self.input_size = input_size

        # Auto-detect if we should use ONNX
        if use_onnx is None:
            use_onnx = model_path.endswith(".onnx") and ONNX_AVAILABLE

        self.use_onnx = use_onnx

        # Load model
        if self.use_onnx:
            self._load_onnx_model()
        else:
            self._load_pytorch_model()

        # Set up preprocessing
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Load class names
        self._load_class_names(class_names_path)

        logger.info(f"Initialized TurtleIdentifier with model: {model_path}")
        logger.info(f"Found {len(self.class_names)} classes")
        logger.info(
            f"Using {'ONNX runtime' if self.use_onnx else 'PyTorch'} for inference"
        )

    def _load_pytorch_model(self):
        """Load PyTorch model."""
        device = torch.device(self.device)
        self.model, self.model_info = load_model(self.model_path, device)
        self.model.eval()

        # Get model type
        self.model_type = self.model_info.get("model_type", "classification")

        # Check if it's a metric learning model
        self.is_metric_learning = self.model_type in ["siamese", "triplet", "arcface"]

        # If it's a metric learning model, we need to load a classifier or do similarity search
        if self.is_metric_learning and not hasattr(self.model, "classify"):
            logger.warning(
                "Metric learning model without a classify method. Will use similarity search."
            )
            self.use_similarity_search = True

            # Try to load gallery embeddings and labels
            gallery_path = os.path.join(
                os.path.dirname(self.model_path), "gallery_embeddings.pt"
            )
            if os.path.exists(gallery_path):
                logger.info(f"Loading gallery embeddings from {gallery_path}")
                gallery_data = torch.load(gallery_path, map_location=device)
                self.gallery_embeddings = gallery_data["embeddings"]
                self.gallery_labels = gallery_data["labels"]
            else:
                logger.warning(
                    f"Gallery embeddings not found at {gallery_path}. Similarity search may not work."
                )
                self.gallery_embeddings = None
                self.gallery_labels = None
        else:
            self.use_similarity_search = False

    def _load_onnx_model(self):
        """Load ONNX model."""
        if not ONNX_AVAILABLE:
            raise ImportError(
                "onnxruntime not available. Install with pip install onnxruntime"
            )

        # Set up ONNX runtime session
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device.lower() == "cuda"
            and "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )

        self.session = ort.InferenceSession(self.model_path, providers=providers)

        # Get input and output details
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Try to load metadata
        try:
            import onnx

            onnx_model = onnx.load(self.model_path)
            metadata = {prop.key: prop.value for prop in onnx_model.metadata_props}

            self.model_type = metadata.get("model_type", "classification")
            self.is_metric_learning = self.model_type in [
                "siamese",
                "triplet",
                "arcface",
            ]

            # If it's a metric learning model, we need to load gallery embeddings and labels
            if self.is_metric_learning:
                logger.info(f"Loaded metric learning model of type {self.model_type}")
                self.use_similarity_search = True

                # Try to load gallery embeddings and labels
                gallery_path = os.path.join(
                    os.path.dirname(self.model_path), "gallery_embeddings.npz"
                )
                if os.path.exists(gallery_path):
                    logger.info(f"Loading gallery embeddings from {gallery_path}")
                    gallery_data = np.load(gallery_path)
                    self.gallery_embeddings = gallery_data["embeddings"]
                    self.gallery_labels = gallery_data["labels"]
                else:
                    logger.warning(
                        f"Gallery embeddings not found at {gallery_path}. Similarity search may not work."
                    )
                    self.gallery_embeddings = None
                    self.gallery_labels = None
            else:
                self.use_similarity_search = False

        except ImportError:
            logger.warning("onnx module not found. Could not load model metadata.")
            self.model_type = "classification"
            self.is_metric_learning = False
            self.use_similarity_search = False

    def _load_class_names(self, class_names_path: Optional[str] = None):
        """
        Load class names.

        Args:
            class_names_path: Path to class names JSON file
        """
        if class_names_path and os.path.exists(class_names_path):
            # Load from specified file
            with open(class_names_path, "r") as f:
                self.class_names = json.load(f)

            # Handle different formats
            if isinstance(self.class_names, dict):
                # Convert string keys to int if needed
                if all(k.isdigit() for k in self.class_names.keys()):
                    self.class_names = {int(k): v for k, v in self.class_names.items()}

                # Create list from dict
                max_idx = max(self.class_names.keys())
                class_list = ["Unknown"] * (max_idx + 1)
                for idx, name in self.class_names.items():
                    class_list[idx] = name

                self.class_names = class_list
        else:
            # Try to find class names in model directory
            model_dir = os.path.dirname(self.model_path)
            potential_paths = [
                os.path.join(model_dir, "class_names.json"),
                os.path.join(model_dir, "classes.json"),
                os.path.join(model_dir, "labels.json"),
            ]

            for path in potential_paths:
                if os.path.exists(path):
                    with open(path, "r") as f:
                        self.class_names = json.load(f)

                    # Handle different formats
                    if isinstance(self.class_names, dict):
                        # Convert string keys to int if needed
                        if all(k.isdigit() for k in self.class_names.keys()):
                            self.class_names = {
                                int(k): v for k, v in self.class_names.items()
                            }

                        # Create list from dict
                        max_idx = max(self.class_names.keys())
                        class_list = ["Unknown"] * (max_idx + 1)
                        for idx, name in self.class_names.items():
                            class_list[idx] = name

                        self.class_names = class_list

                    break
            else:
                # If no class names file found, create generic names
                if self.use_onnx:
                    # Try to get from ONNX metadata
                    try:
                        import onnx

                        onnx_model = onnx.load(self.model_path)
                        metadata = {
                            prop.key: prop.value for prop in onnx_model.metadata_props
                        }
                        num_classes = int(metadata.get("num_classes", 100))
                    except (ImportError, KeyError, ValueError):
                        num_classes = 100
                else:
                    # Get from PyTorch model
                    num_classes = self.model_info.get("num_classes", 100)

                self.class_names = [f"Class {i}" for i in range(num_classes)]

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess an image for inference.

        Args:
            image: PIL Image

        Returns:
            Preprocessed tensor
        """
        # Apply preprocessing
        tensor = self.preprocess(image)

        # Add batch dimension
        tensor = tensor.unsqueeze(0)

        return tensor

    def inference_pytorch(
        self, image_tensor: torch.Tensor, top_k: int = 3, threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Perform inference using PyTorch.

        Args:
            image_tensor: Input tensor
            top_k: Number of top predictions to return
            threshold: Confidence threshold

        Returns:
            List of predictions
        """
        # Move tensor to device
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            if self.use_similarity_search:
                # For metric learning models
                if hasattr(self.model, "get_embedding"):
                    embedding = self.model.get_embedding(image_tensor)
                else:
                    # Try to extract embedding
                    if self.model_type == "arcface":
                        _, _, embedding = self.model(image_tensor)
                    else:
                        # Last resort - use output as embedding
                        embedding = self.model(image_tensor)

                # Perform similarity search
                if self.gallery_embeddings is not None:
                    # Calculate similarity scores
                    embedding = embedding.cpu().numpy()

                    # Normalize embeddings
                    embedding = embedding / np.linalg.norm(
                        embedding, axis=1, keepdims=True
                    )
                    gallery_embeddings = self.gallery_embeddings / np.linalg.norm(
                        self.gallery_embeddings, axis=1, keepdims=True
                    )

                    # Calculate cosine similarity
                    similarity = np.dot(embedding, gallery_embeddings.T)[0]

                    # Get top-k indices
                    top_indices = np.argsort(similarity)[::-1][:top_k]

                    # Get labels and scores
                    results = []
                    for i in top_indices:
                        if similarity[i] >= threshold:
                            class_id = int(self.gallery_labels[i])
                            results.append(
                                {
                                    "class_id": class_id,
                                    "class_name": (
                                        self.class_names[class_id]
                                        if class_id < len(self.class_names)
                                        else f"Class {class_id}"
                                    ),
                                    "confidence": float(similarity[i]),
                                }
                            )

                    return results
                else:
                    logger.warning(
                        "Gallery embeddings not available for similarity search"
                    )
                    return []
            else:
                # For classification models
                logits = self.model(image_tensor)
                probabilities = torch.softmax(logits, dim=1)[0]

                # Get top-k indices and scores
                if top_k > 0:
                    topk_values, topk_indices = torch.topk(
                        probabilities, min(top_k, len(probabilities))
                    )

                    # Convert to numpy for easier handling
                    indices = topk_indices.cpu().numpy()
                    scores = topk_values.cpu().numpy()
                else:
                    # Get all predictions above threshold
                    mask = probabilities >= threshold
                    indices = torch.nonzero(mask).squeeze().cpu().numpy()
                    if indices.size == 0:
                        indices = probabilities.argmax().unsqueeze(0).cpu().numpy()
                    scores = probabilities[mask].cpu().numpy()

                # Create results
                results = []
                for i, class_id in enumerate(indices):
                    if i < len(scores) and scores[i] >= threshold:
                        results.append(
                            {
                                "class_id": int(class_id),
                                "class_name": (
                                    self.class_names[class_id]
                                    if class_id < len(self.class_names)
                                    else f"Class {class_id}"
                                ),
                                "confidence": float(scores[i]),
                            }
                        )

                return results

    def inference_onnx(
        self, image_tensor: torch.Tensor, top_k: int = 3, threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Perform inference using ONNX runtime.

        Args:
            image_tensor: Input tensor
            top_k: Number of top predictions to return
            threshold: Confidence threshold

        Returns:
            List of predictions
        """
        # Convert PyTorch tensor to numpy array
        input_array = image_tensor.numpy()

        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: input_array})
        output = outputs[0]

        if self.use_similarity_search:
            # For metric learning models
            embedding = output

            # Perform similarity search
            if self.gallery_embeddings is not None:
                # Normalize embeddings
                embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
                gallery_embeddings = self.gallery_embeddings / np.linalg.norm(
                    self.gallery_embeddings, axis=1, keepdims=True
                )

                # Calculate cosine similarity
                similarity = np.dot(embedding, gallery_embeddings.T)[0]

                # Get top-k indices
                top_indices = np.argsort(similarity)[::-1][:top_k]

                # Get labels and scores
                results = []
                for i in top_indices:
                    if similarity[i] >= threshold:
                        class_id = int(self.gallery_labels[i])
                        results.append(
                            {
                                "class_id": class_id,
                                "class_name": (
                                    self.class_names[class_id]
                                    if class_id < len(self.class_names)
                                    else f"Class {class_id}"
                                ),
                                "confidence": float(similarity[i]),
                            }
                        )

                return results
            else:
                logger.warning("Gallery embeddings not available for similarity search")
                return []
        else:
            # For classification models
            probabilities = output[0]

            # Apply softmax if needed
            if probabilities.max() > 1.0 or probabilities.min() < 0.0:
                probabilities = np.exp(probabilities - np.max(probabilities))
                probabilities = probabilities / np.sum(probabilities)

            # Get top-k indices and scores
            if top_k > 0:
                top_indices = np.argsort(probabilities)[::-1][:top_k]
                scores = probabilities[top_indices]
            else:
                # Get all predictions above threshold
                mask = probabilities >= threshold
                top_indices = np.nonzero(mask)[0]
                if top_indices.size == 0:
                    top_indices = np.array([np.argmax(probabilities)])
                scores = probabilities[top_indices]

            # Create results
            results = []
            for i, class_id in enumerate(top_indices):
                if i < len(scores) and scores[i] >= threshold:
                    results.append(
                        {
                            "class_id": int(class_id),
                            "class_name": (
                                self.class_names[class_id]
                                if class_id < len(self.class_names)
                                else f"Class {class_id}"
                            ),
                            "confidence": float(scores[i]),
                        }
                    )

            return results

    def identify(
        self, image: Image.Image, top_k: int = 3, threshold: float = 0.0
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Identify a sea turtle in an image.

        Args:
            image: PIL Image
            top_k: Number of top predictions to return
            threshold: Confidence threshold

        Returns:
            Tuple of (results, processing_time)
        """
        # Record start time
        start_time = time.time()

        # Preprocess image
        image_tensor = self.preprocess_image(image)

        # Perform inference
        if self.use_onnx:
            results = self.inference_onnx(image_tensor, top_k, threshold)
        else:
            results = self.inference_pytorch(image_tensor, top_k, threshold)

        # Calculate processing time
        processing_time = time.time() - start_time

        return results, processing_time


# Create FastAPI app
app = FastAPI(
    title="Sea Turtle Re-Identification API",
    description="API for identifying sea turtles from images",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
identifier = None


@app.on_event("startup")
async def startup_event():
    """Initialize the app on startup."""
    global identifier

    # Get model path from environment variable or command line
    model_path = os.environ.get("MODEL_PATH", None)
    class_names_path = os.environ.get("CLASS_NAMES_PATH", None)
    device = os.environ.get("DEVICE", "cpu")

    if model_path is None:
        logger.warning("MODEL_PATH not set. API will not work until a model is loaded.")
        return

    # Initialize identifier
    try:
        identifier = TurtleIdentifier(
            model_path=model_path, class_names_path=class_names_path, device=device
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        identifier = None


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Sea Turtle Re-Identification API"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": identifier is not None}


@app.get("/info")
async def info():
    """Get information about the model."""
    if identifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    info = {
        "model_path": identifier.model_path,
        "model_type": getattr(identifier, "model_type", "unknown"),
        "num_classes": len(identifier.class_names),
        "class_names": identifier.class_names,
        "use_onnx": identifier.use_onnx,
        "input_size": identifier.input_size,
    }

    return info


@app.post("/identify")
async def identify(
    file: UploadFile = File(...),
    top_k: int = Query(3, ge=0, description="Number of top predictions to return"),
    threshold: float = Query(0.0, ge=0.0, le=1.0, description="Confidence threshold"),
):
    """
    Identify a sea turtle from an image.

    Args:
        file: Image file
        top_k: Number of top predictions to return
        threshold: Confidence threshold

    Returns:
        Identification results
    """
    if identifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Read and validate image
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        image = image.convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

    # Perform identification
    try:
        results, processing_time = identifier.identify(image, top_k, threshold)
    except Exception as e:
        logger.error(f"Error during identification: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error during identification: {str(e)}"
        )

    # Return results
    return {"results": results, "processing_time": processing_time}


@app.post("/load_model")
async def load_model(
    model_path: str = Form(...),
    class_names_path: Optional[str] = Form(None),
    device: str = Form("cpu"),
):
    """
    Load a new model.

    Args:
        model_path: Path to the model
        class_names_path: Path to class names JSON file
        device: Device to use for inference

    Returns:
        Success message
    """
    global identifier

    # Validate model path
    if not os.path.exists(model_path):
        raise HTTPException(status_code=400, detail=f"Model not found: {model_path}")

    # Load model
    try:
        identifier = TurtleIdentifier(
            model_path=model_path, class_names_path=class_names_path, device=device
        )
        logger.info(f"Model loaded from {model_path}")
        return {"message": "Model loaded successfully"}
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Sea Turtle Re-Identification API")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model"
    )
    parser.add_argument(
        "--class_names_path",
        type=str,
        default=None,
        help="Path to class names JSON file",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to listen on")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to use for inference"
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Set environment variables for FastAPI app
    os.environ["MODEL_PATH"] = args.model_path

    if args.class_names_path:
        os.environ["CLASS_NAMES_PATH"] = args.class_names_path

    os.environ["DEVICE"] = args.device

    # Start uvicorn server
    uvicorn.run("app:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
