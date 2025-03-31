# 🐢 Turtle Tinder: Sea Turtle Re-identification API

This project provides a FastAPI-based web API for identifying sea turtles from images using deep learning models.

## 🚀 Quick Start with Development Container

The easiest way to get started is using the included development container. This requires:

1. [Docker](https://www.docker.com/products/docker-desktop/)
2. [Visual Studio Code](https://code.visualstudio.com/)
3. [Remote Development Extension Pack](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack)

### Steps to Start the Development Container

1. Clone this repository
2. Open the project folder in VS Code
3. When prompted, click "Reopen in Container", or press F1 and select "Remote-Containers: Open Folder in Container"
4. Wait for the container to build (this might take a few minutes the first time)
5. You're ready to go! The development environment is fully set up

## 🖥️ Running the API

From within the development container, navigate to the API directory and run:

```bash
cd week4/deployment/api
python app.py --host 0.0.0.0 --port 8000
```

The API will be available at http://localhost:8000

## 📚 API Endpoints

- `GET /`: Root endpoint
- `GET /health`: Health check endpoint
- `GET /info`: Get model information
- `POST /identify`: Identify a turtle from an uploaded image
- `POST /load_model`: Load a new model

## 📋 Dependencies

All dependencies are managed through the requirements.txt file and are automatically installed in the development container. The main dependencies include:

- FastAPI
- Uvicorn
- PyTorch
- ONNX Runtime
- Pillow
- NumPy

## 🧠 Models

The API supports both PyTorch and ONNX models for sea turtle identification.

## 🤖 GPU Support

The development container is configured with CUDA support for GPU acceleration. If you have an NVIDIA GPU, it will be automatically used for faster inference.

## 🔧 Customizing the Environment

If you need to add additional Python packages:

1. Add them to the `requirements.txt` file
2. Rebuild the container (F1 -> "Remote-Containers: Rebuild Container")

## 📘 4-Week Sea Turtle Re-Identification Curriculum

This repository is part of a comprehensive 4-week curriculum on building a sea turtle re-identification system. The curriculum guides you through creating a deep learning system that can recognize individual sea turtles from images.

### [Week 1: Data Preparation and Baseline CNN Models](week1/README.md)

- Understanding the problem domain and data handling
- Setting up a structured dataset of sea turtle images
- Implementing preprocessing and augmentation techniques
- Building a baseline CNN classifier
- Evaluating initial model performance

### [Week 2: Transfer Learning and Fine-Tuning Advanced CNNs](week2/README.md)

- Leveraging pre-trained models (ResNet, EfficientNet)
- Understanding fine-tuning strategies and best practices
- Implementing advanced training techniques
- Comparing model architectures
- Significantly improving identification accuracy

### [Week 3: Metric Learning and Face-Recognition-Style Re-Identification](week3/README.md)

- Moving from classification to similarity learning
- Implementing Siamese networks and triplet loss
- Exploring specialized losses like ArcFace
- Learning embedding spaces for turtle comparison
- Evaluating re-identification performance

### [Week 4: Open-Set Recognition and Deployment](week4/README.md)

- Implementing open-set recognition for new turtles
- Building a system to track and update individual identities
- Evaluating with appropriate metrics (Rank-1, mAP)
- Deploying the model as a web API (this repository!)
- Planning for long-term system maintenance

The FastAPI application in this repository represents the culmination of the Week 4 deployment module, providing a practical implementation of the sea turtle re-identification system.

Happy turtle identifying! 🐢✨
