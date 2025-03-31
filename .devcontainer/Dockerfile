FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# Install essential packages and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install development tools
RUN pip install --no-cache-dir \
    black \
    pylint \
    isort \
    jupyter \
    ipykernel

# Install required libraries for turtles
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV PYTHONPATH=/workspace:$PYTHONPATH
ENV PYTHONUNBUFFERED=1

# Create a non-root user to use if preferred
ARG USERNAME=student
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && mkdir -p /home/$USERNAME/.vscode-server /home/$USERNAME/.vscode-server-insiders \
    && chown -R $USERNAME:$USERNAME /home/$USERNAME \
    && chown -R $USERNAME:$USERNAME /workspace

# Install ONNX Runtime with GPU support
RUN pip install --no-cache-dir onnxruntime-gpu

# Set up Jupyter notebook configuration
RUN mkdir -p /home/$USERNAME/.jupyter \
    && echo "c.NotebookApp.token = ''" > /home/$USERNAME/.jupyter/jupyter_notebook_config.py \
    && echo "c.NotebookApp.password = ''" >> /home/$USERNAME/.jupyter/jupyter_notebook_config.py \
    && echo "c.NotebookApp.ip = '0.0.0.0'" >> /home/$USERNAME/.jupyter/jupyter_notebook_config.py \
    && chown -R $USERNAME:$USERNAME /home/$USERNAME/.jupyter

# Default command
CMD ["bash"] 