FROM ubuntu:22.04

# Set noninteractive installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python packages
RUN pip3 install --no-cache-dir \
    torch==2.4.0 \
    torchvision==0.19.0 \
    numpy==2.1.0 \
    opencv-python-headless==4.10.0.84 \
    matplotlib==3.9.2 \
    segment-anything==1.0 \
    albumentations==1.4.24 \
    pillow==10.4.0 \
    scipy==1.14.1 \
    scikit-learn==1.6.0 \
    pandas==2.2.3 \
    tqdm==4.66.5

# Set Python path
ENV PYTHONPATH=/app

# Set default command
CMD ["bash"] 