# Use the official NVIDIA CUDA image as the base image
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Set the working directory inside the container
WORKDIR /app

# Copy the application code into the container
COPY . /app

# Install necessary dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    build-essential libc6-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --no-cache-dir \
                torch==2.6.0 \
                Pillow==11.1.0 \
                transformers==4.48.3 \
                torchvision==0.21.0 \
                bitsandbytes==0.45.2 \
                einops==0.8.1 \
                xformers==0.029 \
                accelerate==1.3.0