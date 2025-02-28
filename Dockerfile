FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Add metadata
LABEL maintainer="Fatykhoph Denis"
LABEL description="Development environment with Python and ML libraries for Segmentation"

ARG UID
ARG GID

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.10

# Create user and install dependencies in one layer
RUN addgroup --gid ${GID} --system oversir \
    && adduser --uid ${UID} --system \
               --ingroup oversir \
               --home /home/oversir \
               --shell /bin/bash oversir \
    && chown -R oversir:oversir /home/oversir \
    && apt-get update \
    && apt-get install -y \
        python${PYTHON_VERSION} \
        python3-dev \
        python3-pip \
        build-essential libc6-dev \
    && rm -rf /var/lib/apt/lists/* \
    && usermod -aG sudo oversir \
    && echo 'oversir ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Set working directory
WORKDIR /app

# Install Python packages
RUN pip3 install --no-cache-dir \
    torch==2.0.1 \
    torchvision==0.15.2 \
    numpy==1.24.3 \
    opencv-python-headless==4.8.0.74 \
    matplotlib==3.7.1 \
    segment-anything==1.0 \
    albumentations==1.3.1 \
    pillow==9.5.0 \
    scipy==1.10.1 \
    scikit-learn==1.2.2 \
    pandas==2.0.2 \
    tqdm==4.65.0

RUN pip3 install --no-cache-dir \
    torch==2.6.0 \
    Pillow==11.1.0 \
    transformers==4.48.3 \
    torchvision==0.21.0 \
    bitsandbytes==0.45.2 \
    einops==0.8.1 \
    # xformers==0.029 \
    accelerate==1.3.0

USER oversir
WORKDIR /home/oversir

# Set default command
CMD ["/bin/bash"] 