# Segmentation Models Collection

A repository containing various image segmentation models and tools for computer vision tasks.

## Repository Structure

```
segmentation/
├── LeLaN/                  # LeLaN segmentation model implementation
├── VLTSeg/                 # VLTSeg model implementation
├── PlainSeg/              # PlainSeg model implementation
├── mmsegmentation/        # MMSegmentation framework
├── segmentation_models.pytorch/  # PyTorch-based segmentation models
├── sam/                   # Segment Anything Model (SAM) related files
├── segment-anything/      # Original SAM implementation
├── datset_viz.ipynb      # Dataset visualization notebook
└── traverse_dataset_byTimur.py  # Dataset traversal utility
```

## Features

- Multiple state-of-the-art segmentation model implementations
- Integration with Segment Anything Model (SAM)
- Dataset visualization and processing tools
- Jupyter notebooks for experimentation

## Getting Started

1. Clone the repository
2. Before running the docker compose, you need to check your GID/UID and set it in the Dockerfile
3. Using command: 
```bash
docker compose -f 'compose.rucula.yaml' up -d --build develop
```
you can run the docker container.

## Models

- **LeLaN**: Language-enhanced Label-free Segmentation
- **VLTSeg**: Vision-Language Transformer Segmentation
- **PlainSeg**: Lightweight segmentation model
- **SAM**: Segment Anything Model integration

## Tools

- Dataset visualization notebook for data exploration
- Dataset traversal utility for processing large datasets

## Contact

For any questions or support, please contact [Denis Fatykhoph](mailto:D.Fatykhoph@skoltech.ru).
