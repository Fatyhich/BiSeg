# Segmentation Models Collection

A repository containing various image segmentation models and tools for computer vision tasks.

## Repository Structure

```
segmentation/
├── LeLaN/                  # LeLaN segmentation model implementation
│   ├── CogVLM2/            # CogVLM2 model for image captioning
│   └── SAM/                # SAM integration for LeLaN
├── dataset_viz.ipynb       # Dataset visualization notebook
```

## Features

- Integration with Segment Anything Model (SAM)
- Dataset visualization and processing tools
- Jupyter notebooks for experimentation
- CogVLM2 integration for image captioning

## Getting Started

1. Clone the repository
2. Before running the docker compose, you need to check your GID/UID and set it in the Dockerfile
3. Using command: 
```bash
docker compose -f 'compose.rucula.yaml' up -d --build develop
```
or for the tomate environment:
```bash
docker compose -f 'compose.tomate.yaml' up -d --build develop
```
you can run the docker container.

## Models

- **SAM**: Segment Anything Model integration
- **CogVLM2**: Vision-language model for image captioning

## Tools

- Dataset visualization notebook for data exploration
- Dataset traversal utility for processing large datasets
- SAM processor for automatic image segmentation and cropping

## Contact

For any questions or support, please contact [Denis Fatykhoph](mailto:D.Fatykhoph@skoltech.ru).