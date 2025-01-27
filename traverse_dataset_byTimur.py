import PIL.Image
import torch
import datasets
import numpy as np
import PIL

from typing import Callable, Any
from pathlib import Path
# from nip import nip



_IMAGE_KEY = "image"
_MASK_KEY = "mask"
_LABELS_KEY = "labels"
_PIXELS_KEY = "pixel_values"


class TransformsWrapper:

    def __init__(self, 
                 processor: Callable,       # handles final processing
                 transforms: Callable | None) -> None:      # performs augmentations on the raw images
        self._transforms = transforms
        self._processor = processor

    def __call__(self, 
                 input_data: dict[str, torch.Tensor | np.ndarray]) -> dict[str, torch.Tensor | np.ndarray]:
        pixel_values, labels, images, masks = [], [], [], []
        output_data = {}

        for pixel_value, label in zip(input_data[_IMAGE_KEY], input_data[_MASK_KEY]):
            pixel_value = np.array(pixel_value)
            label = np.array(label)
            if self._transforms is not None:
                transformed = self._transforms(image=pixel_value, mask=label)
                pixel_values.append(transformed[_IMAGE_KEY])
                labels.append(transformed[_MASK_KEY])
            else:
                pixel_values.append(pixel_value)
                labels.append(label)
            images.append(PIL.Image.fromarray(pixel_value))
            masks.append(PIL.Image.fromarray(label))

        output_data = {}
        output_data[_MASK_KEY] = masks
        output_data[_IMAGE_KEY] = images

        if self._processor is not None:
            processed_data = self._processor(pixel_values, labels)
            output_data[_PIXELS_KEY] = processed_data[_PIXELS_KEY]
            output_data[_LABELS_KEY] = processed_data[_LABELS_KEY]
        else:
            output_data[_PIXELS_KEY] = pixel_values
            output_data[_LABELS_KEY] = labels

        return output_data


# @nip
class TraverseDatasetFactory:

    def __init__(self,
                 processor: Callable | None = None,
                 augmentations: Callable | None = None,
                 val_fraction: float = 0.2) -> None:
        self._processor = processor
        self._augmentations = augmentations
        self._val_fraction = val_fraction

    def __call__(self, root_path: str | Path, split_name: str) -> datasets.Dataset:
        root_path = Path(root_path)
        image_paths = [str(e) for e in sorted((root_path / "frames").glob("*.jpg"))]
        label_paths = [str(e) for e in sorted((root_path / "masks_area").glob("*.png"))]

        split_idx = int(round(len(image_paths) * (1. - self._val_fraction)))
        if split_name == "train":
            image_paths = image_paths[:split_idx]
            label_paths = label_paths[:split_idx] 
        elif split_name == "val":
            image_paths = image_paths[split_idx:]
            label_paths = label_paths[split_idx:]

        dataset = datasets.Dataset.from_dict({
            "image": image_paths,
            "mask": label_paths,
        })
        dataset = dataset.cast_column("image", datasets.Image())
        dataset = dataset.cast_column("mask", datasets.Image())

        processor = self._processor
        if split_name == "train":
            augmentations = self._augmentations
        else:
            augmentations = None
        dataset.set_transform(TransformsWrapper(processor, augmentations))

        return dataset
    
print("Module loaded!")
print(f"Module contents: {dir()}")