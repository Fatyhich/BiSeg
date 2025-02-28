import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import random
import fire
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def show_anns(anns: List[Dict[str, Any]]) -> None:
    """
    Display annotations/masks on the current matplotlib axis.
    
    Args:
        anns: List of annotations, where each annotation is a dict containing 
              'segmentation' and 'area' keys
    """
    if len(anns) == 0:
        return
        
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], 
                  sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def get_all_frames(root_dir: Union[str, Path]) -> List[Path]:
    """
    Get all JPG files from all 'frames' subdirectories.
    
    Args:
        root_dir: Root directory to search for frames
        
    Returns:
        List of paths to JPG files
        
    Raises:
        ValueError: If no JPG files are found
    """
    root_dir = Path(root_dir)
    all_frames = []
    
    for path in sorted(root_dir.rglob('frames/*.jpg')):
        if path.is_file():
            all_frames.append(path)
    
    if not all_frames:
        raise ValueError("No jpg files found in directory structure")
    
    return all_frames

def crop_by_mask(image: np.ndarray, bbox: tuple) -> np.ndarray:
    """
    Crop image using bounding box coordinates.
    
    Args:
        image: Input image as numpy array
        bbox: Tuple of (x, y, width, height)
        
    Returns:
        Cropped image as numpy array
    """
    x, y, w, h = bbox
    return image[y:y+h, x:x+w]

def process_image(image_path: Union[str, Path], 
                 output_dir: Union[str, Path],
                 masks: List[Dict[str, Any]]) -> None:
    """
    Process a single image and save its crops.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save crops
        masks: List of mask annotations
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    
    # Read and convert image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each mask
    for idx, mask in enumerate(masks):
        bbox = mask['bbox']
        cropped_img = crop_by_mask(image, bbox)
        
        crop_filename = f"{image_path.stem}_crop_{idx}.jpg"
        crop_path = output_dir / crop_filename
        
        cropped_img_bgr = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(crop_path), cropped_img_bgr)
    
    print(f"Saved {len(masks)} crops to directory {output_dir}")

class SAMProcessor:
    """Process images with SAM model and extract crops."""
    
    def process(self, 
                source_dir: str,
                target_dir: str,
                checkpoint_dir: str = "/mnt/datasets/checkpoints",
                num_samples: Optional[int] = None) -> None:
        """
        Process images from source directory and save crops to target directory.
        
        Args:
            source_dir: Source directory containing frames
            target_dir: Target directory for saving crops
            checkpoint_dir: Directory containing SAM checkpoint (default: "/mnt/datasets/checkpoints")
            num_samples: Number of random frames to process (default: all)
        """
        try:
            # Get all frames
            all_frames = get_all_frames(source_dir)
            
            # Sample frames if specified
            if num_samples is not None:
                frames_to_process = random.sample(
                    all_frames, 
                    min(num_samples, len(all_frames))
                )
            else:
                frames_to_process = all_frames
                
            # Process each frame
            for frame in frames_to_process:
                # Initialize SAM model
                sam_checkpoint = checkpoint_dir + "/sam_vit_h_4b8939.pth"
                model_type = "vit_h"
                device = "cuda"

                sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
                sam.to(device=device)

                # Generate masks
                mask_generator = SamAutomaticMaskGenerator(sam)
                image = cv2.imread(str(frame))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                masks = mask_generator.generate(image)
                
                process_image(frame, target_dir, masks)
                
        except ValueError as e:
            print(f"Error: {e}")
            return 1
        
        return 0

def main():
    fire.Fire(SAMProcessor)

if __name__ == "__main__":
    main() 