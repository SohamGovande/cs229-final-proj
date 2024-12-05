import os
import numpy as np
from PIL import Image

class FlipAndMirrorAugmentation(object):
    """Generate 8 augmented versions of each image using flips and mirrors."""
    def __init__(self):
        self.transformations = [
            lambda x: x,  # Original
            lambda x: x.transpose(Image.FLIP_LEFT_RIGHT),  # Horizontal flip
            lambda x: x.transpose(Image.FLIP_TOP_BOTTOM),  # Vertical flip
            lambda x: x.transpose(Image.ROTATE_90),  # 90 degrees rotation
            lambda x: x.transpose(Image.ROTATE_180),  # 180 degrees rotation
            lambda x: x.transpose(Image.ROTATE_270),  # 270 degrees rotation
            lambda x: x.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90),  # Horizontal flip + 90 degrees
            lambda x: x.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_270)  # Horizontal flip + 270 degrees
        ]
    
    def augment_and_save(self, img, output_dir, filename):
        # Create and save each transformation
        for i, transform in enumerate(self.transformations):
            transformed_img = transform(img)
            transformed_filename = f"{os.path.splitext(filename)[0]}_aug_{i}.jpg"
            output_path = os.path.join(output_dir, transformed_filename)
            transformed_img.save(output_path)
            print(f"Saved augmented image: {output_path}")

# Dictionary mapping original image directories to augmented image directories 
dirs = {
    'data/chest_xray/train-bv/BACTERIAL': 'data/chest_xray/train-bv-aug/BACTERIAL', 
    'data/chest_xray/train-bv/VIRAL': 'data/chest_xray/train-bv-aug/VIRAL',
    'data/chest_xray/test-bv/BACTERIAL': 'data/chest_xray/test-bv-aug/BACTERIAL',
    'data/chest_xray/test-bv/VIRAL': 'data/chest_xray/test-bv-aug/VIRAL'
}

for output_dir in dirs.values():
    os.makedirs(output_dir, exist_ok=True)

# Augment images in input directories and save them to output directorites 
augmenter = FlipAndMirrorAugmentation()
for input_dir, output_dir in dirs.items():
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path)
            augmenter.augment_and_save(img, output_dir, filename)
