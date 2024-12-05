import os
import hashlib
from pathlib import Path
from collections import defaultdict

def get_image_hash(image_path):
    """Calculate hash for image files to identify duplicates."""
    with open(image_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def find_and_remove_duplicates(directory):
    """Find and remove duplicate images in the specified directory."""
    hash_dict = defaultdict(list)  

    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpeg', '.jpg', '.jfif')):  
                file_path = Path(root) / file
                file_hash = get_image_hash(file_path)
                hash_dict[file_hash].append(file_path)

    for file_hash, file_paths in hash_dict.items():
        if len(file_paths) > 1:
            print(f"found {len(file_paths)} duplicates for hash {file_hash}. Removing all but one.")
            for file_path in file_paths[1:]:  
                print(f"removing duplicate: {file_path}")
                file_path.unlink()  

    print("deduplication complete.")

if __name__ == "__main__":
    data_dir = Path('/workspace/cs229-proj/data')  
    find_and_remove_duplicates(data_dir)
