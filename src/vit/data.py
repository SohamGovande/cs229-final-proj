import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

from .utils import calculate_class_weights_from_directory


def load_image_data(data_dir:Path, image_size, batch_size=1, num_workers=2, transform=None, use_sampler=False, shuffle=False):
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    # Load data directly from the directory
    dataset = ImageFolder(root=data_dir, transform=transform)


    if not use_sampler:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    _, class_weights, _ = calculate_class_weights_from_directory(data_dir)
    sampler = WeightedRandomSampler(class_weights, len(class_weights), replacement=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
    )
         
