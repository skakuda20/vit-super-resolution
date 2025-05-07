"""
DIV2K Dataset Loader
This module provides a PyTorch Dataset class for loading images from the DIV2K dataset.
It includes functionality to read high-resolution images from a specified directory and apply transformations.
"""

import os
from PIL import Image
from torch.utils.data import Dataset


class DIV2KDataset(Dataset):
    """A PyTorch Dataset class for loading images from the DIV2K dataset."""

    def __init__(self, hr_dir, transform=None):
        self.hr_dir = hr_dir
        self.image_files = sorted(
            [
                os.path.join(hr_dir, fname)
                for fname in os.listdir(hr_dir)
                if fname.endswith(".png") or fname.endswith(".jpg")
            ]
        )
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        hr_image = Image.open(self.image_files[idx]).convert("RGB")
        if self.transform:
            hr_image = self.transform(hr_image)
        return hr_image  # Return a single tensor, not a list
