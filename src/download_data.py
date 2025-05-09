"""
Module: download_data.py
This script is used to download the DIV2K dataset from Kaggle using the kagglehub library.
"""

import kagglehub

# Download latest version
path = kagglehub.dataset_download("soumikrakshit/div2k-high-resolution-images")

print("Path to dataset files:", path)
