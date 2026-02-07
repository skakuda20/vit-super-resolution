"""
Module: evaluation_metrics.py
This module contains functions to calculate various evaluation metrics for image super-resolution tasks.
"""

import torch
import lpips
import numpy as np


def calculate_lpips(image1, image2, model="alex"):
    """
    Calculate the LPIPS (Learned Perceptual Image Patch Similarity) score between two images.

    Args:
        image1 (torch.Tensor): The first image tensor.
        image2 (torch.Tensor): The second image tensor.
        model (str): The LPIPS model to use. Default is 'alex'.

    Returns:
        float: The LPIPS score between the two images.
    """
    # Ensure the images are in the correct format
    if not isinstance(image1, torch.Tensor) or not isinstance(image2, torch.Tensor):
        raise ValueError("Both images must be PyTorch tensors.")

    # Load the LPIPS model
    lpips_model = lpips.LPIPS(net=model)

    # Calculate the LPIPS score
    score = lpips_model.forward(image1, image2)

    return score.item()
