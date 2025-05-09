"""
Utility functions for file path resolution and logging.
This module provides functions to resolve file paths in a UNIX-style format and to log messages to a text file.
"""

import os
import yaml
from pathlib import Path


import os
import yaml
from pathlib import Path

def load_config(config_path="params.yaml"):
    """Load configuration from a YAML file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def resolve_path(path):
    """
    Normalizes the file path for both UNIX and Windows systems.

    Args:
        path: Path object to normalize.

    Returns:
        Normalized Path object.
    """
    # Use Path from pathlib to handle OS-specific path separators
    normalized_path = Path(path).resolve()
    return normalized_path

def get_project_root_path():
    """
    Returns the root project path as a Path object.

    Returns:
        Path: Normalized Path object pointing to the project root.
    """
    # Get the directory of the current script (__file__) and go one level up
    root_dir_path = Path(__file__).resolve().parent.parent
    return root_dir_path



def log_message(log_file, message):
    """
    Saves out message to txt log file.

    Args:
        log_file (str or Path):  Path to CSV file.
        message (str): Message to save to txt log file.
    """
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")
