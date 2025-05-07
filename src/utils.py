"""
Utility functions for file path resolution and logging.
This module provides functions to resolve file paths in a UNIX-style format and to log messages to a text file.
"""

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
    Normalizes UNIX-style file path.

    Args:
        path: Path object to normalize.

    Returns:
        Normalized Path object.
    """
    parts = path.split("/")
    resolved = []

    for part in parts:
        if part == "..":
            if resolved:
                resolved.pop()
        elif part and part != ".":
            resolved.append(part)

    return "/" + "/".join(resolved)


def get_project_root_path():
    """
    Returns the root project path as a Path object.

    Returns:
        Path: Normalized Path object pointing to the project root.
    """
    root_dir_path = os.path.join(__file__, "..")
    root_dir_path = resolve_path(root_dir_path)
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
