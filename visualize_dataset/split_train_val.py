#!/usr/bin/env python3
"""
Script to randomly move 10% of data from train/ to val/ folder.
"""

import os
import shutil
import random
from pathlib import Path


def split_train_val(train_dir='train', val_dir='val', val_ratio=0.1, seed=42):
    """
    Randomly move a portion of subdirectories from train_dir to val_dir.

    Args:
        train_dir: Source directory containing subdirectories to split
        val_dir: Destination directory for validation data
        val_ratio: Ratio of data to move to validation (default: 0.1 for 10%)
        seed: Random seed for reproducibility
    """
    # Set random seed for reproducibility
    random.seed(seed)

    # Get all subdirectories in train folder
    train_path = Path(train_dir)
    if not train_path.exists():
        print(f"Error: {train_dir} directory does not exist!")
        return

    # Get all subdirectories (not files)
    subdirs = [d for d in train_path.iterdir() if d.is_dir()]

    if not subdirs:
        print(f"No subdirectories found in {train_dir}")
        return

    total_count = len(subdirs)
    val_count = max(1, int(total_count * val_ratio))  # At least 1 item

    print(f"Total subdirectories in {train_dir}: {total_count}")
    print(f"Moving {val_count} subdirectories ({val_ratio*100:.1f}%) to {val_dir}")

    # Randomly sample subdirectories for validation
    val_subdirs = random.sample(subdirs, val_count)

    # Create val directory if it doesn't exist
    val_path = Path(val_dir)
    val_path.mkdir(exist_ok=True)
    print(f"Created {val_dir} directory")

    # Move selected subdirectories to val
    print("\nMoving subdirectories:")
    for i, subdir in enumerate(val_subdirs, 1):
        dest = val_path / subdir.name
        print(f"  [{i}/{val_count}] {subdir.name}")
        shutil.move(str(subdir), str(dest))

    print(f"\nDone! Moved {val_count} subdirectories from {train_dir} to {val_dir}")
    print(f"Remaining in {train_dir}: {total_count - val_count}")


if __name__ == "__main__":
    split_train_val()
