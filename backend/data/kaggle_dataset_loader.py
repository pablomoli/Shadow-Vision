#!/usr/bin/env python3
"""
Kaggle Dataset Loader for Gesture Recognition
Replaces Supabase integration with local Kaggle dataset loading
"""

import os
import sys
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

class KaggleDatasetLoader:
    """
    Flexible dataset loader for Kaggle gesture recognition datasets
    Supports various dataset formats and structures
    """

    def __init__(self, dataset_path: str = "data/kaggle_dataset"):
        self.dataset_path = Path(dataset_path)
        self.gesture_classes = {}
        self.dataset_info = {}

    def load_dataset_from_csv(self, csv_path: str, image_dir: str = None) -> bool:
        """
        Load dataset from CSV format (common Kaggle format)
        CSV should have columns: image_path, label
        """
        try:
            import pandas as pd

            csv_path = Path(csv_path)
            if not csv_path.exists():
                print(f"CSV file not found: {csv_path}")
                return False

            df = pd.read_csv(csv_path)
            print(f"Loaded CSV with {len(df)} entries")

            # Auto-detect column names
            possible_image_cols = ['image', 'image_path', 'filename', 'file']
            possible_label_cols = ['label', 'class', 'gesture', 'category']

            image_col = None
            label_col = None

            for col in df.columns:
                if col.lower() in possible_image_cols:
                    image_col = col
                if col.lower() in possible_label_cols:
                    label_col = col

            if not image_col or not label_col:
                print(f"Could not find image and label columns in CSV")
                print(f"Available columns: {list(df.columns)}")
                return False

            # Extract unique gestures and create mapping
            unique_labels = df[label_col].unique()
            self.gesture_classes = {label: idx for idx, label in enumerate(sorted(unique_labels))}

            print(f"Found gesture classes: {list(self.gesture_classes.keys())}")

            # Organize dataset
            base_path = image_dir if image_dir else csv_path.parent
            return self._organize_from_dataframe(df, image_col, label_col, base_path)

        except ImportError:
            print("pandas not installed. Install with: pip install pandas")
            return False
        except Exception as e:
            print(f"Error loading CSV dataset: {e}")
            return False

    def load_dataset_from_folders(self, root_dir: str) -> bool:
        """
        Load dataset from folder structure:
        root_dir/
        ├── class1/
        │   ├── image1.jpg
        │   └── image2.jpg
        └── class2/
            ├── image1.jpg
            └── image2.jpg
        """
        try:
            root_path = Path(root_dir)
            if not root_path.exists():
                print(f"Dataset directory not found: {root_path}")
                return False

            # Find all class directories
            class_dirs = [d for d in root_path.iterdir() if d.is_dir()]
            if not class_dirs:
                print(f"No class directories found in {root_path}")
                return False

            # Create gesture mapping
            class_names = sorted([d.name for d in class_dirs])
            self.gesture_classes = {name: idx for idx, name in enumerate(class_names)}

            print(f"Found gesture classes: {list(self.gesture_classes.keys())}")

            return self._organize_from_folders(root_path)

        except Exception as e:
            print(f"Error loading folder dataset: {e}")
            return False

    def _organize_from_dataframe(self, df, image_col: str, label_col: str, base_path: Path) -> bool:
        """Organize dataset from pandas DataFrame"""
        try:
            # Create organized directory structure
            organized_dir = self.dataset_path
            organized_dir.mkdir(parents=True, exist_ok=True)

            # Create train/val split directories
            train_dir = organized_dir / "train"
            val_dir = organized_dir / "val"
            train_dir.mkdir(exist_ok=True)
            val_dir.mkdir(exist_ok=True)

            # Create class subdirectories
            for gesture in self.gesture_classes.keys():
                (train_dir / gesture).mkdir(exist_ok=True)
                (val_dir / gesture).mkdir(exist_ok=True)

            # Process and copy images with train/val split
            train_count = 0
            val_count = 0

            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
                image_path = base_path / row[image_col]
                label = row[label_col]

                if not image_path.exists():
                    continue

                # 80/20 train/val split
                is_train = (idx % 5) != 0  # 80% train, 20% val
                target_dir = train_dir if is_train else val_dir

                # Copy image to appropriate directory
                target_path = target_dir / label / f"{idx:06d}.jpg"

                try:
                    # Load and resize image
                    image = cv2.imread(str(image_path))
                    if image is not None:
                        image = cv2.resize(image, (224, 224))
                        cv2.imwrite(str(target_path), image)

                        if is_train:
                            train_count += 1
                        else:
                            val_count += 1
                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")
                    continue

            # Save metadata
            self._save_metadata(train_count, val_count)
            print(f"Dataset organized: {train_count} train, {val_count} val images")
            return True

        except Exception as e:
            print(f"Error organizing dataset: {e}")
            return False

    def _organize_from_folders(self, root_path: Path) -> bool:
        """Organize dataset from folder structure"""
        try:
            # Create organized directory structure
            organized_dir = self.dataset_path
            organized_dir.mkdir(parents=True, exist_ok=True)

            train_dir = organized_dir / "train"
            val_dir = organized_dir / "val"
            train_dir.mkdir(exist_ok=True)
            val_dir.mkdir(exist_ok=True)

            # Create class subdirectories
            for gesture in self.gesture_classes.keys():
                (train_dir / gesture).mkdir(exist_ok=True)
                (val_dir / gesture).mkdir(exist_ok=True)

            train_count = 0
            val_count = 0

            # Process each class
            for class_dir in root_path.iterdir():
                if not class_dir.is_dir():
                    continue

                class_name = class_dir.name
                if class_name not in self.gesture_classes:
                    continue

                # Get all images in class directory
                image_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                    image_files.extend(class_dir.glob(ext))

                print(f"Processing {len(image_files)} images for class '{class_name}'")

                for idx, image_path in enumerate(tqdm(image_files, desc=f"Class {class_name}")):
                    # 80/20 train/val split
                    is_train = (idx % 5) != 0
                    target_dir = train_dir if is_train else val_dir

                    target_path = target_dir / class_name / f"{idx:06d}.jpg"

                    try:
                        # Load and resize image
                        image = cv2.imread(str(image_path))
                        if image is not None:
                            image = cv2.resize(image, (224, 224))
                            cv2.imwrite(str(target_path), image)

                            if is_train:
                                train_count += 1
                            else:
                                val_count += 1
                    except Exception as e:
                        print(f"Error processing image {image_path}: {e}")
                        continue

            # Save metadata
            self._save_metadata(train_count, val_count)
            print(f"Dataset organized: {train_count} train, {val_count} val images")
            return True

        except Exception as e:
            print(f"Error organizing dataset: {e}")
            return False

    def _save_metadata(self, train_count: int, val_count: int):
        """Save dataset metadata"""
        metadata = {
            "gesture_classes": self.gesture_classes,
            "num_classes": len(self.gesture_classes),
            "train_samples": train_count,
            "val_samples": val_count,
            "total_samples": train_count + val_count,
            "image_size": [224, 224],
            "dataset_format": "kaggle_organized"
        }

        metadata_path = self.dataset_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Metadata saved to {metadata_path}")

    def get_dataset_stats(self):
        """Display dataset statistics"""
        metadata_path = self.dataset_path / "metadata.json"

        if not metadata_path.exists():
            print("No dataset metadata found. Please organize dataset first.")
            return

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        print("\n" + "="*50)
        print("KAGGLE DATASET STATISTICS")
        print("="*50)
        print(f"Number of classes: {metadata['num_classes']}")
        print(f"Classes: {list(metadata['gesture_classes'].keys())}")
        print(f"Training samples: {metadata['train_samples']}")
        print(f"Validation samples: {metadata['val_samples']}")
        print(f"Total samples: {metadata['total_samples']}")
        print(f"Image size: {metadata['image_size']}")
        print("="*50)

    def validate_dataset(self) -> bool:
        """Validate that the organized dataset is ready for training"""
        try:
            # Check if metadata exists
            metadata_path = self.dataset_path / "metadata.json"
            if not metadata_path.exists():
                print("Dataset metadata not found")
                return False

            # Check train/val directories
            train_dir = self.dataset_path / "train"
            val_dir = self.dataset_path / "val"

            if not train_dir.exists() or not val_dir.exists():
                print("Train/val directories not found")
                return False

            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Check class directories
            for gesture in metadata['gesture_classes'].keys():
                train_class_dir = train_dir / gesture
                val_class_dir = val_dir / gesture

                if not train_class_dir.exists() or not val_class_dir.exists():
                    print(f"Missing directories for class: {gesture}")
                    return False

                # Count images
                train_images = len(list(train_class_dir.glob("*.jpg")))
                val_images = len(list(val_class_dir.glob("*.jpg")))

                if train_images == 0 and val_images == 0:
                    print(f"No images found for class: {gesture}")
                    return False

            print("Dataset validation passed!")
            return True

        except Exception as e:
            print(f"Error validating dataset: {e}")
            return False

def main():
    """Main function for command-line usage"""
    import argparse

    parser = argparse.ArgumentParser(description='Kaggle Dataset Loader for Gesture Recognition')
    parser.add_argument('--source-dir', required=True, help='Source dataset directory')
    parser.add_argument('--target-dir', default='data/kaggle_dataset', help='Target organized dataset directory')
    parser.add_argument('--format', choices=['folders', 'csv'], default='folders',
                       help='Dataset format: folders (class subdirectories) or csv (CSV file)')
    parser.add_argument('--csv-file', help='CSV file path (required if format=csv)')
    parser.add_argument('--image-dir', help='Image directory (for CSV format)')

    args = parser.parse_args()

    loader = KaggleDatasetLoader(args.target_dir)

    if args.format == 'folders':
        success = loader.load_dataset_from_folders(args.source_dir)
    elif args.format == 'csv':
        if not args.csv_file:
            print("CSV file required for CSV format")
            return
        success = loader.load_dataset_from_csv(args.csv_file, args.image_dir)

    if success:
        loader.get_dataset_stats()
        loader.validate_dataset()
        print(f"\nDataset ready for training! Use directory: {args.target_dir}")
    else:
        print("Dataset loading failed")

if __name__ == "__main__":
    main()