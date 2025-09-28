#!/usr/bin/env python3
"""
Local Data Loader for Pose/Gesture Classification
Replaces database-dependent data loading with local Kaggle dataset support
"""

import os
import json
import csv
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import Counter
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LocalDatasetLoader:
    """
    Comprehensive loader for local Kaggle pose/gesture datasets
    Supports multiple dataset formats and automatic organization
    """

    def __init__(self, dataset_path: str = "data/kaggle_dataset"):
        self.dataset_path = Path(dataset_path)
        self.dataset_path.mkdir(parents=True, exist_ok=True)

        self.class_names = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.dataset_info = {}

    def load_from_folder_structure(self, source_path: str, copy_files: bool = True) -> bool:
        """
        Load dataset from standard folder structure:
        source_path/
        ├── class1/
        │   ├── image1.jpg
        │   └── image2.jpg
        └── class2/
            ├── image1.jpg
            └── image2.jpg
        """
        try:
            source_path = Path(source_path)
            if not source_path.exists():
                logger.error(f"Source path does not exist: {source_path}")
                return False

            # Find all class directories
            class_dirs = [d for d in source_path.iterdir() if d.is_dir()]
            if not class_dirs:
                logger.error(f"No class directories found in {source_path}")
                return False

            # Create class mapping
            self.class_names = sorted([d.name for d in class_dirs])
            self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
            self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

            logger.info(f"Found {len(self.class_names)} classes: {self.class_names}")

            if copy_files:
                return self._organize_folder_dataset(source_path)
            else:
                # Just create metadata without copying files
                return self._create_metadata_from_source(source_path)

        except Exception as e:
            logger.error(f"Error loading folder structure dataset: {e}")
            return False

    def load_from_csv(self, csv_path: str, image_column: str = "image",
                     label_column: str = "label", image_dir: str = None) -> bool:
        """
        Load dataset from CSV format:
        CSV columns: image_path, label
        """
        try:
            csv_path = Path(csv_path)
            if not csv_path.exists():
                logger.error(f"CSV file does not exist: {csv_path}")
                return False

            df = pd.read_csv(csv_path)
            logger.info(f"Loaded CSV with {len(df)} entries")

            # Validate columns
            if image_column not in df.columns:
                logger.error(f"Image column '{image_column}' not found in CSV")
                return False
            if label_column not in df.columns:
                logger.error(f"Label column '{label_column}' not found in CSV")
                return False

            # Create class mapping
            unique_labels = sorted(df[label_column].unique())
            self.class_names = unique_labels
            self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
            self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

            logger.info(f"Found {len(self.class_names)} classes: {self.class_names}")

            # Determine base path for images
            base_path = Path(image_dir) if image_dir else csv_path.parent

            return self._organize_csv_dataset(df, image_column, label_column, base_path)

        except Exception as e:
            logger.error(f"Error loading CSV dataset: {e}")
            return False

    def load_from_annotations(self, annotations_path: str, images_dir: str) -> bool:
        """
        Load dataset from annotation file (JSON format)
        Expected format: {"image_path": "label", ...}
        """
        try:
            annotations_path = Path(annotations_path)
            images_dir = Path(images_dir)

            if not annotations_path.exists():
                logger.error(f"Annotations file does not exist: {annotations_path}")
                return False
            if not images_dir.exists():
                logger.error(f"Images directory does not exist: {images_dir}")
                return False

            with open(annotations_path, 'r') as f:
                annotations = json.load(f)

            # Create class mapping
            unique_labels = sorted(set(annotations.values()))
            self.class_names = unique_labels
            self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
            self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

            logger.info(f"Found {len(self.class_names)} classes: {self.class_names}")

            return self._organize_annotation_dataset(annotations, images_dir)

        except Exception as e:
            logger.error(f"Error loading annotation dataset: {e}")
            return False

    def _organize_folder_dataset(self, source_path: Path) -> bool:
        """Organize folder-based dataset with train/val split"""
        try:
            # Create organized structure
            train_dir = self.dataset_path / "train"
            val_dir = self.dataset_path / "val"
            train_dir.mkdir(exist_ok=True)
            val_dir.mkdir(exist_ok=True)

            # Create class subdirectories
            for class_name in self.class_names:
                (train_dir / class_name).mkdir(exist_ok=True)
                (val_dir / class_name).mkdir(exist_ok=True)

            total_train = 0
            total_val = 0
            class_counts = {}

            # Process each class
            for class_name in self.class_names:
                class_dir = source_path / class_name
                if not class_dir.exists():
                    continue

                # Get all image files
                image_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
                    image_files.extend(class_dir.glob(ext))
                    image_files.extend(class_dir.glob(ext.upper()))

                # Shuffle for random split
                random.shuffle(image_files)

                # 80/20 train/val split
                split_idx = int(len(image_files) * 0.8)
                train_files = image_files[:split_idx]
                val_files = image_files[split_idx:]

                # Copy training files
                for i, img_path in enumerate(train_files):
                    target_path = train_dir / class_name / f"{i:06d}.jpg"
                    if self._copy_and_resize_image(img_path, target_path):
                        total_train += 1

                # Copy validation files
                for i, img_path in enumerate(val_files):
                    target_path = val_dir / class_name / f"{i:06d}.jpg"
                    if self._copy_and_resize_image(img_path, target_path):
                        total_val += 1

                class_counts[class_name] = {
                    'train': len(train_files),
                    'val': len(val_files),
                    'total': len(image_files)
                }

                logger.info(f"Class {class_name}: {len(train_files)} train, {len(val_files)} val")

            # Save metadata
            self._save_metadata(total_train, total_val, class_counts)
            logger.info(f"Dataset organized: {total_train} train, {total_val} val images")
            return True

        except Exception as e:
            logger.error(f"Error organizing folder dataset: {e}")
            return False

    def _organize_csv_dataset(self, df: pd.DataFrame, image_col: str,
                             label_col: str, base_path: Path) -> bool:
        """Organize CSV-based dataset with train/val split"""
        try:
            # Create organized structure
            train_dir = self.dataset_path / "train"
            val_dir = self.dataset_path / "val"
            train_dir.mkdir(exist_ok=True)
            val_dir.mkdir(exist_ok=True)

            # Create class subdirectories
            for class_name in self.class_names:
                (train_dir / class_name).mkdir(exist_ok=True)
                (val_dir / class_name).mkdir(exist_ok=True)

            total_train = 0
            total_val = 0
            class_counts = {name: {'train': 0, 'val': 0, 'total': 0} for name in self.class_names}

            # Group by class and split
            for class_name in self.class_names:
                class_df = df[df[label_col] == class_name].copy()
                class_df = class_df.sample(frac=1).reset_index(drop=True)  # Shuffle

                # 80/20 train/val split
                split_idx = int(len(class_df) * 0.8)
                train_df = class_df[:split_idx]
                val_df = class_df[split_idx:]

                # Process training files
                for i, (_, row) in enumerate(train_df.iterrows()):
                    img_path = base_path / row[image_col]
                    target_path = train_dir / class_name / f"{i:06d}.jpg"
                    if self._copy_and_resize_image(img_path, target_path):
                        total_train += 1
                        class_counts[class_name]['train'] += 1

                # Process validation files
                for i, (_, row) in enumerate(val_df.iterrows()):
                    img_path = base_path / row[image_col]
                    target_path = val_dir / class_name / f"{i:06d}.jpg"
                    if self._copy_and_resize_image(img_path, target_path):
                        total_val += 1
                        class_counts[class_name]['val'] += 1

                class_counts[class_name]['total'] = len(class_df)
                logger.info(f"Class {class_name}: {len(train_df)} train, {len(val_df)} val")

            # Save metadata
            self._save_metadata(total_train, total_val, class_counts)
            logger.info(f"Dataset organized: {total_train} train, {total_val} val images")
            return True

        except Exception as e:
            logger.error(f"Error organizing CSV dataset: {e}")
            return False

    def _organize_annotation_dataset(self, annotations: Dict, images_dir: Path) -> bool:
        """Organize annotation-based dataset"""
        try:
            # Create organized structure
            train_dir = self.dataset_path / "train"
            val_dir = self.dataset_path / "val"
            train_dir.mkdir(exist_ok=True)
            val_dir.mkdir(exist_ok=True)

            # Create class subdirectories
            for class_name in self.class_names:
                (train_dir / class_name).mkdir(exist_ok=True)
                (val_dir / class_name).mkdir(exist_ok=True)

            # Group annotations by class
            class_images = {name: [] for name in self.class_names}
            for img_path, label in annotations.items():
                if label in class_images:
                    class_images[label].append(img_path)

            total_train = 0
            total_val = 0
            class_counts = {}

            # Process each class
            for class_name, img_paths in class_images.items():
                random.shuffle(img_paths)  # Shuffle for random split

                # 80/20 train/val split
                split_idx = int(len(img_paths) * 0.8)
                train_paths = img_paths[:split_idx]
                val_paths = img_paths[split_idx:]

                # Process training files
                for i, img_path in enumerate(train_paths):
                    source_path = images_dir / img_path
                    target_path = train_dir / class_name / f"{i:06d}.jpg"
                    if self._copy_and_resize_image(source_path, target_path):
                        total_train += 1

                # Process validation files
                for i, img_path in enumerate(val_paths):
                    source_path = images_dir / img_path
                    target_path = val_dir / class_name / f"{i:06d}.jpg"
                    if self._copy_and_resize_image(source_path, target_path):
                        total_val += 1

                class_counts[class_name] = {
                    'train': len(train_paths),
                    'val': len(val_paths),
                    'total': len(img_paths)
                }

                logger.info(f"Class {class_name}: {len(train_paths)} train, {len(val_paths)} val")

            # Save metadata
            self._save_metadata(total_train, total_val, class_counts)
            logger.info(f"Dataset organized: {total_train} train, {total_val} val images")
            return True

        except Exception as e:
            logger.error(f"Error organizing annotation dataset: {e}")
            return False

    def _copy_and_resize_image(self, source_path: Path, target_path: Path,
                              target_size: Tuple[int, int] = (224, 224)) -> bool:
        """Copy and resize image to target location"""
        try:
            if not source_path.exists():
                logger.warning(f"Source image not found: {source_path}")
                return False

            # Read image
            image = cv2.imread(str(source_path))
            if image is None:
                logger.warning(f"Could not read image: {source_path}")
                return False

            # Resize image
            image = cv2.resize(image, target_size)

            # Save image
            success = cv2.imwrite(str(target_path), image)
            if not success:
                logger.warning(f"Could not save image: {target_path}")
                return False

            return True

        except Exception as e:
            logger.warning(f"Error processing image {source_path}: {e}")
            return False

    def _save_metadata(self, train_count: int, val_count: int, class_counts: Dict):
        """Save dataset metadata"""
        metadata = {
            "dataset_info": {
                "num_classes": len(self.class_names),
                "class_names": self.class_names,
                "class_to_idx": self.class_to_idx,
                "idx_to_class": self.idx_to_class,
                "train_samples": train_count,
                "val_samples": val_count,
                "total_samples": train_count + val_count,
                "image_size": [224, 224],
                "dataset_format": "organized_local"
            },
            "class_counts": class_counts,
            "split_ratio": "80/20 train/val"
        }

        metadata_path = self.dataset_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Metadata saved to {metadata_path}")

    def _create_metadata_from_source(self, source_path: Path) -> bool:
        """Create metadata without copying files (for large datasets)"""
        try:
            class_counts = {}
            total_images = 0

            for class_name in self.class_names:
                class_dir = source_path / class_name
                if class_dir.exists():
                    # Count images
                    image_count = 0
                    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
                        image_count += len(list(class_dir.glob(ext)))
                        image_count += len(list(class_dir.glob(ext.upper())))

                    class_counts[class_name] = {
                        'total': image_count,
                        'train': int(image_count * 0.8),
                        'val': image_count - int(image_count * 0.8)
                    }
                    total_images += image_count

            # Save metadata with source path reference
            metadata = {
                "dataset_info": {
                    "num_classes": len(self.class_names),
                    "class_names": self.class_names,
                    "class_to_idx": self.class_to_idx,
                    "idx_to_class": self.idx_to_class,
                    "total_samples": total_images,
                    "image_size": [224, 224],
                    "dataset_format": "source_reference",
                    "source_path": str(source_path)
                },
                "class_counts": class_counts,
                "split_ratio": "80/20 train/val"
            }

            metadata_path = self.dataset_path / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Metadata created for {total_images} images")
            return True

        except Exception as e:
            logger.error(f"Error creating metadata: {e}")
            return False

    def get_dataset_statistics(self) -> Dict:
        """Get comprehensive dataset statistics"""
        try:
            metadata_path = self.dataset_path / "metadata.json"
            if not metadata_path.exists():
                logger.error("No dataset metadata found. Please organize dataset first.")
                return {}

            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            return metadata

        except Exception as e:
            logger.error(f"Error getting dataset statistics: {e}")
            return {}

    def print_dataset_statistics(self):
        """Print formatted dataset statistics"""
        stats = self.get_dataset_statistics()
        if not stats:
            return

        info = stats.get("dataset_info", {})
        class_counts = stats.get("class_counts", {})

        print("\n" + "="*60)
        print("LOCAL DATASET STATISTICS")
        print("="*60)
        print(f"Number of classes: {info.get('num_classes', 0)}")
        print(f"Total samples: {info.get('total_samples', 0)}")
        print(f"Training samples: {info.get('train_samples', 0)}")
        print(f"Validation samples: {info.get('val_samples', 0)}")
        print(f"Image size: {info.get('image_size', [224, 224])}")
        print(f"Dataset format: {info.get('dataset_format', 'unknown')}")

        print(f"\nClass distribution:")
        for class_name in info.get('class_names', []):
            if class_name in class_counts:
                counts = class_counts[class_name]
                print(f"  {class_name}: {counts.get('total', 0)} total "
                      f"({counts.get('train', 0)} train, {counts.get('val', 0)} val)")

        print("="*60)

    def validate_dataset(self) -> bool:
        """Validate that the organized dataset is ready for training"""
        try:
            metadata_path = self.dataset_path / "metadata.json"
            if not metadata_path.exists():
                logger.error("Dataset metadata not found")
                return False

            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            info = metadata.get("dataset_info", {})

            # Check if using source reference or organized structure
            if info.get("dataset_format") == "source_reference":
                source_path = Path(info.get("source_path", ""))
                if not source_path.exists():
                    logger.error(f"Source dataset path not found: {source_path}")
                    return False
                logger.info("Dataset validation passed (source reference mode)")
                return True

            # Check organized structure
            train_dir = self.dataset_path / "train"
            val_dir = self.dataset_path / "val"

            if not train_dir.exists() or not val_dir.exists():
                logger.error("Train/val directories not found")
                return False

            # Check class directories
            for class_name in info.get('class_names', []):
                train_class_dir = train_dir / class_name
                val_class_dir = val_dir / class_name

                if not train_class_dir.exists() or not val_class_dir.exists():
                    logger.error(f"Missing directories for class: {class_name}")
                    return False

            logger.info("Dataset validation passed")
            return True

        except Exception as e:
            logger.error(f"Error validating dataset: {e}")
            return False


class PoseDataset(Dataset):
    """
    PyTorch Dataset class for pose/gesture classification
    Works with organized local datasets
    """

    def __init__(self, dataset_path: str, split: str = "train", transform=None):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.transform = transform

        # Load metadata
        metadata_path = self.dataset_path / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        self.dataset_info = metadata["dataset_info"]
        self.class_to_idx = self.dataset_info["class_to_idx"]
        self.idx_to_class = self.dataset_info["idx_to_class"]
        self.class_names = self.dataset_info["class_names"]

        # Load image paths and labels
        self.image_paths, self.labels = self._load_split_data()

    def _load_split_data(self) -> Tuple[List[str], List[int]]:
        """Load image paths and labels for the specified split"""
        image_paths = []
        labels = []

        if self.dataset_info.get("dataset_format") == "source_reference":
            # Load from source directory
            source_path = Path(self.dataset_info["source_path"])
            return self._load_from_source(source_path)
        else:
            # Load from organized directory
            split_dir = self.dataset_path / self.split

            if not split_dir.exists():
                raise FileNotFoundError(f"Split directory not found: {split_dir}")

            for class_name in self.class_names:
                class_dir = split_dir / class_name
                if class_dir.exists():
                    class_idx = self.class_to_idx[class_name]
                    for img_path in class_dir.glob("*.jpg"):
                        image_paths.append(str(img_path))
                        labels.append(class_idx)

        return image_paths, labels

    def _load_from_source(self, source_path: Path) -> Tuple[List[str], List[int]]:
        """Load data directly from source (for large datasets)"""
        image_paths = []
        labels = []

        for class_name in self.class_names:
            class_dir = source_path / class_name
            if class_dir.exists():
                class_idx = self.class_to_idx[class_name]

                # Get all images
                all_images = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
                    all_images.extend(class_dir.glob(ext))
                    all_images.extend(class_dir.glob(ext.upper()))

                # Sort for consistent splitting
                all_images.sort()

                # 80/20 split
                split_idx = int(len(all_images) * 0.8)
                if self.split == "train":
                    selected_images = all_images[:split_idx]
                else:  # val
                    selected_images = all_images[split_idx:]

                for img_path in selected_images:
                    image_paths.append(str(img_path))
                    labels.append(class_idx)

        return image_paths, labels

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        try:
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Could not load image: {img_path}")

            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convert to PIL Image for transforms
            image = Image.fromarray(image)

            # Apply transforms
            if self.transform:
                image = self.transform(image)
            else:
                # Default transform
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                ])
                image = transform(image)

            return image, label

        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a blank image tensor in case of error
            blank_image = torch.zeros(3, 224, 224)
            return blank_image, label


def create_data_loaders(dataset_path: str, batch_size: int = 32,
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Create training and validation data loaders

    Returns:
        train_loader, val_loader, idx_to_class mapping
    """

    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = PoseDataset(dataset_path, split="train", transform=train_transform)
    val_dataset = PoseDataset(dataset_path, split="val", transform=val_transform)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    logger.info(f"Created data loaders: {len(train_dataset)} train, {len(val_dataset)} val samples")

    return train_loader, val_loader, train_dataset.idx_to_class


def main():
    """Command-line interface for data loading"""
    import argparse

    parser = argparse.ArgumentParser(description='Local Dataset Loader for Pose Classification')
    parser.add_argument('--source', required=True, help='Source dataset path')
    parser.add_argument('--target', default='data/kaggle_dataset', help='Target dataset path')
    parser.add_argument('--format', choices=['folder', 'csv', 'annotations'],
                       default='folder', help='Dataset format')
    parser.add_argument('--csv-file', help='CSV file path (for CSV format)')
    parser.add_argument('--image-column', default='image', help='Image column name in CSV')
    parser.add_argument('--label-column', default='label', help='Label column name in CSV')
    parser.add_argument('--annotations-file', help='Annotations JSON file (for annotations format)')
    parser.add_argument('--images-dir', help='Images directory (for annotations format)')
    parser.add_argument('--no-copy', action='store_true',
                       help='Create metadata only, don\'t copy files')

    args = parser.parse_args()

    # Create loader
    loader = LocalDatasetLoader(args.target)

    # Load dataset based on format
    success = False
    if args.format == 'folder':
        success = loader.load_from_folder_structure(args.source, copy_files=not args.no_copy)
    elif args.format == 'csv':
        if not args.csv_file:
            logger.error("CSV file required for CSV format")
            return
        success = loader.load_from_csv(args.csv_file, args.image_column,
                                     args.label_column, args.images_dir)
    elif args.format == 'annotations':
        if not args.annotations_file or not args.images_dir:
            logger.error("Annotations file and images directory required")
            return
        success = loader.load_from_annotations(args.annotations_file, args.images_dir)

    if success:
        loader.print_dataset_statistics()
        if loader.validate_dataset():
            print(f"\n✅ Dataset ready for training at: {args.target}")
        else:
            print(f"\n❌ Dataset validation failed")
    else:
        print(f"\n❌ Dataset loading failed")


if __name__ == "__main__":
    main()