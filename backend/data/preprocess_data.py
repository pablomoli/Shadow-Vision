#!/usr/bin/env python3
"""
Data Preprocessing Pipeline for Gesture Recognition
Handles image preprocessing, augmentation, and dataset preparation
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import json
import cv2
# Import our new local data loader
from .data_loader import PoseDataset, create_data_loaders as create_local_data_loaders

class GestureDataset(Dataset):
    """Custom dataset for gesture recognition"""

    def __init__(self, data_dir: str, split: str = "train", transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform

        # Load metadata
        metadata_path = self.data_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        self.gesture_to_idx = self.metadata["gesture_classes"]
        self.idx_to_gesture = {v: k for k, v in self.gesture_to_idx.items()}

        # Load image paths and labels
        self.images, self.labels = self._load_dataset()

    def _load_dataset(self):
        """Load image paths and corresponding labels"""
        images = []
        labels = []

        split_dir = self.data_dir / self.split

        for gesture, idx in self.gesture_to_idx.items():
            gesture_dir = split_dir / gesture
            if gesture_dir.exists():
                for img_path in gesture_dir.glob("*.jpg"):
                    images.append(str(img_path))
                    labels.append(idx)

        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label

class DataPreprocessor:
    """Data preprocessing and augmentation pipeline"""

    def __init__(self, img_size=(224, 224)):
        self.img_size = img_size

    def get_train_transforms(self):
        """Get training data transformations with augmentation"""
        return transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def get_val_transforms(self):
        """Get validation/test data transformations"""
        return transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def get_inference_transforms(self):
        """Get transformations for real-time inference"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

def preprocess_frame_for_inference(frame: np.ndarray, preprocessor: DataPreprocessor) -> torch.Tensor:
    """Preprocess a single frame for model inference"""
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Apply preprocessing
    transform = preprocessor.get_inference_transforms()
    tensor = transform(frame_rgb)

    # Add batch dimension
    return tensor.unsqueeze(0)

def create_data_loaders(data_dir: str, batch_size: int = 32, num_workers: int = 4):
    """Create training and validation data loaders"""

    preprocessor = DataPreprocessor()

    # Create datasets
    train_dataset = GestureDataset(
        data_dir=data_dir,
        split="train",
        transform=preprocessor.get_train_transforms()
    )

    val_dataset = GestureDataset(
        data_dir=data_dir,
        split="val",
        transform=preprocessor.get_val_transforms()
    )

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

    return train_loader, val_loader, train_dataset.idx_to_gesture

def visualize_augmentations(data_dir: str, num_samples: int = 5):
    """Visualize data augmentations"""
    import matplotlib.pyplot as plt

    preprocessor = DataPreprocessor()

    # Original transforms (no augmentation)
    original_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Augmented transforms
    aug_transform = preprocessor.get_train_transforms()

    dataset_orig = GestureDataset(data_dir, "train", original_transform)
    dataset_aug = GestureDataset(data_dir, "train", aug_transform)

    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))

    for i in range(num_samples):
        # Original
        img_orig, label = dataset_orig[i]
        axes[0, i].imshow(img_orig.permute(1, 2, 0))
        axes[0, i].set_title(f"Original: {dataset_orig.idx_to_gesture[label]}")
        axes[0, i].axis('off')

        # Augmented
        img_aug, _ = dataset_aug[i]
        # Denormalize for display
        img_aug = img_aug * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + \
                 torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        img_aug = torch.clamp(img_aug, 0, 1)

        axes[1, i].imshow(img_aug.permute(1, 2, 0))
        axes[1, i].set_title("Augmented")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig("data_augmentation_examples.png")
    print("Augmentation examples saved as 'data_augmentation_examples.png'")

def main():
    """Test the preprocessing pipeline"""
    data_dir = "data/kaggle_dataset"

    if not Path(data_dir).exists():
        print("Dataset not found. Please organize your Kaggle dataset first using data_loader.py")
        return

    print("Creating data loaders...")
    train_loader, val_loader, idx_to_gesture = create_data_loaders(data_dir, batch_size=8)

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Gesture classes: {idx_to_gesture}")

    # Test a batch
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}: {images.shape}, {labels.shape}")
        if batch_idx == 0:
            break

    # Visualize augmentations
    print("Generating augmentation examples...")
    visualize_augmentations(data_dir)

if __name__ == "__main__":
    main()