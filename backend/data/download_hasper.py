#!/usr/bin/env python3
"""
HaSPeR Dataset Downloader
Downloads the Hand Shadow Puppet Recognition dataset from Hugging Face
"""

import os
import sys
from pathlib import Path
from datasets import load_dataset
from PIL import Image
import numpy as np
from tqdm import tqdm
import json

class HaSPeRDownloader:
    def __init__(self, data_dir="data/hasper"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Gesture mappings for our 5 target classes
        self.gesture_map = {
            "dog": 0,
            "bird": 1,
            "rabbit": 2,
            "butterfly": 3,
            "snake": 4
        }

    def download_dataset(self):
        """Download HaSPeR dataset from Hugging Face"""
        print("Downloading HaSPeR dataset from Hugging Face...")

        try:
            dataset = load_dataset("Starscream-11813/HaSPeR")
            print(f"Dataset loaded successfully!")
            print(f"Available splits: {list(dataset.keys())}")

            return dataset

        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Please check your internet connection and try again.")
            return None

    def process_and_save(self, dataset):
        """Process dataset and save in organized structure"""
        if dataset is None:
            return False

        # Create directories for train/val splits
        train_dir = self.data_dir / "train"
        val_dir = self.data_dir / "val"

        for gesture in self.gesture_map.keys():
            (train_dir / gesture).mkdir(parents=True, exist_ok=True)
            (val_dir / gesture).mkdir(parents=True, exist_ok=True)

        # Process training data
        if "train" in dataset:
            self._process_split(dataset["train"], train_dir, "train")

        # Process validation data (or create from train if no val split exists)
        if "validation" in dataset:
            self._process_split(dataset["validation"], val_dir, "validation")
        elif "test" in dataset:
            self._process_split(dataset["test"], val_dir, "test")
        else:
            # Create validation split from training data (80/20 split)
            print("No validation split found, creating from train data...")
            self._create_val_split(dataset["train"], train_dir, val_dir)

        # Save metadata
        self._save_metadata()

        return True

    def _process_split(self, split_data, output_dir, split_name):
        """Process a single data split"""
        print(f"Processing {split_name} split...")

        processed_count = 0
        skipped_count = 0

        for idx, example in enumerate(tqdm(split_data)):
            try:
                # Get image and label
                image = example["image"]
                label = example.get("label", "")

                # Map label to our gesture classes
                gesture_class = self._map_label_to_gesture(label)
                if gesture_class is None:
                    skipped_count += 1
                    continue

                # Save image
                image_path = output_dir / gesture_class / f"{split_name}_{idx:06d}.jpg"
                if isinstance(image, Image.Image):
                    image.save(image_path)
                else:
                    # Convert numpy array to PIL Image
                    if isinstance(image, np.ndarray):
                        image = Image.fromarray(image)
                        image.save(image_path)

                processed_count += 1

            except Exception as e:
                print(f"Error processing example {idx}: {e}")
                skipped_count += 1
                continue

        print(f"Processed {processed_count} images, skipped {skipped_count}")

    def _map_label_to_gesture(self, label):
        """Map dataset labels to our gesture classes"""
        if not isinstance(label, str):
            label = str(label).lower()
        else:
            label = label.lower()

        # Direct mapping
        if label in self.gesture_map:
            return label

        # Fuzzy matching for common variations
        label_mappings = {
            "dog": ["dog", "puppy", "canine"],
            "bird": ["bird", "eagle", "hawk", "dove", "sparrow"],
            "rabbit": ["rabbit", "bunny", "hare"],
            "butterfly": ["butterfly", "moth"],
            "snake": ["snake", "serpent", "cobra"]
        }

        for gesture, variations in label_mappings.items():
            if any(var in label for var in variations):
                return gesture

        return None  # Skip if no mapping found

    def _create_val_split(self, train_data, train_dir, val_dir):
        """Create validation split from training data"""
        # Move 20% of training images to validation
        for gesture in self.gesture_map.keys():
            gesture_train_dir = train_dir / gesture
            gesture_val_dir = val_dir / gesture

            if gesture_train_dir.exists():
                images = list(gesture_train_dir.glob("*.jpg"))
                val_count = len(images) // 5  # 20% for validation

                for i, img_path in enumerate(images[:val_count]):
                    new_name = img_path.name.replace("train_", "val_")
                    img_path.rename(gesture_val_dir / new_name)

    def _save_metadata(self):
        """Save dataset metadata"""
        metadata = {
            "gesture_classes": self.gesture_map,
            "num_classes": len(self.gesture_map),
            "data_structure": {
                "train": str(self.data_dir / "train"),
                "val": str(self.data_dir / "val")
            },
            "preprocessing_notes": "Images should be resized to 224x224 for model training"
        }

        with open(self.data_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Metadata saved to {self.data_dir / 'metadata.json'}")

    def get_dataset_stats(self):
        """Print dataset statistics"""
        if not self.data_dir.exists():
            print("Dataset not found. Please download first.")
            return

        print("\n=== Dataset Statistics ===")
        for split in ["train", "val"]:
            split_dir = self.data_dir / split
            if split_dir.exists():
                print(f"\n{split.capitalize()} split:")
                total = 0
                for gesture in self.gesture_map.keys():
                    gesture_dir = split_dir / gesture
                    if gesture_dir.exists():
                        count = len(list(gesture_dir.glob("*.jpg")))
                        print(f"  {gesture}: {count} images")
                        total += count
                print(f"  Total: {total} images")

def main():
    """Main function"""
    downloader = HaSPeRDownloader()

    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        downloader.get_dataset_stats()
        return

    # Download and process dataset
    dataset = downloader.download_dataset()
    if dataset:
        success = downloader.process_and_save(dataset)
        if success:
            print("\nDataset download and processing completed!")
            downloader.get_dataset_stats()
        else:
            print("Failed to process dataset.")
    else:
        print("Failed to download dataset.")

if __name__ == "__main__":
    main()