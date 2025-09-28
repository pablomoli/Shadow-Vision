#!/usr/bin/env python3
"""
Process HuggingFace Dataset with Real MediaPipe

This script runs inside the MediaPipe Docker container and processes
the entire dataset to extract real MediaPipe hand landmarks.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from collections import Counter
import time

# Import our real MediaPipe extractor
from backend.data.mediapipe_extractor_real import RealMediaPipeExtractor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MediaPipeDatasetProcessor:
    """Process entire dataset with real MediaPipe landmarks."""

    def __init__(self):
        """Initialize MediaPipe dataset processor."""
        self.extractor = RealMediaPipeExtractor()
        self.classes = ['bird', 'cat', 'llama', 'rabbit', 'deer', 'dog', 'snail', 'swan']

        # Results storage
        self.processed_data = []
        self.processing_stats = {
            'total_images': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'per_class_stats': {cls: {'total': 0, 'success': 0} for cls in self.classes}
        }

    def process_dataset(self, data_dir: str = "data/raw"):
        """Process entire dataset with MediaPipe."""
        logger.info("Processing dataset with real MediaPipe landmarks...")
        start_time = time.time()

        data_path = Path(data_dir)
        if not data_path.exists():
            logger.error(f"Dataset directory not found: {data_path}")
            return False

        # Process both train and validation splits
        for split in ['train', 'val']:
            split_dir = data_path / split
            if not split_dir.exists():
                logger.warning(f"Split directory not found: {split_dir}")
                continue

            logger.info(f"Processing {split} split...")

            for class_name in self.classes:
                class_dir = split_dir / class_name
                if not class_dir.exists():
                    logger.warning(f"Class directory not found: {class_dir}")
                    continue

                # Get all images for this class
                image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
                class_total = len(image_files)

                logger.info(f"  Processing {class_name}: {class_total} images")

                self.processing_stats['per_class_stats'][class_name]['total'] += class_total
                self.processing_stats['total_images'] += class_total

                class_success = 0
                for img_path in image_files:
                    try:
                        # Process image with MediaPipe
                        result = self.extractor.process_dataset_image(str(img_path))

                        if result is not None:
                            # Add metadata
                            result.update({
                                'class_name': class_name,
                                'split': split,
                                'class_index': self.classes.index(class_name)
                            })

                            self.processed_data.append(result)
                            self.processing_stats['successful_extractions'] += 1
                            self.processing_stats['per_class_stats'][class_name]['success'] += 1
                            class_success += 1

                        else:
                            self.processing_stats['failed_extractions'] += 1

                    except Exception as e:
                        logger.error(f"Error processing {img_path}: {e}")
                        self.processing_stats['failed_extractions'] += 1

                logger.info(f"    {class_name}: {class_success}/{class_total} successful "
                          f"({class_success/class_total*100:.1f}%)")

        # Processing summary
        processing_time = time.time() - start_time
        total_processed = self.processing_stats['successful_extractions']
        total_images = self.processing_stats['total_images']
        success_rate = total_processed / total_images * 100 if total_images > 0 else 0

        logger.info(f"\nMediaPipe processing completed in {processing_time:.1f}s")
        logger.info(f"Successful extractions: {total_processed}/{total_images} ({success_rate:.1f}%)")

        # Per-class summary
        logger.info("\nPer-class MediaPipe extraction rates:")
        for class_name in self.classes:
            stats = self.processing_stats['per_class_stats'][class_name]
            if stats['total'] > 0:
                rate = stats['success'] / stats['total'] * 100
                logger.info(f"  {class_name}: {stats['success']}/{stats['total']} ({rate:.1f}%)")

        return len(self.processed_data) > 0

    def save_mediapipe_features(self, output_dir: str = "data/mediapipe"):
        """Save extracted MediaPipe features."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        logger.info(f"Saving MediaPipe features to {output_path}")

        if not self.processed_data:
            logger.error("No processed data to save!")
            return False

        # Separate features and labels
        features = []
        labels = []
        metadata = []

        for item in self.processed_data:
            features.append(item['features'])
            labels.append(item['class_name'])
            metadata.append({
                'image_path': item['image_path'],
                'class_name': item['class_name'],
                'class_index': item['class_index'],
                'split': item['split'],
                'feature_count': item['feature_count']
            })

        # Convert to numpy arrays
        features_array = np.array(features, dtype=np.float32)
        labels_array = np.array(labels)

        logger.info(f"Features shape: {features_array.shape}")
        logger.info(f"Labels shape: {labels_array.shape}")

        # Save features
        features_path = output_path / "mediapipe_features.npz"
        np.savez_compressed(features_path,
                           features=features_array,
                           labels=labels_array)

        # Save metadata
        metadata_path = output_path / "mediapipe_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                'processing_stats': self.processing_stats,
                'feature_count': features_array.shape[1],
                'sample_count': features_array.shape[0],
                'classes': self.classes,
                'metadata': metadata
            }, f, indent=2)

        # Save CSV for easy loading
        df = pd.DataFrame({
            'image_path': [item['image_path'] for item in metadata],
            'class_name': [item['class_name'] for item in metadata],
            'class_index': [item['class_index'] for item in metadata],
            'split': [item['split'] for item in metadata]
        })

        csv_path = output_path / "mediapipe_dataset.csv"
        df.to_csv(csv_path, index=False)

        logger.info(f"MediaPipe features saved:")
        logger.info(f"  Features: {features_path}")
        logger.info(f"  Metadata: {metadata_path}")
        logger.info(f"  CSV: {csv_path}")

        return True

    def cleanup(self):
        """Clean up resources."""
        self.extractor.cleanup()

def main():
    """Main processing function."""
    print("MediaPipe Dataset Processing")
    print("=" * 35)
    print("Processing HuggingFace dataset with REAL MediaPipe landmarks")
    print("Expected: 89 features per image (vs 49 pixel-based)")
    print()

    processor = MediaPipeDatasetProcessor()

    try:
        # Process dataset
        success = processor.process_dataset()

        if success:
            # Save features
            processor.save_mediapipe_features()

            print("\n✅ MediaPipe dataset processing completed!")
            print("\nKey improvements over pixel-based approach:")
            print("- Real hand landmark detection (21 precise points)")
            print("- 89 advanced features vs 49 pixel-based")
            print("- Rotation, scale, and lighting invariant")
            print("- Expected accuracy: 95%+ vs current 81%")
            print("\nNext step: Train model with MediaPipe features")

        else:
            print("❌ MediaPipe dataset processing failed!")
            return 1

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1

    finally:
        processor.cleanup()

    return 0

if __name__ == "__main__":
    exit(main())