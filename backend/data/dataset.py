#!/usr/bin/env python3
"""
Shadow-Vision Dataset Utility (Utility #1)

Downloads HuggingFace dataset: adominguez07/Animal-Puppet-Dataset
Contains two splits: Train/ (clean training images) and Val/ (validation images, harder conditions)

Expected Schema:
{
  "image": <PIL.Image (640x640)>,
  "label": 0-7,
  "label_text": "Bird" | "Dog" | ...
}

Classes: bird, dog, dinosaur, duck, goat, rabbit, snail, wolf

Outputs:
- data/raw/train/<label_slug>/...
- data/raw/val/<label_slug>/...
- data/splits/train.csv
- data/splits/val.csv
- data/splits/labels.json
"""

import os
import json
import csv
from pathlib import Path
from collections import Counter
import yaml
from datasets import load_dataset
from PIL import Image
import pandas as pd


def load_config(config_path="config/config.yaml"):
    """Load configuration from YAML file."""
    config_file = Path(__file__).parent.parent.parent / config_path
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def create_directories(config):
    """Create necessary directories for dataset organization."""
    base_dir = Path(__file__).parent.parent.parent

    # Create data directories
    raw_dir = base_dir / config['dataset']['out_dir']
    splits_dir = base_dir / config['dataset']['splits_dir']

    for split in ['train', 'val']:
        split_dir = raw_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        # Create class subdirectories
        for class_name in config['project']['classes']:
            class_dir = split_dir / class_name.lower().replace(' ', '_')
            class_dir.mkdir(parents=True, exist_ok=True)

    splits_dir.mkdir(parents=True, exist_ok=True)

    return raw_dir, splits_dir


def download_and_organize_dataset(config):
    """Download dataset from HuggingFace and organize locally."""
    print(f"Downloading dataset: {config['dataset']['huggingface_repo']}")

    # Load dataset from HuggingFace
    dataset = load_dataset(config['dataset']['huggingface_repo'])

    base_dir = Path(__file__).parent.parent.parent
    raw_dir, splits_dir = create_directories(config)

    # The HuggingFace dataset has all samples in 'train' split
    # We need to create our own train/val split for maximum accuracy
    all_data = dataset['train']

    print(f"Total samples in dataset: {len(all_data)}")
    print(f"Classes: {all_data.features['label'].names}")

    # Create stratified train/val split (80/20)
    from sklearn.model_selection import train_test_split

    # Get all samples and labels
    all_samples = list(range(len(all_data)))
    all_labels = [all_data[i]['label'] for i in all_samples]

    # Stratified split to ensure balanced representation
    train_indices, val_indices = train_test_split(
        all_samples,
        test_size=0.2,  # 20% for validation
        stratify=all_labels,
        random_state=42  # For reproducible splits
    )

    print(f"Train samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")

    # Process both splits
    split_data = {'train': [], 'val': []}
    class_counts = {'train': Counter(), 'val': Counter()}
    split_indices = {'train': train_indices, 'val': val_indices}

    for split_name in ['train', 'val']:
        print(f"\nProcessing {split_name} split...")
        split_data_list = []
        indices = split_indices[split_name]

        for count, data_idx in enumerate(indices):
            sample = all_data[data_idx]
            image = sample['image']
            label = sample['label']

            # Map to our project classes (handle potential name differences)
            original_label_names = all_data.features['label'].names
            label_text = original_label_names[label]

            # Map original names to our project names if needed
            label_mapping = {
                'Bird': 'bird',
                'Cat': 'cat',
                'Llama': 'llama',
                'Rabbit': 'rabbit',
                'deer': 'deer',
                'dog': 'dog',
                'snail': 'snail',
                'swan': 'swan'
            }

            mapped_label_text = label_mapping.get(label_text, label_text.lower())

            # Create filename
            label_slug = mapped_label_text.lower().replace(' ', '_')
            filename = f"{label_slug}_{split_name}_{count:05d}.jpg"

            # Save image
            image_dir = raw_dir / split_name / label_slug
            image_path = image_dir / filename

            # Convert to RGB if necessary and save
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.save(image_path, 'JPEG', quality=95)

            # Add to CSV data
            split_data_list.append({
                'path': str(image_path.relative_to(base_dir)),
                'label': label,
                'label_text': mapped_label_text,
                'split': split_name
            })

            class_counts[split_name][mapped_label_text] += 1

            if (count + 1) % 100 == 0:
                print(f"  Processed {count + 1}/{len(indices)} images...")

        split_data[split_name] = split_data_list
        print(f"  Completed {split_name}: {len(split_data_list)} images")

    # Write CSV files
    for split_name, data in split_data.items():
        csv_path = splits_dir / f"{split_name}.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            if data:  # Check if data exists
                writer = csv.DictWriter(f, fieldnames=['path', 'label', 'label_text', 'split'])
                writer.writeheader()
                writer.writerows(data)
        print(f"Saved {csv_path}")

    # Write labels.json
    labels_data = {
        'classes': config['project']['classes'],
        'num_classes': len(config['project']['classes']),
        'label_to_idx': {label: idx for idx, label in enumerate(config['project']['classes'])},
        'idx_to_label': {idx: label for idx, label in enumerate(config['project']['classes'])}
    }

    labels_path = splits_dir / 'labels.json'
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump(labels_data, f, indent=2)
    print(f"Saved {labels_path}")

    # Print statistics
    print("\n=== Dataset Statistics ===")
    for split_name, counts in class_counts.items():
        print(f"\n{split_name.upper()} split:")
        total = sum(counts.values())
        for class_name in config['project']['classes']:
            count = counts.get(class_name, 0)
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  {class_name:>10}: {count:>4} ({percentage:>5.1f}%)")
        print(f"  {'Total':>10}: {total:>4}")

    return split_data, class_counts


def validate_dataset(config):
    """Perform integrity checks on the organized dataset."""
    print("\n=== Dataset Validation ===")

    base_dir = Path(__file__).parent.parent.parent
    splits_dir = base_dir / config['dataset']['splits_dir']

    # Check CSV files exist
    for split in ['train', 'val']:
        csv_path = splits_dir / f"{split}.csv"
        if not csv_path.exists():
            print(f"❌ Missing CSV file: {csv_path}")
            return False
        print(f"✅ Found CSV file: {csv_path}")

    # Check labels.json exists
    labels_path = splits_dir / 'labels.json'
    if not labels_path.exists():
        print(f"❌ Missing labels file: {labels_path}")
        return False
    print(f"✅ Found labels file: {labels_path}")

    # Validate labels.json content
    with open(labels_path, 'r') as f:
        labels_data = json.load(f)

    expected_classes = set(config['project']['classes'])
    actual_classes = set(labels_data['classes'])

    if expected_classes != actual_classes:
        print(f"❌ Label mismatch. Expected: {expected_classes}, Got: {actual_classes}")
        return False
    print(f"✅ Labels match: {len(expected_classes)} classes")

    # Validate image files exist and can be opened
    validation_errors = []
    sample_count = 0

    for split in ['train', 'val']:
        csv_path = splits_dir / f"{split}.csv"
        df = pd.read_csv(csv_path)

        print(f"\nValidating {split} split ({len(df)} samples)...")

        for idx, row in df.iterrows():
            image_path = base_dir / row['path']

            # Check file exists
            if not image_path.exists():
                validation_errors.append(f"Missing file: {image_path}")
                continue

            # Try to open and verify image
            try:
                with Image.open(image_path) as img:
                    img.verify()
                sample_count += 1
            except Exception as e:
                validation_errors.append(f"Corrupt image {image_path}: {e}")

            # Limit validation to first 50 per split for speed
            if idx >= 49:
                print(f"  Validated first 50 samples...")
                break

    # Check class overlap
    train_df = pd.read_csv(splits_dir / 'train.csv')
    val_df = pd.read_csv(splits_dir / 'val.csv')

    train_classes = set(train_df['label_text'].unique())
    val_classes = set(val_df['label_text'].unique())

    if not val_classes.issubset(train_classes):
        missing_in_train = val_classes - train_classes
        print(f"❌ Validation classes missing in training: {missing_in_train}")
        return False

    print(f"✅ Class overlap validation passed")

    if validation_errors:
        print(f"\n❌ Found {len(validation_errors)} validation errors:")
        for error in validation_errors[:5]:  # Show first 5 errors
            print(f"  {error}")
        if len(validation_errors) > 5:
            print(f"  ... and {len(validation_errors) - 5} more")
        return False

    print(f"✅ All validations passed! Verified {sample_count} images.")
    return True


def main():
    """Main function to run the dataset utility."""
    print("Shadow-Vision Dataset Utility (Utility #1)")
    print("=" * 50)

    # Load configuration
    config = load_config()
    print(f"Loaded config for project: {config['project']['name']}")
    print(f"Target dataset: {config['dataset']['huggingface_repo']}")

    # Download and organize dataset
    try:
        split_data, class_counts = download_and_organize_dataset(config)

        # Validate the organized dataset
        if validate_dataset(config):
            print("\n🎉 Dataset utility completed successfully!")
            print("\nNext steps:")
            print("- Utility #2: Feature Extraction (MediaPipe)")
            print("- Utility #3: Training (k-NN classifier)")
        else:
            print("\n❌ Dataset validation failed!")
            return 1

    except Exception as e:
        print(f"\n❌ Error during dataset processing: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())