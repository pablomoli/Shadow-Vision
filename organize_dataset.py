#!/usr/bin/env python3
"""
Dataset Organization Helper
Easy script to organize Kaggle datasets for pose/gesture classification training
"""

import sys
import argparse
from pathlib import Path

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent / "backend"))

from data.data_loader import LocalDatasetLoader

def main():
    """Main function to organize datasets"""
    parser = argparse.ArgumentParser(
        description='Organize Kaggle dataset for pose/gesture classification training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Organize folder-based dataset
  python organize_dataset.py --source /path/to/kaggle/dataset --format folder

  # Organize CSV-based dataset
  python organize_dataset.py --source /path/to/kaggle --format csv --csv-file dataset.csv

  # Organize with custom target directory
  python organize_dataset.py --source /path/to/kaggle/dataset --target data/my_dataset

  # Create metadata only (don't copy files - for large datasets)
  python organize_dataset.py --source /path/to/kaggle/dataset --no-copy
        """
    )

    parser.add_argument('--source', required=True,
                       help='Source Kaggle dataset directory')
    parser.add_argument('--target', default='data/kaggle_dataset',
                       help='Target organized dataset directory (default: data/kaggle_dataset)')
    parser.add_argument('--format', choices=['folder', 'csv', 'annotations'],
                       default='folder',
                       help='Dataset format (default: folder)')

    # CSV format options
    parser.add_argument('--csv-file',
                       help='CSV file path (required for CSV format)')
    parser.add_argument('--image-column', default='image',
                       help='Image column name in CSV (default: image)')
    parser.add_argument('--label-column', default='label',
                       help='Label column name in CSV (default: label)')
    parser.add_argument('--images-dir',
                       help='Images directory for CSV format (default: same as CSV location)')

    # Annotations format options
    parser.add_argument('--annotations-file',
                       help='Annotations JSON file (required for annotations format)')

    # Options
    parser.add_argument('--no-copy', action='store_true',
                       help='Create metadata only, don\'t copy files (useful for large datasets)')

    args = parser.parse_args()

    print("🚀 Kaggle Dataset Organizer for Pose/Gesture Classification")
    print("="*60)

    # Validate inputs
    if args.format == 'csv' and not args.csv_file:
        print("❌ Error: CSV file required for CSV format")
        return 1

    if args.format == 'annotations' and (not args.annotations_file or not args.images_dir):
        print("❌ Error: Annotations file and images directory required for annotations format")
        return 1

    if not Path(args.source).exists():
        print(f"❌ Error: Source directory does not exist: {args.source}")
        return 1

    # Create loader
    print(f"📁 Source: {args.source}")
    print(f"📁 Target: {args.target}")
    print(f"📋 Format: {args.format}")
    print()

    loader = LocalDatasetLoader(args.target)

    # Load dataset based on format
    print("🔄 Organizing dataset...")
    success = False

    try:
        if args.format == 'folder':
            success = loader.load_from_folder_structure(args.source, copy_files=not args.no_copy)
        elif args.format == 'csv':
            success = loader.load_from_csv(
                args.csv_file,
                args.image_column,
                args.label_column,
                args.images_dir
            )
        elif args.format == 'annotations':
            success = loader.load_from_annotations(args.annotations_file, args.source)

        if success:
            print("✅ Dataset organized successfully!")
            print()
            loader.print_dataset_statistics()

            if loader.validate_dataset():
                print("\n✅ Dataset validation passed!")
                print(f"\n🎯 Your dataset is ready for training!")
                print(f"   Use directory: {args.target}")
                print("\n📚 Next steps:")
                print("   1. Train a model:")
                print(f"      python backend/main.py train --data-dir {args.target}")
                print("   2. Or test the data loading:")
                print(f"      python backend/data/preprocess_data.py")
            else:
                print("\n❌ Dataset validation failed!")
                return 1
        else:
            print("❌ Dataset organization failed!")
            return 1

    except KeyboardInterrupt:
        print("\n⚠️  Organization cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Error during organization: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())