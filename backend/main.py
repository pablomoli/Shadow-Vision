#!/usr/bin/env python3
"""
Main Backend Entry Point
Unified entry point for all backend functionality
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add backend to Python path
sys.path.append(str(Path(__file__).parent))

from api.websocket_server import GestureWebSocketServer
from data.kaggle_dataset_loader import KaggleDatasetLoader
from train_model import GestureTrainer
from cv_pipeline.camera_handler import test_camera
from cv_pipeline.inference_engine import test_inference_engine
from api.supabase_client import test_supabase_client, setup_database_schema

def run_server(args):
    """Run the WebSocket server"""
    print("Starting Gesture Recognition Server...")
    server = GestureWebSocketServer(args.host, args.port, args.model)
    asyncio.run(server.start_server())

def download_dataset(args):
    """Organize Kaggle dataset for training"""
    print("Organizing Kaggle dataset...")
    loader = KaggleDatasetLoader(args.data_dir)

    # Check if source directory exists
    if not Path(args.source_dir).exists():
        print(f"Source directory not found: {args.source_dir}")
        print("Please provide the path to your Kaggle dataset using --source-dir")
        return

    # Determine dataset format and load
    if args.format == "csv":
        if not args.csv_file:
            print("CSV file required for CSV format. Use --csv-file argument")
            return
        success = loader.load_dataset_from_csv(args.csv_file, args.image_dir)
    else:
        success = loader.load_dataset_from_folders(args.source_dir)

    if success:
        print("Kaggle dataset organized successfully!")
        loader.get_dataset_stats()
        loader.validate_dataset()
    else:
        print("Dataset organization failed")

def train_model(args):
    """Train gesture recognition model"""
    print(f"Training {args.model_type} model...")

    if not Path(args.data_dir).exists():
        print("Dataset not found. Please download dataset first:")
        print("python backend/main.py download-dataset")
        return

    trainer = GestureTrainer(data_dir=args.data_dir)
    best_acc = trainer.train(model_type=args.model_type, num_epochs=args.epochs)

    print(f"Training completed! Best accuracy: {best_acc:.2f}%")

def test_components(args):
    """Test individual components"""
    if args.component == "camera":
        print("Testing camera...")
        test_camera()
    elif args.component == "inference":
        print("Testing inference engine...")
        test_inference_engine()
    elif args.component == "database":
        print("Testing database connection...")
        test_supabase_client()
    elif args.component == "all":
        print("Testing all components...")
        print("\n1. Testing camera...")
        # test_camera()  # Skip interactive test
        print("Camera test skipped (interactive)")

        print("\n2. Testing inference engine...")
        test_inference_engine()

        print("\n3. Testing database...")
        test_supabase_client()

        print("\nAll tests completed!")

def setup_database(args):
    """Setup database schema"""
    print("Setting up database schema...")
    setup_database_schema()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Gesture Recognition Backend')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Server command
    server_parser = subparsers.add_parser('server', help='Run WebSocket server')
    server_parser.add_argument('--host', default='localhost', help='Server host')
    server_parser.add_argument('--port', type=int, default=8000, help='Server port')
    server_parser.add_argument('--model', default='backend/trained_models/efficient_best.pth',
                              help='Path to trained model')

    # Download dataset command (now organizes Kaggle datasets)
    download_parser = subparsers.add_parser('download-dataset', help='Organize Kaggle dataset for training')
    download_parser.add_argument('--source-dir', required=True,
                                help='Source Kaggle dataset directory')
    download_parser.add_argument('--data-dir', default='data/kaggle_dataset',
                                help='Target organized dataset directory')
    download_parser.add_argument('--format', choices=['folders', 'csv'], default='folders',
                                help='Dataset format: folders or csv')
    download_parser.add_argument('--csv-file', help='CSV file path (for CSV format)')
    download_parser.add_argument('--image-dir', help='Image directory (for CSV format)')

    # Train model command
    train_parser = subparsers.add_parser('train', help='Train gesture recognition model')
    train_parser.add_argument('--model-type', default='efficient',
                             choices=['lightweight', 'mobilenet', 'efficient'],
                             help='Model architecture')
    train_parser.add_argument('--epochs', type=int, default=None,
                             help='Number of training epochs')
    train_parser.add_argument('--data-dir', default='data/kaggle_dataset',
                             help='Dataset directory')

    # Test command
    test_parser = subparsers.add_parser('test', help='Test components')
    test_parser.add_argument('component', choices=['camera', 'inference', 'database', 'all'],
                            help='Component to test')

    # Setup database command
    setup_parser = subparsers.add_parser('setup-db', help='Setup database schema')

    args = parser.parse_args()

    if args.command == 'server':
        run_server(args)
    elif args.command == 'download-dataset':
        download_dataset(args)
    elif args.command == 'train':
        train_model(args)
    elif args.command == 'test':
        test_components(args)
    elif args.command == 'setup-db':
        setup_database(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()