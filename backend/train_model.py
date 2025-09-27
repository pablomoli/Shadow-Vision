#!/usr/bin/env python3
"""
Training Script for Gesture Recognition Model
Trains CNN model on HaSPeR dataset with experiment tracking
"""

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from pathlib import Path
import time
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.gesture_classifier import load_model_from_config, count_parameters
from data.preprocess_data import create_data_loaders
from api.supabase_client import SupabaseLogger

class GestureTrainer:
    """Training manager for gesture recognition models"""

    def __init__(self, config_path="config/model_config.yaml", data_dir="data/kaggle_dataset"):
        self.config_path = config_path
        self.data_dir = data_dir

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize logging
        self.logger = SupabaseLogger()

        # Training metrics
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

    def setup_model_and_data(self, model_type="efficient"):
        """Setup model, data loaders, and training components"""

        # Create model
        self.model = load_model_from_config(self.config_path, model_type)
        self.model = self.model.to(self.device)

        print(f"Model: {model_type}")
        print(f"Parameters: {count_parameters(self.model):,}")

        # Create data loaders
        train_config = self.config['training']
        self.train_loader, self.val_loader, self.idx_to_gesture = create_data_loaders(
            self.data_dir,
            batch_size=train_config['batch_size'],
            num_workers=4
        )

        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")

        # Setup loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()

        optimizer_name = train_config['optimizer']
        if optimizer_name == "Adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=train_config['learning_rate']
            )
        elif optimizer_name == "SGD":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=train_config['learning_rate'],
                momentum=0.9,
                weight_decay=1e-4
            )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=3,
            verbose=True
        )

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]")

            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                pbar.set_postfix({
                    'Loss': f'{running_loss/(batch_idx+1):.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def train(self, model_type="efficient", num_epochs=None):
        """Main training loop"""

        if num_epochs is None:
            num_epochs = self.config['training']['epochs']

        # Setup model and data
        self.setup_model_and_data(model_type)

        # Early stopping
        early_stopping_config = self.config['training'].get('early_stopping', {})
        best_val_acc = 0
        patience = early_stopping_config.get('patience', 5)
        patience_counter = 0

        print(f"\nStarting training for {num_epochs} epochs...")
        start_time = time.time()

        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)

            # Validate
            val_loss, val_acc = self.validate_epoch(epoch)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            # Learning rate scheduling
            self.scheduler.step(val_acc)

            print(f"\nEpoch {epoch+1}/{num_epochs}:")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Log to database (now uses mock logger)
            self.logger.log_training_metrics(
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_acc,
                val_loss=val_loss,
                val_accuracy=val_acc,
                model_version=model_type
            )

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self.save_model(model_type, epoch, val_acc)
                print(f"New best model saved! Val Acc: {val_acc:.2f}%")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")

        # Plot training curves
        self.plot_training_curves()

        return best_val_acc

    def save_model(self, model_type, epoch, val_acc):
        """Save model checkpoint"""
        save_dir = Path("backend/trained_models")
        save_dir.mkdir(exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'model_type': model_type,
            'config': self.config,
            'idx_to_gesture': self.idx_to_gesture
        }

        # Save best model
        best_path = save_dir / f"{model_type}_best.pth"
        torch.save(checkpoint, best_path)

        # Save latest model
        latest_path = save_dir / f"{model_type}_latest.pth"
        torch.save(checkpoint, latest_path)

        print(f"Model saved to {best_path}")

    def plot_training_curves(self):
        """Plot training and validation curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Loss curves
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()

        # Accuracy curves
        ax2.plot(self.train_accuracies, label='Train Acc')
        ax2.plot(self.val_accuracies, label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()

        plt.tight_layout()
        plt.savefig('training_curves.png')
        print("Training curves saved as 'training_curves.png'")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train gesture recognition model')
    parser.add_argument('--model', type=str, default='efficient',
                       choices=['lightweight', 'mobilenet', 'efficient'],
                       help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--data-dir', type=str, default='data/kaggle_dataset',
                       help='Path to dataset directory')

    args = parser.parse_args()

    # Check if dataset exists
    if not Path(args.data_dir).exists():
        print("Dataset not found. Please organize your Kaggle dataset first:")
        print("python backend/main.py download-dataset --source-dir /path/to/your/kaggle/dataset")
        return

    # Create trainer and start training
    trainer = GestureTrainer(data_dir=args.data_dir)
    best_acc = trainer.train(model_type=args.model, num_epochs=args.epochs)

    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Model saved in backend/trained_models/")

if __name__ == "__main__":
    main()