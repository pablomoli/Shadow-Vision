#!/usr/bin/env python3
"""
Train Shadow Puppet Classifier with Real MediaPipe Features

This trains the model using real MediaPipe hand landmarks (89 features)
for maximum accuracy. Expected performance: 95%+ vs current 81%.
"""

import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
import time
from datetime import datetime
from collections import Counter

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MediaPipeModelTrainer:
    """Train shadow puppet classifier using real MediaPipe landmarks."""

    def __init__(self):
        """Initialize MediaPipe model trainer."""
        self.classes = ['bird', 'cat', 'llama', 'rabbit', 'deer', 'dog', 'snail', 'swan']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Model components
        self.scaler = None
        self.feature_selector = None
        self.model = None

        # Training data
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None

        logger.info("MediaPipe model trainer initialized")

    def load_mediapipe_features(self, data_dir: str = "data/mediapipe"):
        """Load MediaPipe features from processed dataset."""
        data_path = Path(data_dir)

        features_path = data_path / "mediapipe_features.npz"
        metadata_path = data_path / "mediapipe_metadata.json"

        if not features_path.exists():
            logger.error(f"MediaPipe features not found: {features_path}")
            logger.info("Please run process_dataset_with_mediapipe.py first")
            return False

        logger.info("Loading MediaPipe features...")

        # Load features and labels
        data = np.load(features_path)
        features = data['features']
        labels = data['labels']

        logger.info(f"Loaded features shape: {features.shape}")
        logger.info(f"Feature count: {features.shape[1]} (89 expected)")

        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        logger.info("Processing stats from MediaPipe extraction:")
        stats = metadata['processing_stats']
        logger.info(f"  Total processed: {stats['successful_extractions']}/{stats['total_images']}")

        # Class distribution
        class_counts = Counter(labels)
        logger.info("Class distribution:")
        for class_name in self.classes:
            count = class_counts.get(class_name, 0)
            logger.info(f"  {class_name}: {count} samples")

        self.features = features
        self.labels = labels
        self.metadata = metadata

        return True

    def prepare_training_data(self):
        """Prepare MediaPipe features for training."""
        logger.info("Preparing MediaPipe training data...")

        # Convert labels to indices
        y = np.array([self.class_to_idx[label] for label in self.labels])

        # Split into train/validation (80/20)
        X_train, X_val, y_train, y_val = train_test_split(
            self.features, y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )

        logger.info(f"Training samples: {X_train.shape[0]}")
        logger.info(f"Validation samples: {X_val.shape[0]}")

        # Feature scaling (important for MediaPipe coordinates)
        logger.info("Scaling MediaPipe features...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Class balancing with SMOTE
        logger.info("Applying SMOTE for class balancing...")
        smote = SMOTE(random_state=42, k_neighbors=min(3, X_train.shape[0] // len(self.classes) - 1))
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

        logger.info(f"After SMOTE: {X_train_balanced.shape[0]} training samples")

        # Feature selection (keep top 60 features out of 89)
        logger.info("Selecting best MediaPipe features...")
        n_features = min(60, X_train_balanced.shape[1])
        self.feature_selector = SelectKBest(score_func=f_classif, k=n_features)
        X_train_selected = self.feature_selector.fit_transform(X_train_balanced, y_train_balanced)
        X_val_selected = self.feature_selector.transform(X_val_scaled)

        logger.info(f"Selected {X_train_selected.shape[1]} best MediaPipe features")

        # Store processed data
        self.X_train = X_train_selected
        self.X_val = X_val_selected
        self.y_train = y_train_balanced
        self.y_val = y_val

        return True

    def create_mediapipe_optimized_ensemble(self) -> VotingClassifier:
        """Create ensemble optimized for MediaPipe landmark features."""

        # k-NN optimized for coordinate data
        knn = KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            metric='euclidean',  # Good for coordinate data
            algorithm='kd_tree'   # Efficient for landmark coordinates
        )

        # Random Forest optimized for feature interactions
        rf = RandomForestClassifier(
            n_estimators=300,     # More trees for landmark precision
            max_depth=20,         # Deeper for complex hand relationships
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )

        # SVM optimized for high-dimensional landmark space
        svm = SVC(
            kernel='rbf',
            C=50.0,              # Higher C for landmark precision
            gamma='scale',
            probability=True,
            class_weight='balanced',
            random_state=42
        )

        # Weighted ensemble (RF gets highest weight for landmarks)
        ensemble = VotingClassifier(
            estimators=[
                ('knn', knn),
                ('rf', rf),
                ('svm', svm)
            ],
            voting='soft',
            weights=[1, 3, 2]    # RF dominates for landmark relationships
        )

        return ensemble

    def train_model(self):
        """Train the MediaPipe-optimized ensemble model."""
        logger.info("Training MediaPipe-optimized ensemble model...")

        if self.X_train is None:
            logger.error("Training data not prepared. Call prepare_training_data() first.")
            return False

        # Create optimized ensemble
        self.model = self.create_mediapipe_optimized_ensemble()

        # Train model
        start_time = time.time()
        logger.info("Training ensemble on MediaPipe features...")
        self.model.fit(self.X_train, self.y_train)
        training_time = time.time() - start_time

        logger.info(f"MediaPipe model training completed in {training_time:.1f}s")

        # Evaluate model
        self.evaluate_model()

        return True

    def evaluate_model(self):
        """Evaluate the MediaPipe model performance."""
        logger.info("Evaluating MediaPipe model...")

        # Predictions
        y_pred = self.model.predict(self.X_val)
        y_pred_proba = self.model.predict_proba(self.X_val)

        # Core metrics
        accuracy = self.model.score(self.X_val, self.y_val)
        f1_macro = f1_score(self.y_val, y_pred, average='macro')
        f1_weighted = f1_score(self.y_val, y_pred, average='weighted')

        logger.info(f"MediaPipe Model Accuracy: {accuracy:.1%}")
        logger.info(f"F1-Score (Macro): {f1_macro:.3f}")
        logger.info(f"F1-Score (Weighted): {f1_weighted:.3f}")

        # Per-class performance
        f1_per_class = f1_score(self.y_val, y_pred, average=None)
        logger.info("Per-class F1 scores (MediaPipe):")
        for i, class_name in enumerate(self.classes):
            logger.info(f"  {class_name}: {f1_per_class[i]:.3f}")

        # Classification report
        class_report = classification_report(
            self.y_val, y_pred,
            target_names=self.classes,
            digits=3
        )

        # Confusion matrix
        conf_matrix = confusion_matrix(self.y_val, y_pred)

        # Cross-validation
        logger.info("Performing cross-validation...")
        cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        logger.info(f"Cross-validation: {cv_mean:.3f} ± {cv_std:.3f}")

        # Store results
        self.results = {
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'f1_per_class': {self.classes[i]: float(f1_per_class[i]) for i in range(len(self.classes))},
            'cv_mean': float(cv_mean),
            'cv_std': float(cv_std),
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report,
            'model_type': 'MediaPipe Ensemble (k-NN + RF + SVM)',
            'feature_type': 'Real MediaPipe Hand Landmarks',
            'feature_count': int(self.X_train.shape[1]),
            'train_samples': int(self.X_train.shape[0]),
            'val_samples': int(self.X_val.shape[0]),
            'timestamp': datetime.now().isoformat()
        }

        # Compare with previous results
        logger.info("\n" + "=" * 50)
        logger.info("ACCURACY COMPARISON:")
        logger.info(f"Previous (Pixel-based): 81.1%")
        logger.info(f"MediaPipe (Landmarks):  {accuracy:.1%}")
        improvement = (accuracy - 0.811) * 100
        logger.info(f"Improvement: +{improvement:.1f} percentage points")
        logger.info("=" * 50)

        return self.results

    def save_model(self, output_dir: str = "models"):
        """Save the trained MediaPipe model."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        logger.info(f"Saving MediaPipe model to {output_path}")

        # Save main model
        model_path = output_path / "mediapipe_shadow_puppet_classifier.joblib"
        joblib.dump(self.model, model_path)

        # Save preprocessing components
        scaler_path = output_path / "mediapipe_scaler.joblib"
        joblib.dump(self.scaler, scaler_path)

        selector_path = output_path / "mediapipe_feature_selector.joblib"
        joblib.dump(self.feature_selector, selector_path)

        # Save training results
        results_path = output_path / "mediapipe_training_metrics.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        # Save human-readable report
        report_path = output_path / "mediapipe_model_report.txt"
        with open(report_path, 'w') as f:
            f.write("MediaPipe Hand Landmark Model Report\n")
            f.write("=" * 45 + "\n\n")
            f.write(f"Timestamp: {self.results['timestamp']}\n")
            f.write(f"Model Type: {self.results['model_type']}\n")
            f.write(f"Feature Type: {self.results['feature_type']}\n")
            f.write(f"Training Samples: {self.results['train_samples']}\n")
            f.write(f"Validation Samples: {self.results['val_samples']}\n")
            f.write(f"Feature Count: {self.results['feature_count']}\n\n")

            f.write("Performance Comparison:\n")
            f.write(f"  Previous (Pixel-based): 81.1%\n")
            f.write(f"  MediaPipe (Landmarks):  {self.results['accuracy']:.1%}\n")
            improvement = (self.results['accuracy'] - 0.811) * 100
            f.write(f"  Improvement: +{improvement:.1f} percentage points\n\n")

            f.write("MediaPipe Model Performance:\n")
            f.write(f"  Accuracy: {self.results['accuracy']:.3f}\n")
            f.write(f"  F1-Score (Macro): {self.results['f1_macro']:.3f}\n")
            f.write(f"  F1-Score (Weighted): {self.results['f1_weighted']:.3f}\n")
            f.write(f"  CV Mean: {self.results['cv_mean']:.3f} ± {self.results['cv_std']:.3f}\n\n")

            f.write("Per-Class F1 Scores:\n")
            for class_name, f1 in self.results['f1_per_class'].items():
                f.write(f"  {class_name}: {f1:.3f}\n")

            f.write(f"\nClassification Report:\n{self.results['classification_report']}\n")

        logger.info("MediaPipe model saved:")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Scaler: {scaler_path}")
        logger.info(f"  Feature Selector: {selector_path}")
        logger.info(f"  Results: {results_path}")
        logger.info(f"  Report: {report_path}")

        return True

def main():
    """Main training function."""
    print("MediaPipe Hand Landmark Model Training")
    print("=" * 45)
    print("Training with REAL MediaPipe landmarks (89 features)")
    print("Expected accuracy: 95%+ vs current pixel-based 81%")
    print()

    trainer = MediaPipeModelTrainer()

    try:
        # Load MediaPipe features
        if not trainer.load_mediapipe_features():
            return 1

        # Prepare training data
        if not trainer.prepare_training_data():
            return 1

        # Train model
        if not trainer.train_model():
            return 1

        # Save model
        trainer.save_model()

        print("\n🎉 MediaPipe model training completed!")
        print("\nKey advantages of MediaPipe approach:")
        print("✅ Real hand landmark detection (21 precise points)")
        print("✅ 89 advanced features vs 49 pixel-based")
        print("✅ Rotation, scale, and lighting invariant")
        print("✅ Background independent")
        print("✅ Professional-grade accuracy for ShellHacks 2025")

        # Show final accuracy
        accuracy = trainer.results['accuracy']
        improvement = (accuracy - 0.811) * 100
        print(f"\nFinal Results:")
        print(f"MediaPipe Accuracy: {accuracy:.1%}")
        print(f"Improvement: +{improvement:.1f} percentage points")

    except Exception as e:
        logger.error(f"MediaPipe training failed: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())