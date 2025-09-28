#!/usr/bin/env python3
"""
Shadow-Vision Training Pipeline (Utility #3)

Trains k-NN classifier for shadow puppet recognition optimized for live demo stability.

Features:
- k-NN classifier (fast, stable, interpretable)
- Comprehensive evaluation metrics
- Cross-validation for robust performance
- Model persistence and versioning
- Live demo optimization settings
"""

import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import json
import joblib
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import warnings

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

try:
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
    from sklearn.metrics import (
        classification_report, confusion_matrix, accuracy_score,
        precision_recall_fscore_support, roc_auc_score
    )
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("WARNING: scikit-learn not available. Installing...")


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    cv_mean: float
    cv_std: float
    confusion_matrix: List[List[int]]
    classification_report: str
    best_k: int
    feature_count: int
    train_samples: int
    val_samples: int
    timestamp: str


class ShadowPuppetClassifier:
    """k-NN classifier optimized for shadow puppet recognition."""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.pipeline = None
        self.classes = None
        self.feature_names = self._get_feature_names()
        self.logger = logging.getLogger(__name__)

    def _get_feature_names(self) -> List[str]:
        """Define feature names for interpretability."""
        base_features = [
            'area', 'perimeter', 'convexity_ratio', 'solidity',
            'aspect_ratio', 'extent', 'bounding_box_area',
            'finger_count', 'convexity_defects',
            'orientation_angle', 'major_axis_length', 'minor_axis_length',
            'centroid_x', 'centroid_y'
        ]
        hu_features = [f'hu_moment_{i}' for i in range(7)]
        return base_features + hu_features

    def find_optimal_k(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[int, float]:
        """Find optimal k value using cross-validation."""
        print("Finding optimal k value...")

        # Test k values (odd numbers to avoid ties)
        k_range = range(1, min(21, len(X_train) // 2), 2)
        cv_scores = []

        # Use stratified k-fold for better evaluation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for k in k_range:
            # Create pipeline with scaling and k-NN
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('knn', KNeighborsClassifier(n_neighbors=k, weights='distance'))
            ])

            # Cross-validation scores
            scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')
            cv_scores.append(scores.mean())

            print(f"  k={k}: CV accuracy = {scores.mean():.3f} ± {scores.std():.3f}")

        # Find best k
        best_idx = np.argmax(cv_scores)
        best_k = list(k_range)[best_idx]
        best_score = cv_scores[best_idx]

        print(f"Optimal k: {best_k} (CV accuracy: {best_score:.3f})")
        return best_k, best_score

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              classes: List[str]) -> ModelMetrics:
        """Train the k-NN classifier."""
        print(f"Training k-NN classifier...")
        print(f"Training samples: {len(X_train)}")
        print(f"Features: {X_train.shape[1]}")
        print(f"Classes: {len(classes)}")

        self.classes = classes

        # Find optimal k
        best_k, cv_score = self.find_optimal_k(X_train, y_train)

        # Create final pipeline with optimal k
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(
                n_neighbors=best_k,
                weights='distance',  # Weight by distance for better performance
                algorithm='auto',    # Let sklearn choose the best algorithm
                metric='euclidean'   # Standard metric for geometric features
            ))
        ])

        # Train the model
        self.pipeline.fit(X_train, y_train)

        # Calculate training accuracy
        train_accuracy = self.pipeline.score(X_train, y_train)
        print(f"Training accuracy: {train_accuracy:.3f}")

        # Cross-validation for robust evaluation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.pipeline, X_train, y_train, cv=cv, scoring='accuracy')

        print(f"Cross-validation: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        # Create metrics object (will be updated after validation)
        metrics = ModelMetrics(
            accuracy=train_accuracy,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            cv_mean=cv_scores.mean(),
            cv_std=cv_scores.std(),
            confusion_matrix=[],
            classification_report="",
            best_k=best_k,
            feature_count=X_train.shape[1],
            train_samples=len(X_train),
            val_samples=0,
            timestamp=datetime.now().isoformat()
        )

        return metrics

    def evaluate(self, X_val: np.ndarray, y_val: np.ndarray,
                 metrics: ModelMetrics) -> ModelMetrics:
        """Evaluate the trained model on validation set."""
        print(f"\nEvaluating on validation set...")
        print(f"Validation samples: {len(X_val)}")

        # Predictions
        y_pred = self.pipeline.predict(X_val)

        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='weighted')

        # Confusion matrix
        cm = confusion_matrix(y_val, y_pred)

        # Classification report
        report = classification_report(
            y_val, y_pred,
            target_names=self.classes,
            zero_division=0
        )

        # Update metrics
        metrics.accuracy = accuracy
        metrics.precision = precision
        metrics.recall = recall
        metrics.f1_score = f1
        metrics.confusion_matrix = cm.tolist()
        metrics.classification_report = report
        metrics.val_samples = len(X_val)

        print(f"Validation accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1-score: {f1:.3f}")

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.pipeline is None:
            raise ValueError("Model not trained yet!")
        return self.pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if self.pipeline is None:
            raise ValueError("Model not trained yet!")
        return self.pipeline.predict_proba(X)

    def get_feature_importance(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Analyze feature importance using permutation importance."""
        if self.pipeline is None:
            raise ValueError("Model not trained yet!")

        baseline_score = self.pipeline.score(X, y)
        importances = {}

        for i, feature_name in enumerate(self.feature_names):
            # Permute feature
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, i])

            # Calculate score drop
            permuted_score = self.pipeline.score(X_permuted, y)
            importance = baseline_score - permuted_score
            importances[feature_name] = importance

        return importances


def load_config(config_path="config/config.yaml"):
    """Load configuration from YAML file."""
    config_file = Path(__file__).parent.parent.parent / config_path
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def load_features(splits_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load extracted features from both splits."""
    # Load training features
    train_path = splits_dir / "train_features.npz"
    if not train_path.exists():
        raise FileNotFoundError(f"Training features not found: {train_path}")

    train_data = np.load(train_path)
    X_train, y_train = train_data['X'], train_data['y']

    # Load validation features
    val_path = splits_dir / "val_features.npz"
    if not val_path.exists():
        raise FileNotFoundError(f"Validation features not found: {val_path}")

    val_data = np.load(val_path)
    X_val, y_val = val_data['X'], val_data['y']

    print(f"Loaded features:")
    print(f"  Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Validation: {X_val.shape[0]} samples, {X_val.shape[1]} features")

    return X_train, y_train, X_val, y_val


def save_model_and_metrics(classifier: ShadowPuppetClassifier, metrics: ModelMetrics,
                          config: Dict, base_dir: Path):
    """Save trained model and evaluation metrics."""
    # Create models directory
    models_dir = base_dir / "models"
    models_dir.mkdir(exist_ok=True)

    # Save model
    model_path = models_dir / "shadow_puppet_classifier.joblib"
    joblib.dump({
        'pipeline': classifier.pipeline,
        'classes': classifier.classes,
        'feature_names': classifier.feature_names,
        'config': config,
        'metrics': asdict(metrics)
    }, model_path)

    print(f"Model saved to: {model_path}")

    # Save metrics as JSON
    metrics_path = models_dir / "training_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(asdict(metrics), f, indent=2)

    print(f"Metrics saved to: {metrics_path}")

    # Save human-readable report
    report_path = models_dir / "model_report.txt"
    with open(report_path, 'w') as f:
        f.write("Shadow-Vision Model Training Report\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Timestamp: {metrics.timestamp}\n")
        f.write(f"Model Type: k-NN (k={metrics.best_k})\n")
        f.write(f"Training Samples: {metrics.train_samples}\n")
        f.write(f"Validation Samples: {metrics.val_samples}\n")
        f.write(f"Feature Count: {metrics.feature_count}\n\n")

        f.write("Performance Metrics:\n")
        f.write(f"  Accuracy: {metrics.accuracy:.3f}\n")
        f.write(f"  Precision: {metrics.precision:.3f}\n")
        f.write(f"  Recall: {metrics.recall:.3f}\n")
        f.write(f"  F1-Score: {metrics.f1_score:.3f}\n")
        f.write(f"  CV Mean: {metrics.cv_mean:.3f} ± {metrics.cv_std:.3f}\n\n")

        f.write("Classification Report:\n")
        f.write(metrics.classification_report)

    print(f"Report saved to: {report_path}")


def main():
    """Main training function."""
    print("Shadow-Vision Training Pipeline (Utility #3)")
    print("=" * 50)

    # Check sklearn availability
    if not SKLEARN_AVAILABLE:
        print("ERROR: scikit-learn not available!")
        print("Install with: pip install scikit-learn")
        return 1

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    try:
        # Load configuration
        config = load_config()
        print(f"Loaded config for project: {config['project']['name']}")

        base_dir = Path(__file__).parent.parent.parent
        splits_dir = base_dir / config['dataset']['splits_dir']

        # Load features
        X_train, y_train, X_val, y_val = load_features(splits_dir)

        # Initialize classifier
        classifier = ShadowPuppetClassifier()

        # Train model
        metrics = classifier.train(X_train, y_train, config['project']['classes'])

        # Evaluate on validation set
        metrics = classifier.evaluate(X_val, y_val, metrics)

        # Feature importance analysis
        print("\nAnalyzing feature importance...")
        importances = classifier.get_feature_importance(X_val, y_val)
        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)

        print("Top 10 most important features:")
        for i, (feature, importance) in enumerate(sorted_features[:10]):
            print(f"  {i+1:2d}. {feature:20s}: {importance:+.4f}")

        # Save model and metrics
        save_model_and_metrics(classifier, metrics, config, base_dir)

        # Print final summary
        print(f"\n🎉 Training completed successfully!")
        print(f"📊 Final Performance:")
        print(f"   Validation Accuracy: {metrics.accuracy:.1%}")
        print(f"   Cross-validation: {metrics.cv_mean:.1%} ± {metrics.cv_std:.1%}")
        print(f"📁 Model saved: models/shadow_puppet_classifier.joblib")
        print(f"\n🚀 Ready for live demo!")

        return 0

    except Exception as e:
        print(f"❌ Training failed: {e}")
        logging.exception("Training failed")
        return 1


if __name__ == "__main__":
    exit(main())