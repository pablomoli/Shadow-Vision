#!/usr/bin/env python3
"""
Advanced Training Pipeline for Real-World Shadow Puppet Recognition

Major improvements for live demo accuracy:
1. Advanced ensemble classifier (k-NN + Random Forest + SVM)
2. Class balancing techniques for minority classes
3. Feature selection and engineering
4. Cross-validation with stratification
5. Real-world validation metrics
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

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import (
    cross_val_score, GridSearchCV, StratifiedKFold, cross_validate
)
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, f1_score as sklearn_f1_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


@dataclass
class AdvancedModelMetrics:
    """Container for advanced model evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    cv_mean: float
    cv_std: float
    per_class_f1: Dict[str, float]
    confusion_matrix: List[List[int]]
    classification_report: str
    model_type: str
    feature_count: int
    train_samples: int
    val_samples: int
    class_balance_technique: str
    timestamp: str


class AdvancedShadowPuppetClassifier:
    """Advanced ensemble classifier for shadow puppet recognition."""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.pipeline = None
        self.classes = None
        self.logger = logging.getLogger(__name__)

    def create_ensemble_model(self, n_features: int) -> VotingClassifier:
        """Create an ensemble of complementary classifiers."""
        # k-NN: Good for local decision boundaries
        knn = KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            algorithm='auto',
            metric='euclidean'
        )

        # Random Forest: Good for feature interactions
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'  # Handle class imbalance
        )

        # SVM: Good for complex decision boundaries
        svm = SVC(
            kernel='rbf',
            probability=True,  # Needed for voting
            random_state=42,
            class_weight='balanced'
        )

        # Create voting ensemble
        ensemble = VotingClassifier(
            estimators=[
                ('knn', knn),
                ('rf', rf),
                ('svm', svm)
            ],
            voting='soft'  # Use probabilities
        )

        return ensemble

    def find_optimal_parameters(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """Find optimal hyperparameters using grid search."""
        print("Finding optimal hyperparameters...")

        # Simplified grid search for speed
        param_grid = {
            'knn__n_neighbors': [3, 5, 7],
            'rf__n_estimators': [50, 100],
            'rf__max_depth': [None, 10],
            'svm__C': [0.1, 1, 10]
        }

        # Use stratified k-fold
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        # Grid search
        ensemble = self.create_ensemble_model(X_train.shape[1])
        grid_search = GridSearchCV(
            ensemble,
            param_grid,
            cv=cv,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.3f}")

        return grid_search.best_params_

    def balance_dataset(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Balance the dataset using SMOTE for minority classes."""
        print("Balancing dataset with SMOTE...")

        # Count samples per class
        unique, counts = np.unique(y_train, return_counts=True)
        print("Original class distribution:")
        for class_idx, count in zip(unique, counts):
            print(f"  Class {class_idx}: {count} samples")

        # Apply SMOTE
        smote = SMOTE(random_state=42, k_neighbors=3)
        try:
            X_balanced, y_balanced = smote.fit_resample(X_train, y_train)

            # Show new distribution
            unique, counts = np.unique(y_balanced, return_counts=True)
            print("Balanced class distribution:")
            for class_idx, count in zip(unique, counts):
                print(f"  Class {class_idx}: {count} samples")

            return X_balanced, y_balanced

        except Exception as e:
            print(f"SMOTE failed: {e}. Using original data with class weights.")
            return X_train, y_train

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              classes: List[str]) -> AdvancedModelMetrics:
        """Train the advanced ensemble classifier."""
        print(f"Training advanced ensemble classifier...")
        print(f"Training samples: {len(X_train)}")
        print(f"Features: {X_train.shape[1]}")
        print(f"Classes: {len(classes)}")

        self.classes = classes

        # Feature selection
        print("Selecting best features...")
        n_features_to_select = min(30, X_train.shape[1])  # Select top 30 features
        self.feature_selector = SelectKBest(f_classif, k=n_features_to_select)
        X_train_selected = self.feature_selector.fit_transform(X_train, y_train)

        print(f"Selected {n_features_to_select} features out of {X_train.shape[1]}")

        # Balance dataset
        X_balanced, y_balanced = self.balance_dataset(X_train_selected, y_train)

        # Create pipeline with scaling and ensemble
        ensemble = self.create_ensemble_model(X_balanced.shape[1])

        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ensemble', ensemble)
        ])

        # Train the model
        print("Training ensemble model...")
        self.pipeline.fit(X_balanced, y_balanced)

        # Calculate training accuracy
        train_predictions = self.pipeline.predict(X_train_selected)
        train_accuracy = accuracy_score(y_train, train_predictions)
        print(f"Training accuracy: {train_accuracy:.3f}")

        # Cross-validation for robust evaluation
        print("Performing cross-validation...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        cv_results = cross_validate(
            self.pipeline, X_balanced, y_balanced,
            cv=cv,
            scoring=['accuracy', 'f1_weighted'],
            return_train_score=False
        )

        cv_accuracy = cv_results['test_accuracy']
        cv_f1 = cv_results['test_f1_weighted']

        print(f"Cross-validation accuracy: {cv_accuracy.mean():.3f} ± {cv_accuracy.std():.3f}")
        print(f"Cross-validation F1: {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")

        # Create metrics object
        metrics = AdvancedModelMetrics(
            accuracy=train_accuracy,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            cv_mean=cv_accuracy.mean(),
            cv_std=cv_accuracy.std(),
            per_class_f1={},
            confusion_matrix=[],
            classification_report="",
            model_type="Ensemble (k-NN + RF + SVM)",
            feature_count=n_features_to_select,
            train_samples=len(X_balanced),
            val_samples=0,
            class_balance_technique="SMOTE" if len(X_balanced) > len(X_train) else "Class Weights",
            timestamp=datetime.now().isoformat()
        )

        return metrics

    def evaluate(self, X_val: np.ndarray, y_val: np.ndarray,
                 metrics: AdvancedModelMetrics) -> AdvancedModelMetrics:
        """Evaluate the trained model on validation set."""
        print(f"Evaluating on validation set...")
        print(f"Validation samples: {len(X_val)}")

        # Apply feature selection
        X_val_selected = self.feature_selector.transform(X_val)

        # Predictions
        y_pred = self.pipeline.predict(X_val_selected)
        y_pred_proba = self.pipeline.predict_proba(X_val_selected)

        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='weighted')

        # Per-class F1 scores
        per_class_f1 = sklearn_f1_score(y_val, y_pred, average=None)
        per_class_f1_dict = {
            self.classes[i]: float(per_class_f1[i])
            for i in range(len(self.classes))
        }

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
        metrics.per_class_f1 = per_class_f1_dict
        metrics.confusion_matrix = cm.tolist()
        metrics.classification_report = report
        metrics.val_samples = len(X_val)

        print(f"Validation accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1-score: {f1:.3f}")

        # Show per-class performance
        print("Per-class F1 scores:")
        for class_name, f1_score in per_class_f1_dict.items():
            status = "EXCELLENT" if f1_score > 0.9 else "GOOD" if f1_score > 0.8 else "NEEDS WORK"
            print(f"  {class_name:>8}: {f1_score:.3f} ({status})")

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.pipeline is None or self.feature_selector is None:
            raise ValueError("Model not trained yet!")

        X_selected = self.feature_selector.transform(X)
        return self.pipeline.predict(X_selected)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if self.pipeline is None or self.feature_selector is None:
            raise ValueError("Model not trained yet!")

        X_selected = self.feature_selector.transform(X)
        return self.pipeline.predict_proba(X_selected)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from Random Forest component."""
        if self.pipeline is None:
            raise ValueError("Model not trained yet!")

        # Get Random Forest from ensemble
        rf = self.pipeline.named_steps['ensemble'].named_estimators_['rf']
        feature_importance = rf.feature_importances_

        # Map to selected feature indices
        selected_features = self.feature_selector.get_support(indices=True)

        importance_dict = {}
        for i, importance in enumerate(feature_importance):
            feature_idx = selected_features[i]
            importance_dict[f'feature_{feature_idx}'] = float(importance)

        return importance_dict


def load_config(config_path="config/config.yaml"):
    """Load configuration from YAML file."""
    config_file = Path(__file__).parent.parent.parent / config_path
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def load_advanced_features(splits_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load extracted advanced features from both splits."""
    # Load training features
    train_path = splits_dir / "train_advanced_features.npz"
    if not train_path.exists():
        raise FileNotFoundError(f"Advanced training features not found: {train_path}")

    train_data = np.load(train_path)
    X_train, y_train = train_data['X'], train_data['y']

    # Load validation features
    val_path = splits_dir / "val_advanced_features.npz"
    if not val_path.exists():
        raise FileNotFoundError(f"Advanced validation features not found: {val_path}")

    val_data = np.load(val_path)
    X_val, y_val = val_data['X'], val_data['y']

    print(f"Loaded advanced features:")
    print(f"  Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Validation: {X_val.shape[0]} samples, {X_val.shape[1]} features")

    return X_train, y_train, X_val, y_val


def save_advanced_model_and_metrics(classifier: AdvancedShadowPuppetClassifier,
                                   metrics: AdvancedModelMetrics,
                                   config: Dict, base_dir: Path):
    """Save trained model and evaluation metrics."""
    # Create models directory
    models_dir = base_dir / "models"
    models_dir.mkdir(exist_ok=True)

    # Save model
    model_path = models_dir / "advanced_shadow_puppet_classifier.joblib"
    joblib.dump({
        'pipeline': classifier.pipeline,
        'feature_selector': classifier.feature_selector,
        'classes': classifier.classes,
        'config': config,
        'metrics': asdict(metrics)
    }, model_path)

    print(f"Advanced model saved to: {model_path}")

    # Save metrics as JSON
    metrics_path = models_dir / "advanced_training_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(asdict(metrics), f, indent=2)

    print(f"Advanced metrics saved to: {metrics_path}")

    # Save human-readable report
    report_path = models_dir / "advanced_model_report.txt"
    with open(report_path, 'w') as f:
        f.write("Advanced Shadow-Vision Model Training Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Timestamp: {metrics.timestamp}\n")
        f.write(f"Model Type: {metrics.model_type}\n")
        f.write(f"Training Samples: {metrics.train_samples}\n")
        f.write(f"Validation Samples: {metrics.val_samples}\n")
        f.write(f"Feature Count: {metrics.feature_count}\n")
        f.write(f"Class Balancing: {metrics.class_balance_technique}\n\n")

        f.write("Performance Metrics:\n")
        f.write(f"  Accuracy: {metrics.accuracy:.3f}\n")
        f.write(f"  Precision: {metrics.precision:.3f}\n")
        f.write(f"  Recall: {metrics.recall:.3f}\n")
        f.write(f"  F1-Score: {metrics.f1_score:.3f}\n")
        f.write(f"  CV Mean: {metrics.cv_mean:.3f} ± {metrics.cv_std:.3f}\n\n")

        f.write("Per-Class F1 Scores:\n")
        for class_name, f1_score in metrics.per_class_f1.items():
            f.write(f"  {class_name}: {f1_score:.3f}\n")

        f.write("\nClassification Report:\n")
        f.write(metrics.classification_report)

    print(f"Advanced report saved to: {report_path}")


def main():
    """Main training function."""
    print("Advanced Shadow-Vision Training Pipeline")
    print("=" * 50)
    print("Improvements:")
    print("- Ensemble classifier (k-NN + Random Forest + SVM)")
    print("- SMOTE for class balancing")
    print("- Feature selection")
    print("- Advanced cross-validation")
    print()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    try:
        # Load configuration
        config = load_config()
        print(f"Loaded config for project: {config['project']['name']}")

        base_dir = Path(__file__).parent.parent.parent
        splits_dir = base_dir / config['dataset']['splits_dir']

        # Load advanced features
        X_train, y_train, X_val, y_val = load_advanced_features(splits_dir)

        # Initialize classifier
        classifier = AdvancedShadowPuppetClassifier()

        # Train model
        metrics = classifier.train(X_train, y_train, config['project']['classes'])

        # Evaluate on validation set
        metrics = classifier.evaluate(X_val, y_val, metrics)

        # Feature importance analysis
        print("\nAnalyzing feature importance...")
        try:
            importances = classifier.get_feature_importance()
            sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)

            print("Top 10 most important features:")
            for i, (feature, importance) in enumerate(sorted_features[:10]):
                print(f"  {i+1:2d}. {feature:20s}: {importance:.4f}")
        except Exception as e:
            print(f"Feature importance analysis failed: {e}")

        # Save model and metrics
        save_advanced_model_and_metrics(classifier, metrics, config, base_dir)

        # Print final summary
        print(f"\nAdvanced training completed successfully!")
        print(f"Final Performance:")
        print(f"   Validation Accuracy: {metrics.accuracy:.1%}")
        print(f"   Cross-validation: {metrics.cv_mean:.1%} ± {metrics.cv_std:.1%}")
        print(f"   Model Type: {metrics.model_type}")
        print(f"   Features Used: {metrics.feature_count}")
        print(f"Model saved: models/advanced_shadow_puppet_classifier.joblib")
        print(f"\nReady for real-world demo!")

        return 0

    except Exception as e:
        print(f"Advanced training failed: {e}")
        logging.exception("Advanced training failed")
        return 1


if __name__ == "__main__":
    exit(main())