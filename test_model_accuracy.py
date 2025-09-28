#!/usr/bin/env python3
"""
Test script to verify advanced model accuracy on specific problematic gestures.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
import json
from backend.data.advanced_feature_extractor import AdvancedFeatureExtractor
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_on_samples():
    """Test the advanced model on validation samples."""

    # Load the advanced model
    model_path = Path("models/advanced_shadow_puppet_classifier.joblib")
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return

    model = joblib.load(model_path)
    logger.info(f"Advanced model loaded: {model_path}")

    # Load validation metrics
    metrics_path = Path("models/advanced_training_metrics.json")
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    logger.info(f"Model accuracy: {metrics['accuracy']:.1%}")
    logger.info("Per-class F1 scores:")
    for animal, f1 in metrics['per_class_f1'].items():
        logger.info(f"  {animal}: {f1:.3f}")

    # Initialize feature extractor
    extractor = AdvancedFeatureExtractor()

    # Test with validation dataset
    val_dir = Path("data/raw/val")
    if not val_dir.exists():
        logger.error(f"Validation directory not found: {val_dir}")
        return

    # Focus on problematic classes mentioned by user
    problematic_classes = ['snail', 'swan', 'dog', 'llama']

    for class_name in problematic_classes:
        class_dir = val_dir / class_name
        if not class_dir.exists():
            continue

        logger.info(f"\nTesting {class_name} samples:")

        # Test first 5 images from each problematic class
        image_files = list(class_dir.glob("*.jpg"))[:5]

        correct_predictions = 0
        total_predictions = 0

        for img_path in image_files:
            try:
                # Load and process image
                image = cv2.imread(str(img_path))
                if image is None:
                    continue

                # Extract features
                features = extractor.extract_features(image)
                if features is None:
                    logger.warning(f"  Failed to extract features from {img_path.name}")
                    continue

                # Make prediction
                prediction = model.predict([features])[0]
                confidence = max(model.predict_proba([features])[0])

                total_predictions += 1
                if prediction == class_name:
                    correct_predictions += 1
                    status = "✓"
                else:
                    status = "✗"

                logger.info(f"  {status} {img_path.name}: {prediction} (conf: {confidence:.2f})")

            except Exception as e:
                logger.warning(f"  Error processing {img_path.name}: {e}")

        if total_predictions > 0:
            accuracy = correct_predictions / total_predictions
            logger.info(f"  {class_name} accuracy: {correct_predictions}/{total_predictions} ({accuracy:.1%})")

    logger.info("\nAdvanced model test completed!")
    logger.info("Key improvements over basic model:")
    logger.info("- 49 comprehensive features vs 21 basic features")
    logger.info("- Ensemble classifier (k-NN + RF + SVM)")
    logger.info("- SMOTE balancing for minority classes")
    logger.info("- Advanced hand detection for complex backgrounds")

if __name__ == "__main__":
    test_model_on_samples()