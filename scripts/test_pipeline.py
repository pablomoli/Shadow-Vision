#!/usr/bin/env python3
"""
Shadow-Vision Pipeline Testing Suite

Comprehensive testing for live demo readiness:
- End-to-end pipeline validation
- Performance benchmarking
- Accuracy consistency checks
- Real-time stability testing
"""

import sys
import time
import numpy as np
import cv2
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple
import argparse

# Add backend paths
backend_path = Path(__file__).parent.parent / "backend"
sys.path.append(str(backend_path / "data"))
sys.path.append(str(backend_path / "cv_pipeline"))

try:
    from feature_extractor import FeatureExtractor, load_config
    from train_classifier import ShadowPuppetClassifier, load_features
    from realtime_inference import RealTimeClassifier
    import joblib
    import pandas as pd
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the gesture-puppets directory")
    sys.exit(1)


class PipelineTester:
    """Comprehensive pipeline testing for live demo readiness."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.config = load_config(str(base_dir / "config" / "config.yaml"))
        self.logger = logging.getLogger(__name__)

        # Test results
        self.test_results = {}

    def test_dataset_integrity(self) -> bool:
        """Test that dataset was properly prepared."""
        print("🔍 Testing dataset integrity...")

        splits_dir = self.base_dir / self.config['dataset']['splits_dir']

        # Check required files
        required_files = [
            'train.csv', 'val.csv', 'labels.json'
        ]

        for filename in required_files:
            filepath = splits_dir / filename
            if not filepath.exists():
                print(f"❌ Missing file: {filepath}")
                return False

        # Check labels.json
        with open(splits_dir / 'labels.json', 'r') as f:
            labels_data = json.load(f)

        expected_classes = set(self.config['project']['classes'])
        actual_classes = set(labels_data['classes'])

        if expected_classes != actual_classes:
            print(f"❌ Class mismatch: {expected_classes} vs {actual_classes}")
            return False

        # Check CSV files
        for split in ['train', 'val']:
            csv_path = splits_dir / f"{split}.csv"
            df = pd.read_csv(csv_path)

            if len(df) == 0:
                print(f"❌ Empty {split} split")
                return False

            # Check all images exist
            missing_images = 0
            for _, row in df.head(10).iterrows():  # Check first 10
                image_path = self.base_dir / row['path']
                if not image_path.exists():
                    missing_images += 1

            if missing_images > 0:
                print(f"❌ {missing_images} missing images in {split} split")
                return False

        print("✅ Dataset integrity check passed")
        return True

    def test_feature_extraction(self) -> bool:
        """Test feature extraction functionality."""
        print("🔍 Testing feature extraction...")

        splits_dir = self.base_dir / self.config['dataset']['splits_dir']

        # Check feature files exist
        for split in ['train', 'val']:
            feature_path = splits_dir / f"{split}_features.npz"
            if not feature_path.exists():
                print(f"❌ Missing feature file: {feature_path}")
                return False

            # Load and validate features
            data = np.load(feature_path)
            X, y = data['X'], data['y']

            if len(X) == 0:
                print(f"❌ Empty feature file: {feature_path}")
                return False

            # Check feature dimensions
            expected_features = 21  # 14 basic + 7 Hu moments
            if X.shape[1] != expected_features:
                print(f"❌ Wrong feature count: {X.shape[1]} (expected {expected_features})")
                return False

            # Check for invalid values
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                print(f"❌ Invalid values in {split} features")
                return False

        print("✅ Feature extraction check passed")
        return True

    def test_model_loading(self) -> bool:
        """Test that trained model can be loaded and used."""
        print("🔍 Testing model loading...")

        model_path = self.base_dir / "models" / "shadow_puppet_classifier.joblib"
        if not model_path.exists():
            print(f"❌ Model not found: {model_path}")
            return False

        try:
            # Load model
            model_data = joblib.load(model_path)

            required_keys = ['pipeline', 'classes', 'feature_names']
            for key in required_keys:
                if key not in model_data:
                    print(f"❌ Missing key in model: {key}")
                    return False

            # Test prediction
            pipeline = model_data['pipeline']
            n_features = len(model_data['feature_names'])

            # Create dummy feature vector
            dummy_features = np.random.rand(1, n_features)
            prediction = pipeline.predict(dummy_features)
            probabilities = pipeline.predict_proba(dummy_features)

            if len(prediction) != 1:
                print("❌ Model prediction format error")
                return False

            print("✅ Model loading check passed")
            return True

        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return False

    def test_realtime_performance(self) -> bool:
        """Test real-time inference performance."""
        print("🔍 Testing real-time performance...")

        try:
            # Initialize classifier
            classifier = RealTimeClassifier("models/shadow_puppet_classifier.joblib")

            # Create test frames
            test_frames = []
            for _ in range(10):
                # Generate synthetic hand-like image
                frame = np.zeros((480, 640, 3), dtype=np.uint8)

                # Draw a simple hand-like shape
                center = (320, 240)
                cv2.ellipse(frame, center, (80, 120), 0, 0, 360, (255, 255, 255), -1)

                # Add some "fingers"
                for i in range(5):
                    angle = i * 45 - 90
                    x = int(center[0] + 100 * np.cos(np.radians(angle)))
                    y = int(center[1] + 100 * np.sin(np.radians(angle)))
                    cv2.circle(frame, (x, y), 20, (255, 255, 255), -1)

                test_frames.append(frame)

            # Time inference
            processing_times = []
            successful_predictions = 0

            for frame in test_frames:
                start_time = time.time()
                result = classifier.predict_from_frame(frame)
                processing_time = time.time() - start_time

                processing_times.append(processing_time)

                if result.hand_detected:
                    successful_predictions += 1

            # Analyze performance
            avg_time = np.mean(processing_times)
            max_time = np.max(processing_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0

            print(f"  Average processing time: {avg_time*1000:.1f}ms")
            print(f"  Maximum processing time: {max_time*1000:.1f}ms")
            print(f"  Estimated FPS: {fps:.1f}")
            print(f"  Hand detection rate: {successful_predictions}/{len(test_frames)}")

            # Check if performance is acceptable for live demo
            if avg_time > 0.1:  # 100ms is too slow
                print("❌ Performance too slow for live demo")
                return False

            if successful_predictions < len(test_frames) * 0.5:  # At least 50% detection
                print("❌ Hand detection rate too low")
                return False

            self.test_results['performance'] = {
                'avg_processing_time_ms': avg_time * 1000,
                'max_processing_time_ms': max_time * 1000,
                'estimated_fps': fps,
                'detection_rate': successful_predictions / len(test_frames)
            }

            print("✅ Real-time performance check passed")
            return True

        except Exception as e:
            print(f"❌ Performance test failed: {e}")
            return False

    def test_model_accuracy(self) -> bool:
        """Test model accuracy on validation set."""
        print("🔍 Testing model accuracy...")

        try:
            splits_dir = self.base_dir / self.config['dataset']['splits_dir']

            # Load validation features
            val_data = np.load(splits_dir / "val_features.npz")
            X_val, y_val = val_data['X'], val_data['y']

            # Load model
            model_path = self.base_dir / "models" / "shadow_puppet_classifier.joblib"
            model_data = joblib.load(model_path)
            pipeline = model_data['pipeline']

            # Make predictions
            y_pred = pipeline.predict(X_val)
            accuracy = np.mean(y_pred == y_val)

            print(f"  Validation accuracy: {accuracy:.1%}")

            # Per-class accuracy
            classes = self.config['project']['classes']
            for i, class_name in enumerate(classes):
                class_mask = (y_val == i)
                if np.any(class_mask):
                    class_accuracy = np.mean(y_pred[class_mask] == y_val[class_mask])
                    print(f"  {class_name}: {class_accuracy:.1%}")

            # Check if accuracy is acceptable
            if accuracy < 0.7:  # 70% minimum for live demo
                print("❌ Model accuracy too low for live demo")
                return False

            self.test_results['accuracy'] = {
                'overall': accuracy,
                'per_class': {
                    classes[i]: float(np.mean(y_pred[y_val == i] == y_val[y_val == i]))
                    for i in range(len(classes))
                    if np.any(y_val == i)
                }
            }

            print("✅ Model accuracy check passed")
            return True

        except Exception as e:
            print(f"❌ Accuracy test failed: {e}")
            return False

    def test_stability(self) -> bool:
        """Test prediction stability with similar inputs."""
        print("🔍 Testing prediction stability...")

        try:
            classifier = RealTimeClassifier("models/shadow_puppet_classifier.joblib")

            # Create base frame
            base_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            center = (320, 240)
            cv2.ellipse(base_frame, center, (80, 120), 0, 0, 360, (255, 255, 255), -1)

            # Test with small variations
            predictions = []
            for i in range(10):
                # Add small random noise
                frame = base_frame.copy()
                noise = np.random.randint(-10, 11, frame.shape, dtype=np.int16)
                frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

                result = classifier.predict_from_frame(frame)
                if result.hand_detected:
                    predictions.append(result.predicted_class)

            if len(predictions) > 0:
                # Check consistency
                most_common = max(set(predictions), key=predictions.count)
                consistency = predictions.count(most_common) / len(predictions)

                print(f"  Prediction consistency: {consistency:.1%}")

                if consistency < 0.7:  # 70% consistency required
                    print("❌ Predictions not stable enough")
                    return False

                self.test_results['stability'] = {
                    'consistency': consistency,
                    'predictions_made': len(predictions),
                    'total_tests': 10
                }
            else:
                print("❌ No predictions made during stability test")
                return False

            print("✅ Stability check passed")
            return True

        except Exception as e:
            print(f"❌ Stability test failed: {e}")
            return False

    def run_all_tests(self) -> bool:
        """Run all tests and return overall success."""
        print("Shadow-Vision Pipeline Testing Suite")
        print("=" * 40)

        tests = [
            ("Dataset Integrity", self.test_dataset_integrity),
            ("Feature Extraction", self.test_feature_extraction),
            ("Model Loading", self.test_model_loading),
            ("Real-time Performance", self.test_realtime_performance),
            ("Model Accuracy", self.test_model_accuracy),
            ("Prediction Stability", self.test_stability)
        ]

        passed_tests = 0
        total_tests = len(tests)

        for test_name, test_func in tests:
            print(f"\n{test_name}:")
            try:
                if test_func():
                    passed_tests += 1
                else:
                    print(f"❌ {test_name} FAILED")
            except Exception as e:
                print(f"❌ {test_name} ERROR: {e}")

        print(f"\n{'='*40}")
        print(f"Test Results: {passed_tests}/{total_tests} passed")

        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED - Ready for live demo!")
            return True
        else:
            print("❌ Some tests failed - Check issues before demo")
            return False

    def save_test_report(self):
        """Save detailed test report."""
        report_path = self.base_dir / "test_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        print(f"📋 Test report saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Shadow-Vision Pipeline Testing")
    parser.add_argument("--base-dir", default=".",
                       help="Base directory of the project")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during testing

    base_dir = Path(args.base_dir).resolve()

    tester = PipelineTester(base_dir)
    success = tester.run_all_tests()

    if success:
        tester.save_test_report()

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())