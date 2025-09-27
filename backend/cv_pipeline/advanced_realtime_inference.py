#!/usr/bin/env python3
"""
Advanced Real-time Inference for Shadow-Vision

Optimized for real-world demo conditions:
- Uses advanced ensemble model (k-NN + RF + SVM)
- Advanced hand detection for complex backgrounds
- 49 comprehensive features with selection
- SMOTE-balanced training for minority classes
- Real-time performance optimizations
"""

import cv2
import numpy as np
import joblib
import json
import time
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import threading
import queue
import yaml
import sys

# Add path for advanced feature extractor
sys.path.append(str(Path(__file__).parent.parent / "data"))
from advanced_feature_extractor import AdvancedHandDetector, AdvancedFeatureExtractor

# OSC communication (optional)
try:
    from pythonosc import udp_client
    OSC_AVAILABLE = True
except ImportError:
    OSC_AVAILABLE = False


@dataclass
class AdvancedInferenceResult:
    """Container for advanced inference results."""
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    processing_time: float
    hand_detected: bool
    features_extracted: bool
    ensemble_agreement: float  # How much the ensemble models agree
    error: Optional[str] = None


class AdvancedRealTimeClassifier:
    """Advanced real-time shadow puppet classifier."""

    def __init__(self, model_path: str = "models/advanced_shadow_puppet_classifier.joblib",
                 config_path: str = "config/config.yaml"):
        self.logger = logging.getLogger(__name__)

        # Load configuration
        self.config = self._load_config(config_path)

        # Load trained model
        self.model_data = self._load_model(model_path)
        self.pipeline = self.model_data['pipeline']
        self.feature_selector = self.model_data['feature_selector']
        self.classes = self.model_data['classes']

        # Initialize advanced feature extractor
        self.feature_extractor = AdvancedFeatureExtractor()

        # Performance settings for live demo
        self.min_confidence = 0.4  # Minimum confidence threshold
        self.smoothing_window = 7  # Number of frames for prediction smoothing
        self.prediction_history = []

        # Ensemble agreement threshold
        self.min_ensemble_agreement = 0.6

        # OSC client for TouchDesigner communication
        self.osc_client = None
        if OSC_AVAILABLE and 'touchdesigner' in self.config:
            try:
                osc_config = self.config['touchdesigner']['osc']
                self.osc_client = udp_client.SimpleUDPClient(
                    osc_config['host'], osc_config['port']
                )
                self.logger.info(f"OSC client initialized: {osc_config['host']}:{osc_config['port']}")
            except Exception as e:
                self.logger.warning(f"OSC initialization failed: {e}")

        # Performance monitoring
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.processing_times = []

        self.logger.info("AdvancedRealTimeClassifier initialized successfully")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration file."""
        config_file = Path(__file__).parent.parent.parent / config_path
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    def _load_model(self, model_path: str) -> Dict:
        """Load the trained advanced model."""
        model_file = Path(__file__).parent.parent.parent / model_path
        if not model_file.exists():
            raise FileNotFoundError(f"Advanced model not found: {model_file}")

        model_data = joblib.load(model_file)
        self.logger.info(f"Advanced model loaded: {model_file}")
        self.logger.info(f"Model type: {model_data['metrics']['model_type']}")
        self.logger.info(f"Validation accuracy: {model_data['metrics']['accuracy']:.1%}")
        self.logger.info(f"Classes: {model_data['classes']}")
        return model_data

    def _calculate_ensemble_agreement(self, probabilities: np.ndarray) -> float:
        """Calculate how much the ensemble components agree."""
        # Get individual predictions from ensemble components
        # This is a simplified measure based on prediction confidence
        max_prob = np.max(probabilities)
        second_max = np.partition(probabilities, -2)[-2]

        # Agreement is higher when there's a clear winner
        agreement = (max_prob - second_max) / max_prob if max_prob > 0 else 0
        return float(agreement)

    def _smooth_predictions(self, prediction: str, confidence: float,
                           agreement: float) -> Tuple[str, float]:
        """Advanced prediction smoothing with ensemble agreement."""
        self.prediction_history.append((prediction, confidence, agreement))

        # Keep only recent predictions
        if len(self.prediction_history) > self.smoothing_window:
            self.prediction_history.pop(0)

        # If we have enough history, use weighted consensus
        if len(self.prediction_history) >= 3:
            # Weight predictions by confidence and ensemble agreement
            votes = {}
            total_weight = 0

            for pred, conf, agr in self.prediction_history:
                if conf >= self.min_confidence and agr >= self.min_ensemble_agreement:
                    weight = conf * agr  # Combined weight
                    votes[pred] = votes.get(pred, 0) + weight
                    total_weight += weight

            if votes and total_weight > 0:
                # Return the class with highest weighted votes
                best_class = max(votes.keys(), key=lambda k: votes[k])
                avg_confidence = votes[best_class] / total_weight
                return best_class, min(avg_confidence, 1.0)

        return prediction, confidence

    def predict_from_frame(self, frame: np.ndarray) -> AdvancedInferenceResult:
        """Predict shadow puppet class from camera frame using advanced features."""
        start_time = time.time()

        try:
            # Extract advanced features from frame
            features = self.feature_extractor.extract_features_from_frame(frame)

            if features is None:
                return AdvancedInferenceResult(
                    predicted_class="none",
                    confidence=0.0,
                    probabilities={},
                    processing_time=time.time() - start_time,
                    hand_detected=False,
                    features_extracted=False,
                    ensemble_agreement=0.0,
                    error="No hand detected"
                )

            # Convert to feature vector and apply feature selection
            feature_vector = features.to_vector().reshape(1, -1)
            feature_vector_selected = self.feature_selector.transform(feature_vector)

            # Get prediction and probabilities from ensemble
            prediction = self.pipeline.predict(feature_vector_selected)[0]
            probabilities = self.pipeline.predict_proba(feature_vector_selected)[0]

            # Calculate ensemble agreement
            ensemble_agreement = self._calculate_ensemble_agreement(probabilities)

            # Convert to class names and probabilities
            class_probs = {
                self.classes[i]: float(prob)
                for i, prob in enumerate(probabilities)
            }

            predicted_class = self.classes[prediction]
            confidence = float(probabilities[prediction])

            # Apply advanced smoothing
            smoothed_class, smoothed_confidence = self._smooth_predictions(
                predicted_class, confidence, ensemble_agreement
            )

            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)

            # Keep only recent processing times
            if len(self.processing_times) > 100:
                self.processing_times.pop(0)

            return AdvancedInferenceResult(
                predicted_class=smoothed_class,
                confidence=smoothed_confidence,
                probabilities=class_probs,
                processing_time=processing_time,
                hand_detected=True,
                features_extracted=True,
                ensemble_agreement=ensemble_agreement
            )

        except Exception as e:
            self.logger.error(f"Advanced prediction failed: {e}")
            return AdvancedInferenceResult(
                predicted_class="error",
                confidence=0.0,
                probabilities={},
                processing_time=time.time() - start_time,
                hand_detected=False,
                features_extracted=False,
                ensemble_agreement=0.0,
                error=str(e)
            )

    def send_to_touchdesigner(self, result: AdvancedInferenceResult):
        """Send prediction to TouchDesigner via OSC."""
        if self.osc_client is None:
            return

        try:
            # Send main prediction with quality metrics
            self.osc_client.send_message("/shadow_puppet/class", result.predicted_class)
            self.osc_client.send_message("/shadow_puppet/confidence", result.confidence)
            self.osc_client.send_message("/shadow_puppet/agreement", result.ensemble_agreement)
            self.osc_client.send_message("/shadow_puppet/quality",
                                       result.confidence * result.ensemble_agreement)

            # Send all probabilities
            for class_name, prob in result.probabilities.items():
                self.osc_client.send_message(f"/shadow_puppet/prob/{class_name}", prob)

        except Exception as e:
            self.logger.warning(f"OSC send failed: {e}")

    def get_performance_stats(self) -> Dict:
        """Get current performance statistics."""
        current_time = time.time()
        elapsed = current_time - self.fps_start_time

        if elapsed > 0:
            fps = self.fps_counter / elapsed
        else:
            fps = 0

        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0

        return {
            'fps': fps,
            'avg_processing_time_ms': avg_processing_time * 1000,
            'frames_processed': self.fps_counter,
            'elapsed_time': elapsed
        }

    def reset_performance_stats(self):
        """Reset performance counters."""
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.processing_times = []


# Add frame-based feature extraction to AdvancedFeatureExtractor
def extract_features_from_frame(self, frame: np.ndarray):
    """Extract features directly from camera frame."""
    try:
        # Advanced preprocessing
        binary_mask, gray_image = self.detector.preprocess_image(frame)

        # Find hand contour
        contour = self.detector.find_hand_contour(binary_mask, gray_image)
        if contour is None:
            return None

        # Use the same feature extraction as for images
        # (This would normally save the frame temporarily, but for performance
        # we can extract features directly from the contour and ROI)

        # For now, use the image-based method but with temporary file
        temp_path = "temp_frame.jpg"
        cv2.imwrite(temp_path, frame)
        features = self.extract_features(temp_path)

        # Clean up
        try:
            Path(temp_path).unlink()
        except:
            pass

        return features

    except Exception as e:
        self.logger.error(f"Frame feature extraction failed: {e}")
        return None

# Monkey patch the method
AdvancedFeatureExtractor.extract_features_from_frame = extract_features_from_frame


class AdvancedLiveDemoApp:
    """Advanced live demo application with enhanced UI and monitoring."""

    def __init__(self, model_path: str = "models/advanced_shadow_puppet_classifier.joblib"):
        self.classifier = AdvancedRealTimeClassifier(model_path)
        self.camera = None
        self.running = False

        # Display settings
        self.display_size = (1000, 700)
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def start_camera(self, camera_id: int = 0):
        """Initialize camera."""
        self.camera = cv2.VideoCapture(camera_id)
        if not self.camera.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")

        # Set camera properties for better performance
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)

    def draw_advanced_results(self, frame: np.ndarray, result: AdvancedInferenceResult) -> np.ndarray:
        """Draw enhanced classification results on frame."""
        # Resize frame for display
        display_frame = cv2.resize(frame, self.display_size)

        # Background for UI
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (10, 10), (self.display_size[0]-10, 250), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)

        # Title
        cv2.putText(display_frame, "Shadow-Vision Advanced Demo", (20, 40),
                   self.font, 1.2, (0, 255, 255), 2)

        # Model info
        cv2.putText(display_frame, "Model: Ensemble (k-NN + RF + SVM)", (20, 70),
                   self.font, 0.6, (200, 200, 200), 1)

        # Main prediction
        if result.hand_detected and result.confidence > self.classifier.min_confidence:
            # Color based on quality
            quality = result.confidence * result.ensemble_agreement
            if quality > 0.7:
                color = (0, 255, 0)  # Green for high quality
            elif quality > 0.5:
                color = (0, 255, 255)  # Yellow for medium quality
            else:
                color = (0, 165, 255)  # Orange for low quality

            cv2.putText(display_frame, f"Prediction: {result.predicted_class.upper()}", (20, 110),
                       self.font, 1.0, color, 2)

            cv2.putText(display_frame, f"Confidence: {result.confidence:.1%}", (20, 140),
                       self.font, 0.7, (255, 255, 255), 2)

            cv2.putText(display_frame, f"Ensemble Agreement: {result.ensemble_agreement:.1%}", (20, 165),
                       self.font, 0.6, (255, 255, 255), 1)

            cv2.putText(display_frame, f"Quality Score: {quality:.1%}", (20, 185),
                       self.font, 0.6, (255, 255, 255), 1)

        elif result.error:
            cv2.putText(display_frame, f"Error: {result.error}", (20, 110),
                       self.font, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(display_frame, "No hand detected or low confidence", (20, 110),
                       self.font, 0.7, (255, 255, 0), 2)

        # Performance stats
        stats = self.classifier.get_performance_stats()
        perf_text = f"FPS: {stats['fps']:.1f} | Processing: {stats['avg_processing_time_ms']:.1f}ms"
        cv2.putText(display_frame, perf_text, (20, 210),
                   self.font, 0.6, (255, 255, 255), 1)

        # TouchDesigner status
        if result.hand_detected and result.confidence > 0.5:
            cv2.putText(display_frame, f"OSC -> TouchDesigner: {result.predicted_class}",
                       (20, self.display_size[1] - 60), self.font, 0.5, (0, 255, 0), 1)

        # Instructions
        cv2.putText(display_frame, "Press 'q' to quit, 'r' to reset stats",
                   (20, self.display_size[1] - 30), self.font, 0.6, (200, 200, 200), 1)

        # Real-world tips
        cv2.putText(display_frame, "Tips: Good lighting, clear hand silhouette, avoid cluttered background",
                   (20, self.display_size[1] - 10), self.font, 0.5, (150, 150, 150), 1)

        return display_frame

    def run(self):
        """Run the advanced live demo."""
        print("Advanced Shadow-Vision Live Demo")
        print("=" * 40)
        print("Model: Advanced Ensemble (k-NN + RF + SVM)")
        print("Features: 49 advanced features -> 30 selected")
        print("Accuracy: 81.1% on validation set")
        print("Press 'q' to quit, 'r' to reset stats")
        print("")

        try:
            self.start_camera()
            self.running = True

            cv2.namedWindow("Advanced Shadow-Vision Demo", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Advanced Shadow-Vision Demo",
                           self.display_size[0], self.display_size[1])

            while self.running:
                # Capture frame
                ret, frame = self.camera.read()
                if not ret:
                    print("Failed to capture frame")
                    break

                # Classify with advanced model
                result = self.classifier.predict_from_frame(frame)

                # Send to TouchDesigner if high quality
                if (result.hand_detected and
                    result.confidence > 0.6 and
                    result.ensemble_agreement > 0.5):
                    self.classifier.send_to_touchdesigner(result)

                # Update FPS counter
                self.classifier.fps_counter += 1

                # Display
                display_frame = self.draw_advanced_results(frame, result)
                cv2.imshow("Advanced Shadow-Vision Demo", display_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.classifier.reset_performance_stats()
                    print("Performance stats reset")

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        self.running = False
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()

        # Print final stats
        stats = self.classifier.get_performance_stats()
        print(f"\nFinal Performance Stats:")
        print(f"  Frames processed: {stats['frames_processed']}")
        print(f"  Average FPS: {stats['fps']:.1f}")
        print(f"  Average processing time: {stats['avg_processing_time_ms']:.1f}ms")


def main():
    """Main function for advanced live demo."""
    import argparse

    parser = argparse.ArgumentParser(description="Advanced Shadow-Vision Real-time Demo")
    parser.add_argument("--model", default="models/advanced_shadow_puppet_classifier.joblib",
                       help="Path to trained advanced model")
    parser.add_argument("--camera", type=int, default=0,
                       help="Camera ID")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    try:
        app = AdvancedLiveDemoApp(args.model)
        app.run()
    except Exception as e:
        print(f"Advanced demo failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())