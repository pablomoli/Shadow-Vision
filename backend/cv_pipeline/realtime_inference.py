#!/usr/bin/env python3
"""
Shadow-Vision Real-time Inference Pipeline

Optimized for live demo stability and performance:
- Fast OpenCV-based hand detection
- Real-time feature extraction
- k-NN classification with confidence scoring
- TouchDesigner OSC integration
- Robust error handling for demo environments
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

# OSC communication (optional - install python-osc if needed)
try:
    from pythonosc import udp_client
    OSC_AVAILABLE = True
except ImportError:
    OSC_AVAILABLE = False

# Import our feature extraction
import sys
sys.path.append(str(Path(__file__).parent.parent / "data"))
from feature_extractor import HandDetector, FeatureExtractor, HandFeatures


@dataclass
class InferenceResult:
    """Container for inference results."""
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    processing_time: float
    hand_detected: bool
    error: Optional[str] = None


class RealTimeClassifier:
    """Real-time shadow puppet classifier optimized for live demos."""

    def __init__(self, model_path: str, config_path: str = "config/config.yaml"):
        self.logger = logging.getLogger(__name__)

        # Load configuration
        self.config = self._load_config(config_path)

        # Load trained model
        self.model_data = self._load_model(model_path)
        self.pipeline = self.model_data['pipeline']
        self.classes = self.model_data['classes']
        self.feature_names = self.model_data['feature_names']

        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor()

        # Performance settings for live demo
        self.min_confidence = 0.3  # Minimum confidence threshold
        self.smoothing_window = 5  # Number of frames for prediction smoothing
        self.prediction_history = []

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

        self.logger.info("RealTimeClassifier initialized successfully")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration file."""
        config_file = Path(__file__).parent.parent.parent / config_path
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    def _load_model(self, model_path: str) -> Dict:
        """Load the trained model."""
        model_file = Path(__file__).parent.parent.parent / model_path
        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_file}")

        model_data = joblib.load(model_file)
        self.logger.info(f"Model loaded: {model_file}")
        self.logger.info(f"Classes: {model_data['classes']}")
        return model_data

    def _smooth_predictions(self, prediction: str, confidence: float) -> Tuple[str, float]:
        """Smooth predictions over multiple frames for stability."""
        self.prediction_history.append((prediction, confidence))

        # Keep only recent predictions
        if len(self.prediction_history) > self.smoothing_window:
            self.prediction_history.pop(0)

        # If we have enough history, use consensus
        if len(self.prediction_history) >= 3:
            # Count votes for each class
            votes = {}
            total_confidence = 0

            for pred, conf in self.prediction_history:
                if conf >= self.min_confidence:  # Only count high-confidence predictions
                    votes[pred] = votes.get(pred, 0) + conf
                    total_confidence += conf

            if votes:
                # Return the class with highest confidence sum
                best_class = max(votes.keys(), key=lambda k: votes[k])
                avg_confidence = votes[best_class] / len([p for p, c in self.prediction_history if c >= self.min_confidence])
                return best_class, min(avg_confidence, 1.0)

        return prediction, confidence

    def predict_from_frame(self, frame: np.ndarray) -> InferenceResult:
        """Predict shadow puppet class from camera frame."""
        start_time = time.time()

        try:
            # Extract features from frame
            features = self.feature_extractor.extract_features_from_frame(frame)

            if features is None:
                return InferenceResult(
                    predicted_class="none",
                    confidence=0.0,
                    probabilities={},
                    processing_time=time.time() - start_time,
                    hand_detected=False,
                    error="No hand detected"
                )

            # Convert to feature vector
            feature_vector = features.to_vector().reshape(1, -1)

            # Get prediction and probabilities
            prediction = self.pipeline.predict(feature_vector)[0]
            probabilities = self.pipeline.predict_proba(feature_vector)[0]

            # Convert to class names and probabilities
            class_probs = {
                self.classes[i]: float(prob)
                for i, prob in enumerate(probabilities)
            }

            predicted_class = self.classes[prediction]
            confidence = float(probabilities[prediction])

            # Apply smoothing
            smoothed_class, smoothed_confidence = self._smooth_predictions(predicted_class, confidence)

            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)

            # Keep only recent processing times
            if len(self.processing_times) > 100:
                self.processing_times.pop(0)

            return InferenceResult(
                predicted_class=smoothed_class,
                confidence=smoothed_confidence,
                probabilities=class_probs,
                processing_time=processing_time,
                hand_detected=True
            )

        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return InferenceResult(
                predicted_class="error",
                confidence=0.0,
                probabilities={},
                processing_time=time.time() - start_time,
                hand_detected=False,
                error=str(e)
            )

    def send_to_touchdesigner(self, result: InferenceResult):
        """Send prediction to TouchDesigner via OSC."""
        if self.osc_client is None:
            return

        try:
            # Send main prediction
            self.osc_client.send_message("/shadow_puppet/class", result.predicted_class)
            self.osc_client.send_message("/shadow_puppet/confidence", result.confidence)

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


# Add frame-based feature extraction to FeatureExtractor
def extract_features_from_frame(self, frame: np.ndarray) -> Optional[HandFeatures]:
    """Extract features directly from camera frame."""
    try:
        # Preprocess
        binary = self.detector.preprocess_image(frame)

        # Find hand contour
        contour = self.detector.find_hand_contour(binary)
        if contour is None:
            return None

        # Extract features (same as extract_features but for frame)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        # Convex hull properties
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        convexity_ratio = area / hull_area if hull_area > 0 else 0

        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        bounding_box_area = w * h
        aspect_ratio = w / h if h > 0 else 0
        extent = area / bounding_box_area if bounding_box_area > 0 else 0

        # Solidity
        solidity = area / hull_area if hull_area > 0 else 0

        # Finger counting
        finger_count, defect_count = self.detector.count_fingers(contour)

        # Orientation using fitted ellipse
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            orientation_angle = ellipse[2]
            major_axis = max(ellipse[1])
            minor_axis = min(ellipse[1])
        else:
            orientation_angle = 0
            major_axis = 0
            minor_axis = 0

        # Centroid (normalized to image size)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            centroid_x = M["m10"] / M["m00"] / frame.shape[1]
            centroid_y = M["m01"] / M["m00"] / frame.shape[0]
        else:
            centroid_x = 0.5
            centroid_y = 0.5

        # Hu moments
        hu_moments = cv2.HuMoments(M).flatten()
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

        return HandFeatures(
            area=area,
            perimeter=perimeter,
            convexity_ratio=convexity_ratio,
            solidity=solidity,
            aspect_ratio=aspect_ratio,
            extent=extent,
            bounding_box_area=bounding_box_area,
            finger_count=finger_count,
            convexity_defects=defect_count,
            orientation_angle=orientation_angle,
            major_axis_length=major_axis,
            minor_axis_length=minor_axis,
            centroid_x=centroid_x,
            centroid_y=centroid_y,
            hu_moments=hu_moments.tolist()
        )

    except Exception as e:
        self.logger.error(f"Frame feature extraction failed: {e}")
        return None

# Monkey patch the method
FeatureExtractor.extract_features_from_frame = extract_features_from_frame


class LiveDemoApp:
    """Live demo application with camera feed and real-time classification."""

    def __init__(self, model_path: str = "models/shadow_puppet_classifier.joblib"):
        self.classifier = RealTimeClassifier(model_path)
        self.camera = None
        self.running = False

        # Display settings
        self.display_size = (800, 600)
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

    def draw_results(self, frame: np.ndarray, result: InferenceResult) -> np.ndarray:
        """Draw classification results on frame."""
        # Resize frame for display
        display_frame = cv2.resize(frame, self.display_size)

        # Draw prediction text
        if result.hand_detected and result.confidence > 0.3:
            text = f"Class: {result.predicted_class.upper()}"
            confidence_text = f"Confidence: {result.confidence:.1%}"

            # Background rectangles for text
            cv2.rectangle(display_frame, (10, 10), (400, 80), (0, 0, 0), -1)

            # Draw text
            cv2.putText(display_frame, text, (20, 40), self.font, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, confidence_text, (20, 70), self.font, 0.7, (255, 255, 255), 2)

        elif result.error:
            cv2.putText(display_frame, f"Error: {result.error}", (20, 40), self.font, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(display_frame, "No hand detected", (20, 40), self.font, 0.7, (255, 255, 0), 2)

        # Draw performance stats
        stats = self.classifier.get_performance_stats()
        perf_text = f"FPS: {stats['fps']:.1f} | Processing: {stats['avg_processing_time_ms']:.1f}ms"
        cv2.putText(display_frame, perf_text, (20, display_frame.shape[0] - 20),
                   self.font, 0.5, (255, 255, 255), 1)

        return display_frame

    def run(self):
        """Run the live demo."""
        print("Shadow-Vision Live Demo")
        print("=" * 30)
        print("Press 'q' to quit")
        print("Press 'r' to reset performance stats")
        print("")

        try:
            self.start_camera()
            self.running = True

            while self.running:
                # Capture frame
                ret, frame = self.camera.read()
                if not ret:
                    print("Failed to capture frame")
                    break

                # Flip frame horizontally to match dataset format
                frame = cv2.flip(frame, 1)

                # Classify
                result = self.classifier.predict_from_frame(frame)

                # Send to TouchDesigner
                if result.hand_detected and result.confidence > 0.5:
                    self.classifier.send_to_touchdesigner(result)

                # Update FPS counter
                self.classifier.fps_counter += 1

                # Display
                display_frame = self.draw_results(frame, result)
                cv2.imshow("Shadow-Vision Live Demo", display_frame)

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
    """Main function for live demo."""
    import argparse

    parser = argparse.ArgumentParser(description="Shadow-Vision Real-time Demo")
    parser.add_argument("--model", default="models/shadow_puppet_classifier.joblib",
                       help="Path to trained model")
    parser.add_argument("--camera", type=int, default=0,
                       help="Camera ID")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    try:
        app = LiveDemoApp(args.model)
        app.run()
    except Exception as e:
        print(f"Demo failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())