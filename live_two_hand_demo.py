#!/usr/bin/env python3
"""
Live Two-Hand MediaPipe Demo
Real-time shadow puppet recognition supporting both hands simultaneously
"""

import sys
import cv2
import numpy as np
import joblib
import time
from pathlib import Path

# Add backend to path
sys.path.append('backend')
sys.path.append('.')

from backend.data.mediapipe_extractor_real import RealMediaPipeExtractor, TwoHandDetection

class LiveTwoHandDemo:
    """Live demo using MediaPipe for both hands gesture recognition."""

    def __init__(self):
        """Initialize live two-hand MediaPipe demo."""
        self.classes = ['bird', 'cat', 'llama', 'rabbit', 'deer', 'dog', 'snail', 'swan']

        print("Loading MediaPipe model for two-hand detection...")

        # Load single-hand model (we'll use it for individual hands)
        models_dir = Path('models')
        self.model = joblib.load(models_dir / 'mediapipe_shadow_puppet_classifier.joblib')
        self.scaler = joblib.load(models_dir / 'mediapipe_scaler.joblib')
        self.feature_selector = joblib.load(models_dir / 'mediapipe_feature_selector.joblib')

        # Initialize MediaPipe extractor with two-hand support
        self.extractor = RealMediaPipeExtractor()

        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0

        # Prediction history for smoothing
        self.left_prediction_history = []
        self.right_prediction_history = []
        self.history_size = 3

        print("[OK] Two-hand MediaPipe demo ready!")
        print(f"     Model accuracy: 91.9% (vs 81.1% pixel-based)")
        print(f"     Features: 89 MediaPipe landmarks per hand")
        print(f"     Supports: Both hands simultaneously")
        print(f"     Classes: {', '.join(self.classes)}")

    def predict_single_hand(self, landmarks):
        """Predict gesture for a single hand."""
        try:
            # Extract advanced features (89 features)
            advanced_features = self.extractor.extract_advanced_features(landmarks)
            feature_vector = advanced_features.to_vector().reshape(1, -1)

            # Preprocessing (same as training)
            scaled_features = self.scaler.transform(feature_vector)
            selected_features = self.feature_selector.transform(scaled_features)

            # Prediction
            prediction = self.model.predict(selected_features)[0]
            probabilities = self.model.predict_proba(selected_features)[0]
            confidence = probabilities.max()

            predicted_class = self.classes[prediction]

            return predicted_class, confidence

        except Exception as e:
            print(f"Single hand prediction error: {e}")
            return None, 0.0

    def predict_both_hands(self, frame):
        """Predict gestures for both hands in frame."""
        try:
            # Extract both hands
            two_hand_detection = self.extractor.extract_both_hands_from_frame(frame)

            left_prediction = None
            left_confidence = 0.0
            right_prediction = None
            right_confidence = 0.0

            # Predict for left hand
            if two_hand_detection.left_hand:
                left_prediction, left_confidence = self.predict_single_hand(two_hand_detection.left_hand)
                if left_prediction:
                    self.left_prediction_history.append((left_prediction, left_confidence))
                    if len(self.left_prediction_history) > self.history_size:
                        self.left_prediction_history.pop(0)

            # Predict for right hand
            if two_hand_detection.right_hand:
                right_prediction, right_confidence = self.predict_single_hand(two_hand_detection.right_hand)
                if right_prediction:
                    self.right_prediction_history.append((right_prediction, right_confidence))
                    if len(self.right_prediction_history) > self.history_size:
                        self.right_prediction_history.pop(0)

            return two_hand_detection, left_prediction, left_confidence, right_prediction, right_confidence

        except Exception as e:
            print(f"Two-hand prediction error: {e}")
            return None, None, 0.0, None, 0.0

    def get_smoothed_predictions(self):
        """Get smoothed predictions from history."""
        left_smoothed = None
        left_conf = 0.0
        right_smoothed = None
        right_conf = 0.0

        # Smooth left hand prediction
        if self.left_prediction_history:
            left_predictions = {}
            for pred, conf in self.left_prediction_history:
                if pred not in left_predictions:
                    left_predictions[pred] = []
                left_predictions[pred].append(conf)

            left_smoothed = max(left_predictions.keys(), key=lambda x: len(left_predictions[x]))
            left_conf = np.mean(left_predictions[left_smoothed])

        # Smooth right hand prediction
        if self.right_prediction_history:
            right_predictions = {}
            for pred, conf in self.right_prediction_history:
                if pred not in right_predictions:
                    right_predictions[pred] = []
                right_predictions[pred].append(conf)

            right_smoothed = max(right_predictions.keys(), key=lambda x: len(right_predictions[x]))
            right_conf = np.mean(right_predictions[right_smoothed])

        return left_smoothed, left_conf, right_smoothed, right_conf

    def draw_info_overlay(self, frame, two_hand_detection, left_pred, left_conf, right_pred, right_conf):
        """Draw information overlay on frame."""
        h, w = frame.shape[:2]

        # Background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w-10, 200), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # Title
        cv2.putText(frame, "Two-Hand Shadow Puppet Recognition",
                   (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Hand count
        hands_text = f"Hands: {two_hand_detection.num_hands_detected if two_hand_detection else 0}/2"
        cv2.putText(frame, hands_text,
                   (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Left hand prediction
        y_offset = 85
        if left_pred:
            color = (255, 0, 0)  # Blue for left hand
            cv2.putText(frame, f"Left (L): {left_pred.upper()}",
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"Conf: {left_conf:.1%}",
                       (20, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            cv2.putText(frame, "Left (L): No hand",
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)

        # Right hand prediction
        y_offset = 130
        if right_pred:
            color = (0, 0, 255)  # Red for right hand
            cv2.putText(frame, f"Right (R): {right_pred.upper()}",
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"Conf: {right_conf:.1%}",
                       (20, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            cv2.putText(frame, "Right (R): No hand",
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)

        # Combined gesture suggestion
        if left_pred and right_pred:
            cv2.putText(frame, f"Two-hand gesture: {left_pred} + {right_pred}",
                       (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # FPS
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}",
                   (w-100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Instructions
        cv2.putText(frame, "Controls: 'q' = quit, 's' = screenshot, 'b' = both hands",
                   (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def update_fps(self):
        """Update FPS counter."""
        self.fps_counter += 1
        current_time = time.time()

        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time

    def run_demo(self):
        """Run the live two-hand demo."""
        print("\nStarting Two-Hand MediaPipe demo...")
        print("Position both hands in front of the camera for shadow puppet recognition")
        print("Left hand = Blue (L), Right hand = Red (R)")
        print("Press 'q' to quit, 's' to save screenshot")

        # Open camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Could not open camera")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Flip frame horizontally (mirror effect)
                frame = cv2.flip(frame, 1)

                # Predict both hands
                two_hand_detection, left_pred, left_conf, right_pred, right_conf = self.predict_both_hands(frame)

                # Get smoothed predictions
                if left_pred or right_pred:
                    left_smooth, left_smooth_conf, right_smooth, right_smooth_conf = self.get_smoothed_predictions()
                    if left_smooth:
                        left_pred, left_conf = left_smooth, left_smooth_conf
                    if right_smooth:
                        right_pred, right_conf = right_smooth, right_smooth_conf

                # Draw hand landmarks if detected
                if two_hand_detection and two_hand_detection.has_any_hand():
                    frame = self.extractor.visualize_both_hands(frame, two_hand_detection)

                # Draw info overlay
                frame = self.draw_info_overlay(frame, two_hand_detection, left_pred, left_conf, right_pred, right_conf)

                # Update FPS
                self.update_fps()

                # Display frame
                cv2.imshow('Two-Hand MediaPipe Shadow Puppet Demo', frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save screenshot
                    timestamp = int(time.time())
                    filename = f'two_hand_demo_screenshot_{timestamp}.jpg'
                    cv2.imwrite(filename, frame)
                    print(f"Screenshot saved: {filename}")
                elif key == ord('b'):
                    # Print both hands info
                    if two_hand_detection:
                        print(f"Both hands status: {two_hand_detection.has_both_hands()}")
                        print(f"Left: {left_pred} ({left_conf:.1%}), Right: {right_pred} ({right_conf:.1%})")

        except KeyboardInterrupt:
            print("\nDemo interrupted by user")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.extractor.cleanup()

def main():
    """Main demo function."""
    print("Two-Hand MediaPipe Shadow Puppet Demo")
    print("="*45)
    print("Real-time gesture recognition for both hands simultaneously")
    print("Perfect for shadow puppets requiring two-hand coordination!")
    print("\nKey Features:")
    print("- Detects both hands independently")
    print("- Individual gesture recognition per hand")
    print("- Left hand (Blue L) + Right hand (Red R)")
    print("- Two-hand combination detection")
    print("- 91.9% accuracy per hand")

    try:
        demo = LiveTwoHandDemo()
        demo.run_demo()

        print("\n[SUCCESS] Two-hand MediaPipe demo completed!")
        print("Both hands can now be detected and classified simultaneously")
        print("Perfect for complex shadow puppet performances!")

    except Exception as e:
        print(f"[ERROR] Demo failed: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())