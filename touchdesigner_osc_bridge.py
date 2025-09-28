#!/usr/bin/env python3
"""
TouchDesigner OSC Bridge for Shadow-Vision

Sends gesture recognition results to TouchDesigner via OSC.
"""

import cv2
import numpy as np
import logging
import json
import time
from pathlib import Path
from backend.data.advanced_feature_extractor import AdvancedFeatureExtractor
import joblib
from pythonosc import udp_client
from pythonosc.osc_message_builder import OscMessageBuilder

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TouchDesignerBridge:
    """Bridge between Shadow-Vision ML and TouchDesigner."""

    def __init__(self, td_ip="127.0.0.1", td_port=7000, send_port=7001):
        """Initialize TouchDesigner bridge."""
        self.td_ip = td_ip
        self.td_port = td_port
        self.send_port = send_port

        # OSC client for sending to TouchDesigner
        self.osc_client = udp_client.SimpleUDPClient(td_ip, td_port)

        # Load model
        self.model = None
        self.extractor = None
        self.classes = ['bird', 'cat', 'llama', 'rabbit', 'deer', 'dog', 'snail', 'swan']
        self.load_model()

        # State tracking
        self.last_prediction = None
        self.prediction_confidence = 0.0
        self.prediction_stability = 0
        self.frame_count = 0

        logger.info(f"TouchDesigner bridge initialized - sending to {td_ip}:{td_port}")

    def load_model(self):
        """Load the advanced ML model."""
        model_path = Path("models/advanced_shadow_puppet_classifier.joblib")
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return False

        try:
            self.model = joblib.load(model_path)
            self.extractor = AdvancedFeatureExtractor()
            logger.info("Advanced model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def send_gesture_data(self, gesture, confidence, frame_data=None):
        """Send gesture data to TouchDesigner via OSC."""
        try:
            # Main gesture message
            self.osc_client.send_message("/gesture/name", gesture)
            self.osc_client.send_message("/gesture/confidence", float(confidence))
            self.osc_client.send_message("/gesture/stability", int(self.prediction_stability))
            self.osc_client.send_message("/gesture/frame_count", int(self.frame_count))

            # Individual gesture triggers (for TouchDesigner logic)
            for i, class_name in enumerate(self.classes):
                is_active = 1.0 if class_name == gesture else 0.0
                self.osc_client.send_message(f"/gesture/{class_name}", is_active)

            # Additional data
            if frame_data:
                self.osc_client.send_message("/camera/width", frame_data.get('width', 640))
                self.osc_client.send_message("/camera/height", frame_data.get('height', 480))

            # Performance metrics
            self.osc_client.send_message("/system/fps", frame_data.get('fps', 0.0) if frame_data else 0.0)

            logger.debug(f"Sent to TD: {gesture} ({confidence:.2f})")

        except Exception as e:
            logger.error(f"OSC send error: {e}")

    def send_hand_position(self, contour, frame_shape):
        """Send hand position and size data to TouchDesigner."""
        try:
            if contour is not None and len(contour) > 0:
                # Calculate hand center
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # Normalize to 0-1 range for TouchDesigner
                    norm_x = cx / frame_shape[1]
                    norm_y = cy / frame_shape[0]

                    # Calculate hand size
                    area = cv2.contourArea(contour)
                    norm_area = area / (frame_shape[0] * frame_shape[1])

                    # Send position data
                    self.osc_client.send_message("/hand/x", float(norm_x))
                    self.osc_client.send_message("/hand/y", float(norm_y))
                    self.osc_client.send_message("/hand/size", float(norm_area))
                    self.osc_client.send_message("/hand/detected", 1.0)
                else:
                    self.osc_client.send_message("/hand/detected", 0.0)
            else:
                self.osc_client.send_message("/hand/detected", 0.0)

        except Exception as e:
            logger.error(f"Hand position send error: {e}")

    def run_touchdesigner_bridge(self):
        """Main loop for TouchDesigner integration."""
        if not self.model:
            logger.error("Model not loaded")
            return False

        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Cannot open camera")
            return False

        logger.info("Starting TouchDesigner bridge...")
        logger.info("Send OSC messages to TouchDesigner at /gesture/name, /gesture/confidence, etc.")

        # Performance tracking
        fps_start_time = time.time()
        fps_frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Flip frame horizontally to match dataset format
                frame = cv2.flip(frame, 1)

                self.frame_count += 1

                # Calculate FPS
                fps_frame_count += 1
                if fps_frame_count >= 30:
                    current_time = time.time()
                    fps = fps_frame_count / (current_time - fps_start_time)
                    fps_start_time = current_time
                    fps_frame_count = 0
                else:
                    fps = 0.0

                # Extract features and predict
                try:
                    features = self.extractor.extract_features(frame)
                    if features is not None:
                        # Get prediction
                        prediction = self.model.predict([features])[0]
                        probabilities = self.model.predict_proba([features])[0]
                        confidence = max(probabilities)

                        # Stability tracking (how many frames same prediction)
                        if prediction == self.last_prediction:
                            self.prediction_stability += 1
                        else:
                            self.prediction_stability = 1

                        self.last_prediction = prediction
                        self.prediction_confidence = confidence

                        # Send to TouchDesigner
                        frame_data = {
                            'width': frame.shape[1],
                            'height': frame.shape[0],
                            'fps': fps
                        }

                        self.send_gesture_data(prediction, confidence, frame_data)

                        # Send hand position if we have contour
                        contour = self.extractor.detector.detect_hand_contour(frame)
                        self.send_hand_position(contour, frame.shape)

                        # Display for debugging
                        cv2.putText(frame, f"{prediction}: {confidence:.2f}",
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, f"Stability: {self.prediction_stability}",
                                  (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        cv2.putText(frame, f"FPS: {fps:.1f}",
                                  (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    else:
                        # No hand detected
                        self.send_gesture_data("none", 0.0, {'width': frame.shape[1], 'height': frame.shape[0], 'fps': fps})
                        self.send_hand_position(None, frame.shape)

                except Exception as e:
                    logger.warning(f"Frame processing error: {e}")

                # Show frame (optional for debugging)
                cv2.imshow('Shadow-Vision → TouchDesigner', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        except KeyboardInterrupt:
            logger.info("Stopped by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()

        return True

def main():
    """Main function with configuration options."""
    import argparse

    parser = argparse.ArgumentParser(description='TouchDesigner Bridge for Shadow-Vision')
    parser.add_argument('--td-ip', default='127.0.0.1', help='TouchDesigner IP address')
    parser.add_argument('--td-port', type=int, default=7000, help='TouchDesigner OSC port')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    print("Shadow-Vision → TouchDesigner Bridge")
    print("="*50)
    print(f"Sending OSC to: {args.td_ip}:{args.td_port}")
    print("OSC Messages:")
    print("  /gesture/name      - Current gesture (string)")
    print("  /gesture/confidence - Confidence 0-1 (float)")
    print("  /gesture/stability  - Frames stable (int)")
    print("  /gesture/bird       - Bird active 0/1 (float)")
    print("  /gesture/cat        - Cat active 0/1 (float)")
    print("  /hand/x            - Hand X position 0-1 (float)")
    print("  /hand/y            - Hand Y position 0-1 (float)")
    print("  /hand/size         - Hand size 0-1 (float)")
    print("="*50)

    bridge = TouchDesignerBridge(td_ip=args.td_ip, td_port=args.td_port)
    success = bridge.run_touchdesigner_bridge()

    return 0 if success else 1

if __name__ == "__main__":
    exit(main())