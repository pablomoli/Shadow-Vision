#!/usr/bin/env python3
"""
Enhanced MediaPipe TouchDesigner Bridge with Full Landmark Streaming
Provides both gesture recognition AND raw landmark data for TouchDesigner.
Optimized for live demo performance with minimal latency.
"""

import cv2
import numpy as np
import logging
import json
import time
import joblib
from pathlib import Path
from pythonosc import udp_client
from pythonosc.osc_message_builder import OscMessageBuilder
import sys

# Add backend to path
sys.path.append('backend')
sys.path.append('.')

from backend.data.mediapipe_extractor_real import RealMediaPipeExtractor, TwoHandDetection

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedTouchDesignerBridge:
    """Enhanced MediaPipe TouchDesigner bridge with full landmark streaming."""

    def __init__(self, td_ip="127.0.0.1", td_port=7000,
                 confidence_threshold=0.7, stability_duration=1.0,
                 stream_landmarks=True, stream_video=False,
                 label_port=7010):
        """Initialize enhanced TouchDesigner bridge."""
        self.td_ip = td_ip
        self.td_port = td_port
        self.stream_landmarks = stream_landmarks
        self.stream_video = stream_video
        self.label_port = label_port

        # OSC client for sending to TouchDesigner (full data)
        self.osc_client = udp_client.SimpleUDPClient(td_ip, td_port)
        # Secondary OSC client dedicated to sending only the confirmed label
        try:
            self.label_client = udp_client.SimpleUDPClient(td_ip, label_port)
        except Exception as e:
            logger.error(f"Failed to create secondary OSC client on {td_ip}:{label_port}: {e}")
            self.label_client = None

        # Load MediaPipe model
        self.classes = ['bird', 'cat', 'llama', 'rabbit', 'deer', 'dog', 'snail', 'swan']
        self.load_mediapipe_model()

        # Initialize stable output buffer
        from mediapipe_touchdesigner_bridge import TwoHandStableOutputBuffer
        self.output_buffer = TwoHandStableOutputBuffer(confidence_threshold, stability_duration)

        # Performance tracking
        self.frame_count = 0
        self.predictions_sent = 0
        self.landmarks_sent = 0
        self.last_fps_time = time.time()
        self.fps = 0.0

        # TouchDesigner landmark format options
        self.landmark_format = "individual"  # "individual", "array", "both"

        # Init logging summary
        logger.info(f"Enhanced MediaPipe TouchDesigner bridge ready - {self.td_ip}:{self.td_port}")
        logger.info(f"Selected label port: {self.td_ip}:{self.label_port}")
        logger.info(f"Landmark streaming: {self.stream_landmarks}, Video streaming: {self.stream_video}")
        logger.info(f"Stable output: {confidence_threshold:.0%} confidence, {stability_duration:.1f}s duration")

    def load_mediapipe_model(self):
        """Load the MediaPipe ML model."""
        try:
            models_dir = Path('models')
            self.model = joblib.load(models_dir / 'mediapipe_shadow_puppet_classifier.joblib')
            self.scaler = joblib.load(models_dir / 'mediapipe_scaler.joblib')
            self.feature_selector = joblib.load(models_dir / 'mediapipe_feature_selector.joblib')

            # Initialize MediaPipe extractor
            self.extractor = RealMediaPipeExtractor()

            logger.info("MediaPipe model loaded successfully")
            logger.info(f"Classes: {', '.join(self.classes)}")

        except Exception as e:
            logger.error(f"Failed to load MediaPipe model: {e}")
            raise

    def predict_single_hand(self, landmarks):
        """Predict gesture for a single hand."""
        try:
            if landmarks is None:
                return None, 0.0

            # Extract advanced features
            advanced_features = self.extractor.extract_advanced_features(landmarks)
            feature_vector = advanced_features.to_vector().reshape(1, -1)

            # Preprocessing
            scaled_features = self.scaler.transform(feature_vector)
            selected_features = self.feature_selector.transform(scaled_features)

            # Prediction
            prediction = self.model.predict(selected_features)[0]
            probabilities = self.model.predict_proba(selected_features)[0]
            confidence = probabilities.max()

            predicted_class = self.classes[prediction]
            return predicted_class, confidence

        except Exception as e:
            logger.error(f"Single hand prediction error: {e}")
            return None, 0.0

    def predict_both_hands(self, frame):
        """Predict gestures for both hands."""
        try:
            # Extract both hands
            two_hand_detection = self.extractor.extract_both_hands_from_frame(frame)

            left_animal = None
            left_confidence = 0.0
            right_animal = None
            right_confidence = 0.0

            # Predict for left hand
            if two_hand_detection.left_hand:
                left_animal, left_confidence = self.predict_single_hand(two_hand_detection.left_hand)

            # Predict for right hand
            if two_hand_detection.right_hand:
                right_animal, right_confidence = self.predict_single_hand(two_hand_detection.right_hand)

            return two_hand_detection, left_animal, left_confidence, right_animal, right_confidence

        except Exception as e:
            logger.error(f"Two-hand prediction error: {e}")
            return None, None, 0.0, None, 0.0

    def send_landmark_data_to_touchdesigner(self, two_hand_detection):
        """Send raw landmark data to TouchDesigner in optimized format."""
        if not self.stream_landmarks or not two_hand_detection:
            return

        try:
            # Send left hand landmarks
            if two_hand_detection.left_hand:
                self.send_hand_landmarks("left", two_hand_detection.left_hand)

            # Send right hand landmarks
            if two_hand_detection.right_hand:
                self.send_hand_landmarks("right", two_hand_detection.right_hand)

            # Send hand detection status
            self.osc_client.send_message("/landmarks/left/detected",
                                       1.0 if two_hand_detection.left_hand else 0.0)
            self.osc_client.send_message("/landmarks/right/detected",
                                       1.0 if two_hand_detection.right_hand else 0.0)

            self.landmarks_sent += 1

        except Exception as e:
            logger.error(f"Failed to send landmark data: {e}")

    def send_hand_landmarks(self, hand_side, landmarks):
        """Send individual hand landmarks in TouchDesigner-optimized format."""
        try:
            # Convert landmarks to list format
            landmark_array = landmarks.to_vector()  # 63 values (21 landmarks × 3 coords)

            if self.landmark_format in ["individual", "both"]:
                # Send individual landmark coordinates (preferred for TouchDesigner arrays)
                for i in range(21):
                    idx = i * 3
                    x, y, z = landmark_array[idx], landmark_array[idx + 1], landmark_array[idx + 2]

                    self.osc_client.send_message(f"/landmarks/{hand_side}/{i}/x", float(x))
                    self.osc_client.send_message(f"/landmarks/{hand_side}/{i}/y", float(y))
                    self.osc_client.send_message(f"/landmarks/{hand_side}/{i}/z", float(z))

                # Send landmark names for TouchDesigner reference
                landmark_names = [
                    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
                    "index_mcp", "index_pip", "index_dip", "index_tip",
                    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
                    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
                    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
                ]

                for i, name in enumerate(landmark_names):
                    idx = i * 3
                    self.osc_client.send_message(f"/landmarks/{hand_side}/{name}/x", float(landmark_array[idx]))
                    self.osc_client.send_message(f"/landmarks/{hand_side}/{name}/y", float(landmark_array[idx + 1]))
                    self.osc_client.send_message(f"/landmarks/{hand_side}/{name}/z", float(landmark_array[idx + 2]))

            if self.landmark_format in ["array", "both"]:
                # Send as array for batch processing in TouchDesigner
                self.osc_client.send_message(f"/landmarks/{hand_side}/array", landmark_array.tolist())

            # Send derived features for advanced TouchDesigner control
            advanced_features = self.extractor.extract_advanced_features(landmarks)

            # Finger lengths (useful for gesture analysis in TD)
            finger_lengths = advanced_features.finger_lengths
            for i, length in enumerate(finger_lengths):
                self.osc_client.send_message(f"/landmarks/{hand_side}/finger_{i}/length", float(length))

            # Finger angles (useful for pose analysis in TD)
            finger_angles = advanced_features.finger_angles
            for i, angle in enumerate(finger_angles):
                self.osc_client.send_message(f"/landmarks/{hand_side}/finger_{i}/angle", float(angle))

            # Hand span and orientation
            self.osc_client.send_message(f"/landmarks/{hand_side}/hand_span_x", float(advanced_features.hand_span[0]))
            self.osc_client.send_message(f"/landmarks/{hand_side}/hand_span_y", float(advanced_features.hand_span[1]))
            self.osc_client.send_message(f"/landmarks/{hand_side}/orientation", float(advanced_features.hand_orientation))

            # Palm center (useful for object positioning in TD)
            self.osc_client.send_message(f"/landmarks/{hand_side}/palm_x", float(advanced_features.palm_center[0]))
            self.osc_client.send_message(f"/landmarks/{hand_side}/palm_y", float(advanced_features.palm_center[1]))
            self.osc_client.send_message(f"/landmarks/{hand_side}/palm_z", float(advanced_features.palm_center[2]))

        except Exception as e:
            logger.error(f"Failed to send {hand_side} hand landmarks: {e}")

    def send_gesture_to_touchdesigner(self, gesture, confidence, left_hand, right_hand, hand_positions=None):
        """Send gesture recognition results to TouchDesigner."""
        try:
            # Main gesture message (existing format)
            self.osc_client.send_message("/shadow_puppet/gesture", gesture)
            self.osc_client.send_message("/shadow_puppet/confidence", confidence)

            # Individual hand animals
            self.osc_client.send_message("/shadow_puppet/left_hand", left_hand if left_hand else "none")
            self.osc_client.send_message("/shadow_puppet/right_hand", right_hand if right_hand else "none")

            # Hand count
            hand_count = (1 if left_hand else 0) + (1 if right_hand else 0)
            self.osc_client.send_message("/shadow_puppet/hand_count", hand_count)

            # Additional data
            timestamp = time.time()
            self.osc_client.send_message("/shadow_puppet/timestamp", timestamp)

            # Hand positions if available
            if hand_positions:
                if hand_positions.get('left'):
                    self.osc_client.send_message("/shadow_puppet/left_x", hand_positions['left'][0])
                    self.osc_client.send_message("/shadow_puppet/left_y", hand_positions['left'][1])
                if hand_positions.get('right'):
                    self.osc_client.send_message("/shadow_puppet/right_x", hand_positions['right'][0])
                    self.osc_client.send_message("/shadow_puppet/right_y", hand_positions['right'][1])

            # Animal indices for TouchDesigner arrays
            left_index = self.classes.index(left_hand) if left_hand and left_hand in self.classes else -1
            right_index = self.classes.index(right_hand) if right_hand and right_hand in self.classes else -1
            self.osc_client.send_message("/shadow_puppet/left_index", left_index)
            self.osc_client.send_message("/shadow_puppet/right_index", right_index)

            # Status
            self.osc_client.send_message("/shadow_puppet/status", "confirmed")

            # Also send simplified confirmed label on secondary port
            self.send_selected_label(gesture)

            self.predictions_sent += 1
            logger.info(f"Sent to TouchDesigner: {gesture} ({confidence:.1%}) - Predictions: {self.predictions_sent}, Landmarks: {self.landmarks_sent}")

        except Exception as e:
            logger.error(f"Failed to send gesture to TouchDesigner: {e}")

    def send_selected_label(self, label):
        """Send only the confirmed (selected) gesture label to the secondary OSC port.

        Path: /shadow_puppet/selected_label
        When no valid label is available, sends 'none'.
        """
        try:
            if self.label_client is None:
                return
            if not label:
                label = "none"
            self.label_client.send_message("/shadow_puppet/selected_label", label)
        except Exception as e:
            logger.error(f"Failed to send selected label: {e}")

    def send_status_to_touchdesigner(self, status="detecting"):
        """Send status message to TouchDesigner."""
        try:
            self.osc_client.send_message("/shadow_puppet/status", status)
            if status == "no_hands":
                self.osc_client.send_message("/shadow_puppet/gesture", "none")
                self.osc_client.send_message("/shadow_puppet/confidence", 0.0)
        except Exception as e:
            logger.error(f"Failed to send status: {e}")

    def calculate_hand_position(self, landmarks):
        """Calculate normalized hand center position."""
        if not landmarks or not landmarks.landmarks:
            return None

        # Use wrist position as hand center
        wrist = landmarks.landmarks[0]  # Wrist is index 0

        # Normalize to 0-1 range (assuming 640x480 camera)
        normalized_x = wrist[0] / 640.0
        normalized_y = wrist[1] / 480.0

        return (normalized_x, normalized_y)

    def update_fps(self):
        """Update FPS calculation."""
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time

    def run_enhanced_bridge(self):
        """Run the enhanced TouchDesigner bridge with landmark streaming."""
        logger.info("Starting Enhanced MediaPipe TouchDesigner bridge...")
        logger.info("Streaming: Gestures + Landmarks + Hand Positions")

        # Open camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Could not open camera")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                self.frame_count += 1

                # Flip frame horizontally (mirror effect)
                frame = cv2.flip(frame, 1)

                # Predict both hands
                two_hand_detection, left_animal, left_confidence, right_animal, right_confidence = self.predict_both_hands(frame)

                if two_hand_detection and two_hand_detection.has_any_hand():
                    # Stream landmark data to TouchDesigner
                    if self.stream_landmarks:
                        self.send_landmark_data_to_touchdesigner(two_hand_detection)

                    # Update stable output buffer with gesture recognition
                    status = self.output_buffer.update(left_animal, left_confidence, right_animal, right_confidence)

                    # Send gesture recognition to TouchDesigner if confirmed
                    if status['should_send'] and status['confirmed_gesture']:
                        # Calculate hand positions
                        hand_positions = {}
                        if two_hand_detection.left_hand:
                            hand_positions['left'] = self.calculate_hand_position(two_hand_detection.left_hand)
                        if two_hand_detection.right_hand:
                            hand_positions['right'] = self.calculate_hand_position(two_hand_detection.right_hand)

                        # Send confirmed gesture to TouchDesigner
                        self.send_gesture_to_touchdesigner(
                            status['confirmed_gesture'],
                            status['confirmed_confidence'],
                            status['confirmed_left_hand'],
                            status['confirmed_right_hand'],
                            hand_positions
                        )

                        # Mark as sent
                        self.output_buffer.mark_sent()

                    # Visualize both hands
                    frame = self.extractor.visualize_both_hands(frame, two_hand_detection)

                    # Draw enhanced overlay
                    self.draw_enhanced_overlay(frame, two_hand_detection, left_animal, left_confidence, right_animal, right_confidence, status)

                else:
                    # No hands detected
                    self.send_status_to_touchdesigner("no_hands")
                    self.draw_no_hands_overlay(frame)

                # Update FPS
                self.update_fps()

                # Display frame
                cv2.imshow('Enhanced MediaPipe TouchDesigner Bridge', frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # Reset buffer
                    from mediapipe_touchdesigner_bridge import TwoHandStableOutputBuffer
                    self.output_buffer = TwoHandStableOutputBuffer(
                        self.output_buffer.confidence_threshold,
                        self.output_buffer.stability_duration
                    )
                    logger.info("Output buffer reset")
                elif key == ord('l'):
                    # Toggle landmark streaming
                    self.stream_landmarks = not self.stream_landmarks
                    logger.info(f"Landmark streaming: {self.stream_landmarks}")
                elif key == ord('f'):
                    # Cycle landmark format
                    formats = ["individual", "array", "both"]
                    current_index = formats.index(self.landmark_format)
                    self.landmark_format = formats[(current_index + 1) % len(formats)]
                    logger.info(f"Landmark format: {self.landmark_format}")

        except KeyboardInterrupt:
            logger.info("Bridge interrupted by user")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.extractor.cleanup()
            logger.info("Enhanced TouchDesigner bridge stopped")

    def draw_enhanced_overlay(self, frame, two_hand_detection, left_animal, left_confidence, right_animal, right_confidence, status):
        """Draw enhanced overlay with landmark streaming info."""
        h, w = frame.shape[:2]

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w-10, 280), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # Title
        cv2.putText(frame, "Enhanced MediaPipe → TouchDesigner",
                   (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # TouchDesigner output
        if status['confirmed_gesture']:
            cv2.putText(frame, f"TouchDesigner: {status['confirmed_gesture']}",
                       (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "TouchDesigner: Detecting...",
                       (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Streaming status
        y_pos = 95
        streaming_info = []
        if self.stream_landmarks:
            streaming_info.append(f"Landmarks ({self.landmark_format})")
        streaming_info.append("Gestures")

        cv2.putText(frame, f"Streaming: {', '.join(streaming_info)}",
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        y_pos += 25

        # Hand detections
        cv2.putText(frame, f"Hands: {two_hand_detection.num_hands_detected}/2",
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += 25

        # Left hand
        if left_animal:
            color = (255, 0, 0) if left_confidence >= self.output_buffer.confidence_threshold else (128, 0, 0)
            cv2.putText(frame, f"Left: {left_animal} ({left_confidence:.1%})",
                       (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            cv2.putText(frame, "Left: No hand",
                       (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)
        y_pos += 20

        # Right hand
        if right_animal:
            color = (0, 0, 255) if right_confidence >= self.output_buffer.confidence_threshold else (0, 0, 128)
            cv2.putText(frame, f"Right: {right_animal} ({right_confidence:.1%})",
                       (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            cv2.putText(frame, "Right: No hand",
                       (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)
        y_pos += 25

        # Status
        cv2.putText(frame, f"Status: {status['status_message']}",
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        y_pos += 20

        # Performance stats
        cv2.putText(frame, f"FPS: {self.fps:.1f} | Gestures: {self.predictions_sent} | Landmarks: {self.landmarks_sent}",
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        y_pos += 25

        # Controls
        cv2.putText(frame, "Controls: 'q'=quit, 'r'=reset, 'l'=toggle landmarks, 'f'=format",
                   (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def draw_no_hands_overlay(self, frame):
        """Draw overlay when no hands are detected."""
        h, w = frame.shape[:2]

        cv2.putText(frame, "No hands detected",
                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(frame, "Show your hands to start detection",
                   (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def main():
    """Main enhanced bridge function."""
    print("Enhanced MediaPipe TouchDesigner Bridge")
    print("=" * 45)
    print("Streams gesture recognition AND raw landmark data")
    print("Optimized for TouchDesigner integration")

    try:
        # Configuration for live demo
        td_ip = "127.0.0.1"
        td_port = 7000
        confidence_threshold = 0.7    # Higher for demo stability
        stability_duration = 1.0      # Faster response for demo
        stream_landmarks = True       # Enable landmark streaming
        stream_video = False          # Video streaming (future feature)

        print(f"\nConfiguration:")
        print(f"TouchDesigner: {td_ip}:{td_port}")
        print(f"Confidence threshold: {confidence_threshold:.0%}")
        print(f"Stability duration: {stability_duration:.1f}s")
        print(f"Landmark streaming: {stream_landmarks}")

        bridge = EnhancedTouchDesignerBridge(
            td_ip, td_port, confidence_threshold, stability_duration,
            stream_landmarks, stream_video
        )

        bridge.run_enhanced_bridge()

        print("\n[SUCCESS] Enhanced TouchDesigner bridge completed!")

    except Exception as e:
        print(f"[ERROR] Enhanced bridge failed: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())