#!/usr/bin/env python3
"""
MediaPipe TouchDesigner OSC Bridge with Stable Output
Sends stable gesture recognition results to TouchDesigner via OSC.
Only sends answers when confident and stable over time.
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

class TwoHandStableOutputBuffer:
    """Buffer system to ensure stable two-hand output to TouchDesigner."""

    def __init__(self, confidence_threshold=0.6, stability_duration=1.5):
        """Initialize buffer with stability requirements."""
        self.confidence_threshold = confidence_threshold
        self.stability_duration = stability_duration

        # Current confirmed output (what TouchDesigner sees)
        self.confirmed_gesture = None  # This can be single animal or "left+right" combination
        self.confirmed_confidence = 0.0
        self.confirmed_since = None
        self.confirmed_left_hand = None
        self.confirmed_right_hand = None

        # Candidate for new output
        self.candidate_gesture = None
        self.candidate_start_time = None
        self.candidate_count = 0
        self.candidate_left_hand = None
        self.candidate_right_hand = None

        # Last sent to TouchDesigner
        self.last_sent_gesture = None
        self.last_sent_time = 0

        logger.info(f"Two-hand stable output buffer: {confidence_threshold:.0%} confidence, {stability_duration:.1f}s duration")

    def update(self, left_animal, left_confidence, right_animal, right_confidence):
        """Update buffer with two-hand predictions."""
        current_time = time.time()

        # Create gesture description
        gesture_parts = []
        avg_confidence = 0.0
        confidence_count = 0

        if left_animal and left_confidence >= self.confidence_threshold:
            gesture_parts.append(f"L:{left_animal}")
            avg_confidence += left_confidence
            confidence_count += 1

        if right_animal and right_confidence >= self.confidence_threshold:
            gesture_parts.append(f"R:{right_animal}")
            avg_confidence += right_confidence
            confidence_count += 1

        if confidence_count == 0:
            return self._get_status("Low confidence - not considering", left_animal, left_confidence, right_animal, right_confidence)

        # Calculate average confidence
        avg_confidence = avg_confidence / confidence_count

        # Create gesture string
        current_gesture = "+".join(gesture_parts) if gesture_parts else None

        if not current_gesture:
            return self._get_status("No valid gestures", left_animal, left_confidence, right_animal, right_confidence)

        # Check if this matches our confirmed gesture
        if self.confirmed_gesture == current_gesture:
            return self._get_status(f"Confirmed: {current_gesture}", left_animal, left_confidence, right_animal, right_confidence)

        # Check if this matches our current candidate
        if self.candidate_gesture == current_gesture:
            self.candidate_count += 1
            time_elapsed = current_time - self.candidate_start_time

            # Check if candidate has been stable long enough
            if time_elapsed >= self.stability_duration:
                # Promote candidate to confirmed
                self.confirmed_gesture = current_gesture
                self.confirmed_confidence = avg_confidence
                self.confirmed_since = current_time
                self.confirmed_left_hand = left_animal if left_confidence >= self.confidence_threshold else None
                self.confirmed_right_hand = right_animal if right_confidence >= self.confidence_threshold else None

                # Reset candidate
                self.candidate_gesture = None
                self.candidate_start_time = None
                self.candidate_count = 0
                self.candidate_left_hand = None
                self.candidate_right_hand = None

                return self._get_status(f"NEW CONFIRMED: {current_gesture}", left_animal, left_confidence, right_animal, right_confidence)
            else:
                remaining = self.stability_duration - time_elapsed
                return self._get_status(f"Evaluating {current_gesture} ({remaining:.1f}s left)", left_animal, left_confidence, right_animal, right_confidence)

        else:
            # New candidate
            self.candidate_gesture = current_gesture
            self.candidate_start_time = current_time
            self.candidate_count = 1
            self.candidate_left_hand = left_animal if left_confidence >= self.confidence_threshold else None
            self.candidate_right_hand = right_animal if right_confidence >= self.confidence_threshold else None
            return self._get_status(f"New candidate: {current_gesture}", left_animal, left_confidence, right_animal, right_confidence)

    def _get_status(self, message, left_animal, left_confidence, right_animal, right_confidence):
        """Get current buffer status."""
        return {
            'confirmed_gesture': self.confirmed_gesture,
            'confirmed_confidence': self.confirmed_confidence,
            'confirmed_left_hand': self.confirmed_left_hand,
            'confirmed_right_hand': self.confirmed_right_hand,
            'candidate_gesture': self.candidate_gesture,
            'status_message': message,
            'should_send': self.confirmed_gesture != self.last_sent_gesture,
            'time_confirmed': time.time() - self.confirmed_since if self.confirmed_since else 0,
            'current_left': left_animal,
            'current_left_conf': left_confidence,
            'current_right': right_animal,
            'current_right_conf': right_confidence
        }

    def mark_sent(self):
        """Mark current confirmed gesture as sent to TouchDesigner."""
        self.last_sent_gesture = self.confirmed_gesture
        self.last_sent_time = time.time()

class MediaPipeTouchDesignerBridge:
    """Bridge between MediaPipe ML and TouchDesigner with stable output."""

    def __init__(self, td_ip="127.0.0.1", td_port=7000,
                 confidence_threshold=0.6, stability_duration=1.5):
        """Initialize MediaPipe TouchDesigner bridge."""
        self.td_ip = td_ip
        self.td_port = td_port

        # OSC client for sending to TouchDesigner
        self.osc_client = udp_client.SimpleUDPClient(td_ip, td_port)

        # Load MediaPipe model
        self.classes = ['bird', 'cat', 'llama', 'rabbit', 'deer', 'dog', 'snail', 'swan']
        self.load_mediapipe_model()

        # Initialize stable output buffer for two hands
        self.output_buffer = TwoHandStableOutputBuffer(confidence_threshold, stability_duration)

        # Performance tracking
        self.frame_count = 0
        self.predictions_sent = 0
        self.last_fps_time = time.time()
        self.fps = 0.0

        logger.info(f"MediaPipe TouchDesigner bridge ready - sending to {td_ip}:{td_port}")
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

    def send_two_hand_gesture_to_touchdesigner(self, gesture, confidence, left_hand, right_hand, hand_positions=None):
        """Send stable two-hand gesture to TouchDesigner via OSC."""
        try:
            # Main gesture message
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

            # Animal indices for TouchDesigner
            left_index = self.classes.index(left_hand) if left_hand and left_hand in self.classes else -1
            right_index = self.classes.index(right_hand) if right_hand and right_hand in self.classes else -1
            self.osc_client.send_message("/shadow_puppet/left_index", left_index)
            self.osc_client.send_message("/shadow_puppet/right_index", right_index)

            # Status
            self.osc_client.send_message("/shadow_puppet/status", "confirmed")

            self.predictions_sent += 1
            logger.info(f"Sent to TouchDesigner: {gesture} ({confidence:.1%}) - Total sent: {self.predictions_sent}")

        except Exception as e:
            logger.error(f"Failed to send OSC message: {e}")

    def send_status_to_touchdesigner(self, status="detecting"):
        """Send status message to TouchDesigner."""
        try:
            self.osc_client.send_message("/shadow_puppet/status", status)
            self.osc_client.send_message("/shadow_puppet/animal", "none")
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

    def run_bridge(self):
        """Run the TouchDesigner bridge with camera input."""
        logger.info("Starting MediaPipe TouchDesigner bridge...")
        logger.info("Only stable, confident predictions will be sent to TouchDesigner")

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
                    # Update stable output buffer with both hands
                    status = self.output_buffer.update(left_animal, left_confidence, right_animal, right_confidence)

                    # Send to TouchDesigner only if we have a confirmed, new gesture
                    if status['should_send'] and status['confirmed_gesture']:
                        # Calculate hand positions
                        hand_positions = {}
                        if two_hand_detection.left_hand:
                            hand_positions['left'] = self.calculate_hand_position(two_hand_detection.left_hand)
                        if two_hand_detection.right_hand:
                            hand_positions['right'] = self.calculate_hand_position(two_hand_detection.right_hand)

                        # Send confirmed gesture to TouchDesigner
                        self.send_two_hand_gesture_to_touchdesigner(
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

                    # Draw overlay with both real-time and confirmed info
                    self.draw_two_hand_bridge_overlay(frame, two_hand_detection, left_animal, left_confidence, right_animal, right_confidence, status)

                else:
                    # No hands detected - send status to TouchDesigner
                    self.send_status_to_touchdesigner("no_hands")

                    # Draw "no hands" overlay
                    self.draw_no_hands_overlay(frame)

                # Update FPS
                self.update_fps()

                # Display frame
                cv2.imshow('MediaPipe TouchDesigner Bridge', frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # Reset buffer
                    self.output_buffer = TwoHandStableOutputBuffer(
                        self.output_buffer.confidence_threshold,
                        self.output_buffer.stability_duration
                    )
                    logger.info("Output buffer reset")

        except KeyboardInterrupt:
            logger.info("Bridge interrupted by user")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.extractor.cleanup()
            logger.info("TouchDesigner bridge stopped")

    def draw_bridge_overlay(self, frame, current_animal, current_confidence, status):
        """Draw overlay showing both real-time and TouchDesigner output."""
        h, w = frame.shape[:2]

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w-10, 180), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # Title
        cv2.putText(frame, "MediaPipe -> TouchDesigner Bridge",
                   (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # TouchDesigner output (what's actually sent)
        if status['confirmed_animal']:
            cv2.putText(frame, f"TouchDesigner: {status['confirmed_animal'].upper()}",
                       (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Confirmed for: {status['time_confirmed']:.1f}s",
                       (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "TouchDesigner: Detecting...",
                       (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Real-time detection (what model sees now)
        if current_animal:
            color = (0, 255, 255) if current_confidence >= self.output_buffer.confidence_threshold else (0, 165, 255)
            cv2.putText(frame, f"Real-time: {current_animal} ({current_confidence:.1%})",
                       (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Status
        cv2.putText(frame, f"Status: {status['status_message']}",
                   (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Stats
        cv2.putText(frame, f"FPS: {self.fps:.1f} | Sent: {self.predictions_sent}",
                   (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # Instructions
        cv2.putText(frame, "Controls: 'q'=quit, 'r'=reset buffer",
                   (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def draw_two_hand_bridge_overlay(self, frame, two_hand_detection, left_animal, left_confidence, right_animal, right_confidence, status):
        """Draw overlay showing both real-time and TouchDesigner two-hand output."""
        h, w = frame.shape[:2]

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w-10, 220), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # Title
        cv2.putText(frame, "Two-Hand MediaPipe -> TouchDesigner",
                   (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # TouchDesigner output (what's actually sent)
        if status['confirmed_gesture']:
            cv2.putText(frame, f"TouchDesigner: {status['confirmed_gesture']}",
                       (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Confirmed for: {status['time_confirmed']:.1f}s",
                       (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "TouchDesigner: Detecting...",
                       (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Real-time hand detections
        y_pos = 110
        cv2.putText(frame, f"Hands: {two_hand_detection.num_hands_detected}/2",
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += 20

        # Left hand
        if left_animal:
            color = (255, 0, 0) if left_confidence >= self.output_buffer.confidence_threshold else (128, 0, 0)
            cv2.putText(frame, f"Left (L): {left_animal} ({left_confidence:.1%})",
                       (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            cv2.putText(frame, "Left (L): No hand",
                       (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)
        y_pos += 20

        # Right hand
        if right_animal:
            color = (0, 0, 255) if right_confidence >= self.output_buffer.confidence_threshold else (0, 0, 128)
            cv2.putText(frame, f"Right (R): {right_animal} ({right_confidence:.1%})",
                       (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            cv2.putText(frame, "Right (R): No hand",
                       (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)
        y_pos += 20

        # Status
        cv2.putText(frame, f"Status: {status['status_message']}",
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        y_pos += 20

        # Stats
        cv2.putText(frame, f"FPS: {self.fps:.1f} | Sent: {self.predictions_sent}",
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # Instructions
        cv2.putText(frame, "Controls: 'q'=quit, 'r'=reset buffer",
                   (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def draw_no_hands_overlay(self, frame):
        """Draw overlay when no hands are detected."""
        h, w = frame.shape[:2]

        cv2.putText(frame, "No hands detected",
                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(frame, "Show your hands to start detection",
                   (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def main():
    """Main bridge function."""
    print("MediaPipe TouchDesigner Bridge")
    print("="*40)
    print("Stable output buffer prevents rapid switching in TouchDesigner")
    print("Only confident, sustained predictions are sent")

    try:
        # Configuration
        td_ip = "127.0.0.1"      # TouchDesigner IP
        td_port = 7000           # TouchDesigner port
        confidence_threshold = 0.6   # 60% minimum confidence
        stability_duration = 1.5     # 1.5 seconds stability requirement

        print(f"\nConfiguration:")
        print(f"TouchDesigner: {td_ip}:{td_port}")
        print(f"Confidence threshold: {confidence_threshold:.0%}")
        print(f"Stability duration: {stability_duration:.1f}s")

        bridge = MediaPipeTouchDesignerBridge(
            td_ip, td_port, confidence_threshold, stability_duration
        )

        bridge.run_bridge()

        print("\n[SUCCESS] TouchDesigner bridge completed!")

    except Exception as e:
        print(f"[ERROR] Bridge failed: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())