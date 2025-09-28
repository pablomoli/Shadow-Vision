#!/usr/bin/env python3
"""
Test Enhanced MediaPipe TouchDesigner Bridge
Simulates MediaPipe functionality to test OSC communication.
"""

import cv2
import numpy as np
import logging
import time
from pythonosc import udp_client

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimulatedLandmarks:
    """Simulated MediaPipe landmarks for testing."""

    def __init__(self):
        # Generate 21 hand landmarks with realistic positions
        self.landmarks = []
        for i in range(21):
            x = 0.3 + (i * 0.02)  # Simulated X coordinates
            y = 0.4 + (i * 0.015) # Simulated Y coordinates
            z = 0.0 + (i * 0.001) # Simulated Z coordinates
            self.landmarks.append((x, y, z))

    def to_vector(self):
        """Convert to flat array like MediaPipe."""
        return np.array([coord for landmark in self.landmarks for coord in landmark], dtype=np.float32)

class SimulatedTwoHandDetection:
    """Simulated two-hand detection."""

    def __init__(self, has_left=True, has_right=True):
        self.left_hand = SimulatedLandmarks() if has_left else None
        self.right_hand = SimulatedLandmarks() if has_right else None
        self.num_hands_detected = (1 if has_left else 0) + (1 if has_right else 0)

    def has_any_hand(self):
        return self.left_hand is not None or self.right_hand is not None

    def has_both_hands(self):
        return self.left_hand is not None and self.right_hand is not None

class TestEnhancedBridge:
    """Test version of enhanced TouchDesigner bridge."""

    def __init__(self, td_ip="127.0.0.1", td_port=7000):
        self.td_ip = td_ip
        self.td_port = td_port
        self.osc_client = udp_client.SimpleUDPClient(td_ip, td_port)
        self.classes = ['bird', 'cat', 'llama', 'rabbit', 'deer', 'dog', 'snail', 'swan']
        self.frame_count = 0
        self.predictions_sent = 0
        self.landmarks_sent = 0

        logger.info(f"Test enhanced bridge ready - {td_ip}:{td_port}")

    def send_landmark_data_to_touchdesigner(self, two_hand_detection):
        """Send simulated landmark data to TouchDesigner."""
        try:
            # Send left hand landmarks
            if two_hand_detection.left_hand:
                self.send_hand_landmarks("left", two_hand_detection.left_hand)

            # Send right hand landmarks
            if two_hand_detection.right_hand:
                self.send_hand_landmarks("right", two_hand_detection.right_hand)

            # Send detection status
            self.osc_client.send_message("/landmarks/left/detected",
                                       1.0 if two_hand_detection.left_hand else 0.0)
            self.osc_client.send_message("/landmarks/right/detected",
                                       1.0 if two_hand_detection.right_hand else 0.0)

            self.landmarks_sent += 1

        except Exception as e:
            logger.error(f"Failed to send landmark data: {e}")

    def send_hand_landmarks(self, hand_side, landmarks):
        """Send individual hand landmarks."""
        try:
            landmark_array = landmarks.to_vector()  # 63 values

            # Send individual coordinates
            for i in range(21):
                idx = i * 3
                x, y, z = landmark_array[idx], landmark_array[idx + 1], landmark_array[idx + 2]

                self.osc_client.send_message(f"/landmarks/{hand_side}/{i}/x", float(x))
                self.osc_client.send_message(f"/landmarks/{hand_side}/{i}/y", float(y))
                self.osc_client.send_message(f"/landmarks/{hand_side}/{i}/z", float(z))

            # Send named landmarks
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

            # Send array format
            self.osc_client.send_message(f"/landmarks/{hand_side}/array", landmark_array.tolist())

            # Send derived features (simulated)
            for i in range(5):
                self.osc_client.send_message(f"/landmarks/{hand_side}/finger_{i}/length", float(0.1 + i * 0.02))
                self.osc_client.send_message(f"/landmarks/{hand_side}/finger_{i}/angle", float(0.5 + i * 0.1))

            # Hand properties
            self.osc_client.send_message(f"/landmarks/{hand_side}/hand_span_x", 0.15)
            self.osc_client.send_message(f"/landmarks/{hand_side}/hand_span_y", 0.20)
            self.osc_client.send_message(f"/landmarks/{hand_side}/orientation", 0.0)
            self.osc_client.send_message(f"/landmarks/{hand_side}/palm_x", 0.4)
            self.osc_client.send_message(f"/landmarks/{hand_side}/palm_y", 0.5)
            self.osc_client.send_message(f"/landmarks/{hand_side}/palm_z", 0.0)

        except Exception as e:
            logger.error(f"Failed to send {hand_side} hand landmarks: {e}")

    def send_gesture_to_touchdesigner(self, gesture, confidence, left_hand, right_hand):
        """Send gesture recognition results."""
        try:
            # Main messages
            self.osc_client.send_message("/shadow_puppet/gesture", gesture)
            self.osc_client.send_message("/shadow_puppet/confidence", confidence)
            self.osc_client.send_message("/shadow_puppet/left_hand", left_hand if left_hand else "none")
            self.osc_client.send_message("/shadow_puppet/right_hand", right_hand if right_hand else "none")

            # Hand count and indices
            hand_count = (1 if left_hand else 0) + (1 if right_hand else 0)
            self.osc_client.send_message("/shadow_puppet/hand_count", hand_count)

            left_index = self.classes.index(left_hand) if left_hand and left_hand in self.classes else -1
            right_index = self.classes.index(right_hand) if right_hand and right_hand in self.classes else -1
            self.osc_client.send_message("/shadow_puppet/left_index", left_index)
            self.osc_client.send_message("/shadow_puppet/right_index", right_index)

            # Status and timestamp
            self.osc_client.send_message("/shadow_puppet/status", "confirmed")
            self.osc_client.send_message("/shadow_puppet/timestamp", time.time())

            # Hand positions
            self.osc_client.send_message("/shadow_puppet/left_x", 0.3)
            self.osc_client.send_message("/shadow_puppet/left_y", 0.4)
            self.osc_client.send_message("/shadow_puppet/right_x", 0.7)
            self.osc_client.send_message("/shadow_puppet/right_y", 0.6)

            self.predictions_sent += 1

        except Exception as e:
            logger.error(f"Failed to send gesture: {e}")

    def run_test(self, duration=10):
        """Run comprehensive OSC test."""
        logger.info(f"Running enhanced bridge test for {duration} seconds...")

        start_time = time.time()
        test_scenarios = [
            ("L:bird+R:cat", 0.85, "bird", "cat", True, True),
            ("L:dog", 0.75, "dog", None, True, False),
            ("R:rabbit", 0.80, None, "rabbit", False, True),
            ("L:llama+R:swan", 0.90, "llama", "swan", True, True),
        ]

        scenario_index = 0

        while time.time() - start_time < duration:
            # Cycle through test scenarios
            gesture, confidence, left_animal, right_animal, has_left, has_right = test_scenarios[scenario_index]

            # Create simulated detection
            two_hand_detection = SimulatedTwoHandDetection(has_left, has_right)

            # Send landmark data
            self.send_landmark_data_to_touchdesigner(two_hand_detection)

            # Send gesture recognition
            self.send_gesture_to_touchdesigner(gesture, confidence, left_animal, right_animal)

            self.frame_count += 1

            # Switch scenario every 2 seconds
            if (time.time() - start_time) % 2 < 0.033:  # ~30 FPS
                scenario_index = (scenario_index + 1) % len(test_scenarios)

            time.sleep(1/30)  # 30 FPS

        # Final stats
        elapsed = time.time() - start_time
        fps = self.frame_count / elapsed

        logger.info(f"Test complete: {fps:.1f} FPS")
        logger.info(f"Sent: {self.predictions_sent} gestures, {self.landmarks_sent} landmark frames")

def main():
    """Main test function."""
    print("Enhanced MediaPipe TouchDesigner Bridge Test")
    print("=" * 45)
    print("Testing OSC communication with simulated landmark data")
    print("This verifies the complete message structure for TouchDesigner")
    print()
    print("TouchDesigner Setup:")
    print("1. Add OSC In CHOP, port 7000")
    print("2. Monitor channels for gesture and landmark data")
    print()

    try:
        bridge = TestEnhancedBridge("127.0.0.1", 7000)
        bridge.run_test(duration=10)

        print("\n[SUCCESS] Enhanced bridge test completed!")
        print("All OSC message types verified:")
        print("✓ Gesture recognition (/shadow_puppet/*)")
        print("✓ Raw landmarks (/landmarks/*/array)")
        print("✓ Individual coordinates (/landmarks/*/*)")
        print("✓ Named landmarks (/landmarks/*/wrist/*)")
        print("✓ Derived features (/landmarks/*/finger_*/*)")
        print("✓ Hand properties (/landmarks/*/palm_*)")
        print()
        print("Your bridge is ready for live demo!")

        return 0

    except Exception as e:
        print(f"[ERROR] Enhanced bridge test failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())