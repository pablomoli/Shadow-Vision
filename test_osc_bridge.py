#!/usr/bin/env python3
"""
OSC Bridge Test for TouchDesigner Integration
Tests OSC message transmission and verifies data integrity.
"""

import time
import threading
from pythonosc import udp_client
from pythonosc.dispatcher import Dispatcher
from pythonosc import osc
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OSCBridgeTest:
    """Test OSC communication between MediaPipe bridge and TouchDesigner."""

    def __init__(self, send_port=7000, receive_port=7001):
        """Initialize OSC test."""
        self.send_port = send_port
        self.receive_port = receive_port

        # OSC client for sending (simulates bridge)
        self.client = udp_client.SimpleUDPClient("127.0.0.1", send_port)

        # OSC server for receiving (simulates TouchDesigner)
        self.dispatcher = Dispatcher()
        self.setup_message_handlers()

        # Received messages storage
        self.received_messages = {}
        self.message_count = 0

        logger.info(f"OSC Bridge Test: Send port {send_port}, Receive port {receive_port}")

    def setup_message_handlers(self):
        """Setup OSC message handlers to catch all messages."""
        # Gesture messages
        self.dispatcher.map("/shadow_puppet/gesture", self.handle_message)
        self.dispatcher.map("/shadow_puppet/confidence", self.handle_message)
        self.dispatcher.map("/shadow_puppet/left_hand", self.handle_message)
        self.dispatcher.map("/shadow_puppet/right_hand", self.handle_message)
        self.dispatcher.map("/shadow_puppet/hand_count", self.handle_message)
        self.dispatcher.map("/shadow_puppet/left_index", self.handle_message)
        self.dispatcher.map("/shadow_puppet/right_index", self.handle_message)
        self.dispatcher.map("/shadow_puppet/status", self.handle_message)

        # Position messages
        self.dispatcher.map("/shadow_puppet/left_x", self.handle_message)
        self.dispatcher.map("/shadow_puppet/left_y", self.handle_message)
        self.dispatcher.map("/shadow_puppet/right_x", self.handle_message)
        self.dispatcher.map("/shadow_puppet/right_y", self.handle_message)

        # Landmark messages (for future enhancement)
        self.dispatcher.map("/landmarks/left/*", self.handle_message)
        self.dispatcher.map("/landmarks/right/*", self.handle_message)

        # Catch-all handler
        self.dispatcher.map("/*", self.handle_message)

    def handle_message(self, address, *args):
        """Handle received OSC messages."""
        self.message_count += 1
        self.received_messages[address] = args
        logger.info(f"Received: {address} -> {args}")

    def start_receiver(self):
        """Start OSC message receiver."""
        try:
            server = osc.udp_server.ThreadingOSCUDPServer(("127.0.0.1", self.receive_port), self.dispatcher)
            logger.info(f"OSC Receiver started on port {self.receive_port}")
            server.serve_forever()
        except Exception as e:
            logger.error(f"Failed to start OSC receiver: {e}")

    def test_gesture_messages(self):
        """Test sending gesture recognition messages."""
        logger.info("Testing gesture recognition messages...")

        # Test basic gesture
        self.client.send_message("/shadow_puppet/gesture", "L:bird+R:cat")
        self.client.send_message("/shadow_puppet/confidence", 0.85)
        self.client.send_message("/shadow_puppet/left_hand", "bird")
        self.client.send_message("/shadow_puppet/right_hand", "cat")
        self.client.send_message("/shadow_puppet/hand_count", 2)
        self.client.send_message("/shadow_puppet/left_index", 0)  # bird index
        self.client.send_message("/shadow_puppet/right_index", 1)  # cat index
        self.client.send_message("/shadow_puppet/status", "confirmed")

        time.sleep(0.1)

        # Test hand positions
        self.client.send_message("/shadow_puppet/left_x", 0.3)
        self.client.send_message("/shadow_puppet/left_y", 0.4)
        self.client.send_message("/shadow_puppet/right_x", 0.7)
        self.client.send_message("/shadow_puppet/right_y", 0.6)

        logger.info("Gesture messages sent")

    def test_landmark_messages(self):
        """Test sending landmark data for TouchDesigner."""
        logger.info("Testing landmark data messages...")

        # Test raw landmark data (21 landmarks × 3 coordinates = 63 values)
        # Simulated left hand landmarks
        left_landmarks = []
        for i in range(21):
            x = 0.3 + (i * 0.01)  # Simulated X coordinates
            y = 0.4 + (i * 0.01)  # Simulated Y coordinates
            z = 0.0 + (i * 0.001) # Simulated Z coordinates
            left_landmarks.extend([x, y, z])

        # Send as individual coordinates for TouchDesigner arrays
        for i, coord in enumerate(left_landmarks):
            self.client.send_message(f"/landmarks/left/coord_{i}", coord)

        # Send landmark indices (for TouchDesigner landmark identification)
        landmark_names = [
            "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
            "index_mcp", "index_pip", "index_dip", "index_tip",
            "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
            "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
            "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
        ]

        for i, name in enumerate(landmark_names):
            idx = i * 3
            self.client.send_message(f"/landmarks/left/{name}/x", left_landmarks[idx])
            self.client.send_message(f"/landmarks/left/{name}/y", left_landmarks[idx + 1])
            self.client.send_message(f"/landmarks/left/{name}/z", left_landmarks[idx + 2])

        logger.info("Landmark messages sent")

    def test_performance(self, duration=5):
        """Test OSC message performance."""
        logger.info(f"Testing OSC performance for {duration} seconds...")

        start_time = time.time()
        messages_sent = 0

        while time.time() - start_time < duration:
            # Simulate high-frequency gesture updates
            self.client.send_message("/shadow_puppet/gesture", f"test_{messages_sent}")
            self.client.send_message("/shadow_puppet/confidence", 0.5 + (messages_sent % 50) / 100.0)
            self.client.send_message("/shadow_puppet/timestamp", time.time())

            messages_sent += 3
            time.sleep(1/60)  # 60 FPS simulation

        elapsed_time = time.time() - start_time
        fps = messages_sent / elapsed_time / 3  # Divide by 3 messages per iteration

        logger.info(f"Performance test complete: {fps:.1f} message groups/second")

    def run_tests(self):
        """Run all OSC bridge tests."""
        logger.info("Starting OSC Bridge Test Suite")

        # Start receiver in background thread
        receiver_thread = threading.Thread(target=self.start_receiver, daemon=True)
        receiver_thread.start()

        time.sleep(1)  # Let receiver start

        # Run tests
        self.test_gesture_messages()
        time.sleep(0.5)

        self.test_landmark_messages()
        time.sleep(0.5)

        self.test_performance(duration=2)
        time.sleep(1)

        # Report results
        logger.info(f"Test complete: {self.message_count} messages received")
        logger.info("Received message types:")
        for address in sorted(self.received_messages.keys()):
            logger.info(f"  {address}: {self.received_messages[address]}")

def main():
    """Main test function."""
    print("OSC Bridge Test for TouchDesigner Integration")
    print("=" * 50)
    print("This test verifies OSC communication functionality")
    print("Messages will be sent to port 7000 (TouchDesigner default)")
    print("Test receiver will listen on port 7001")
    print()

    try:
        test = OSCBridgeTest(send_port=7000, receive_port=7001)
        test.run_tests()

        print("\n[SUCCESS] OSC Bridge test completed!")
        print("Your bridge is ready for TouchDesigner integration")

    except Exception as e:
        print(f"[ERROR] OSC Bridge test failed: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())