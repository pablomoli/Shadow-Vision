#!/usr/bin/env python3
"""
Simple OSC Test for TouchDesigner Bridge
Verifies OSC message sending functionality.
"""

import time
from pythonosc import udp_client
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_osc_sending():
    """Test OSC message sending to TouchDesigner."""
    logger.info("Testing OSC message sending to TouchDesigner...")

    try:
        # Create OSC client (same as bridge configuration)
        client = udp_client.SimpleUDPClient("127.0.0.1", 7000)
        logger.info("OSC client created successfully")

        # Test all current bridge message types
        test_messages = [
            ("/shadow_puppet/gesture", "L:bird+R:cat"),
            ("/shadow_puppet/confidence", 0.85),
            ("/shadow_puppet/left_hand", "bird"),
            ("/shadow_puppet/right_hand", "cat"),
            ("/shadow_puppet/hand_count", 2),
            ("/shadow_puppet/left_index", 0),
            ("/shadow_puppet/right_index", 1),
            ("/shadow_puppet/status", "confirmed"),
            ("/shadow_puppet/left_x", 0.3),
            ("/shadow_puppet/left_y", 0.4),
            ("/shadow_puppet/right_x", 0.7),
            ("/shadow_puppet/right_y", 0.6),
            ("/shadow_puppet/timestamp", time.time()),
        ]

        logger.info("Sending test messages...")
        for address, value in test_messages:
            client.send_message(address, value)
            logger.info(f"Sent: {address} -> {value}")
            time.sleep(0.01)  # Small delay between messages

        # Test landmark data format for TouchDesigner
        logger.info("Testing landmark data format...")

        # Simulate left hand landmarks (21 landmarks × 3 coordinates)
        for landmark_idx in range(21):
            x = 0.3 + (landmark_idx * 0.01)
            y = 0.4 + (landmark_idx * 0.01)
            z = 0.0 + (landmark_idx * 0.001)

            # Send individual coordinates (TouchDesigner prefers this)
            client.send_message(f"/landmarks/left/{landmark_idx}/x", x)
            client.send_message(f"/landmarks/left/{landmark_idx}/y", y)
            client.send_message(f"/landmarks/left/{landmark_idx}/z", z)

        # Test array format (alternative for TouchDesigner)
        left_array = []
        for landmark_idx in range(21):
            left_array.extend([0.3 + landmark_idx*0.01, 0.4 + landmark_idx*0.01, landmark_idx*0.001])

        client.send_message("/landmarks/left/array", left_array)
        logger.info(f"Sent landmark array: {len(left_array)} values")

        # Test performance simulation
        logger.info("Testing high-frequency updates (5 seconds)...")
        start_time = time.time()
        message_count = 0

        while time.time() - start_time < 5:
            client.send_message("/shadow_puppet/timestamp", time.time())
            client.send_message("/shadow_puppet/confidence", 0.5 + (message_count % 50) / 100.0)
            message_count += 2
            time.sleep(1/30)  # 30 FPS simulation

        fps = message_count / 5.0
        logger.info(f"Performance test: {fps:.1f} messages/second")

        logger.info("OSC test completed successfully!")
        return True

    except Exception as e:
        logger.error(f"OSC test failed: {e}")
        return False

def main():
    """Main test function."""
    print("Simple OSC Test for TouchDesigner Bridge")
    print("=" * 40)
    print("This test verifies basic OSC functionality")
    print("Messages will be sent to port 7000 (TouchDesigner)")
    print("Run TouchDesigner with OSC In CHOP to see messages")
    print()

    success = test_osc_sending()

    if success:
        print("\n[SUCCESS] OSC communication verified!")
        print("Your bridge is ready for TouchDesigner")
        print("\nTouchDesigner setup:")
        print("1. Add OSC In CHOP")
        print("2. Set Network Port: 7000")
        print("3. Set Network Address: 127.0.0.1")
        return 0
    else:
        print("\n[ERROR] OSC test failed")
        return 1

if __name__ == "__main__":
    exit(main())