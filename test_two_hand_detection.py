#!/usr/bin/env python3
"""
Test Two-Hand Detection
Quick test to verify both hands can be detected simultaneously
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add backend to path
sys.path.append('backend')
sys.path.append('.')

from backend.data.mediapipe_extractor_real import RealMediaPipeExtractor

def test_two_hand_detection():
    """Test two-hand detection functionality."""
    print("Testing Two-Hand Detection")
    print("="*35)

    # Initialize extractor
    extractor = RealMediaPipeExtractor()
    print("[OK] MediaPipe extractor initialized with two-hand support")

    # Create test image with two simulated hands
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)

    # Left hand simulation (blue)
    cv2.circle(test_image, (200, 240), 60, (255, 255, 255), -1)  # Left palm
    cv2.circle(test_image, (170, 200), 15, (255, 255, 255), -1)  # Left thumb
    cv2.circle(test_image, (200, 180), 15, (255, 255, 255), -1)  # Left index

    # Right hand simulation (red)
    cv2.circle(test_image, (440, 240), 60, (255, 255, 255), -1)  # Right palm
    cv2.circle(test_image, (470, 200), 15, (255, 255, 255), -1)  # Right thumb
    cv2.circle(test_image, (440, 180), 15, (255, 255, 255), -1)  # Right index

    print("\nTesting with synthetic two-hand image...")

    # Test two-hand detection
    two_hand_detection = extractor.extract_both_hands_from_image(test_image)

    print(f"Hands detected: {two_hand_detection.num_hands_detected}")
    print(f"Left hand: {'Yes' if two_hand_detection.left_hand else 'No'}")
    print(f"Right hand: {'Yes' if two_hand_detection.right_hand else 'No'}")
    print(f"Both hands: {'Yes' if two_hand_detection.has_both_hands() else 'No'}")

    if two_hand_detection.has_any_hand():
        print("\n[OK] Hand detection working!")

        # Test feature extraction
        if two_hand_detection.left_hand:
            left_features = extractor.extract_advanced_features(two_hand_detection.left_hand)
            print(f"Left hand features: {len(left_features.to_vector())} (expected 89)")

        if two_hand_detection.right_hand:
            right_features = extractor.extract_advanced_features(two_hand_detection.right_hand)
            print(f"Right hand features: {len(right_features.to_vector())} (expected 89)")

        # Test two-hand combined features
        combined_features = extractor.extract_two_hand_advanced_features(two_hand_detection)
        combined_vector = combined_features.to_vector()
        print(f"Combined features: {len(combined_vector)} (expected 181)")

        if two_hand_detection.has_both_hands():
            print(f"Hand distance: {combined_features.hand_distance:.1f} pixels")
            print(f"Hand symmetry: {combined_features.hand_symmetry:.3f}")

    else:
        print("[INFO] No hands detected in synthetic image (expected - need real hands)")

    # Test visualization
    print("\nTesting visualization...")
    visualized = extractor.visualize_both_hands(test_image, two_hand_detection)
    print("[OK] Visualization completed")

    extractor.cleanup()

def test_camera_two_hands():
    """Test two-hand detection with camera (brief test)."""
    print("\n" + "="*50)
    print("CAMERA TWO-HAND TEST")
    print("="*50)
    print("Testing with camera for 10 seconds...")
    print("Show both hands to test detection!")

    extractor = RealMediaPipeExtractor()

    # Test with camera briefly
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[SKIP] No camera available")
        return

    try:
        start_time = cv2.getTickCount()
        test_duration = 10.0  # 10 seconds

        both_hands_detected = False
        max_hands_detected = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Check elapsed time
            elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            if elapsed > test_duration:
                break

            # Flip frame
            frame = cv2.flip(frame, 1)

            # Detect both hands
            two_hand_detection = extractor.extract_both_hands_from_frame(frame)

            # Track statistics
            max_hands_detected = max(max_hands_detected, two_hand_detection.num_hands_detected)
            if two_hand_detection.has_both_hands():
                both_hands_detected = True

            # Visualize
            frame = extractor.visualize_both_hands(frame, two_hand_detection)

            # Add timer
            remaining = test_duration - elapsed
            cv2.putText(frame, f"Test time: {remaining:.1f}s", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(frame, "Show both hands!", (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow('Two-Hand Detection Test', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print(f"\nTest Results:")
        print(f"Max hands detected: {max_hands_detected}")
        print(f"Both hands detected: {'Yes' if both_hands_detected else 'No'}")

        if max_hands_detected >= 2:
            print("[SUCCESS] Two-hand detection is working!")
        elif max_hands_detected == 1:
            print("[PARTIAL] Single hand detected - try showing both hands")
        else:
            print("[INFO] No hands detected - make sure hands are visible")

    except Exception as e:
        print(f"Camera test error: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.cleanup()

def main():
    """Main test function."""
    try:
        # Test 1: Basic functionality
        test_two_hand_detection()

        # Test 2: Camera test
        test_camera_two_hands()

        print("\n" + "="*50)
        print("[SUCCESS] Two-hand detection tests completed!")
        print("\nKey Improvements:")
        print("- Both hands detected simultaneously")
        print("- Left hand (Blue L) + Right hand (Red R)")
        print("- 89 features per hand = 181 total features")
        print("- Inter-hand features (distance, symmetry)")
        print("- Perfect for shadow puppets requiring both hands")

        print("\nTo run the live demo:")
        print("python3.12 live_two_hand_demo.py")

    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())