#!/usr/bin/env python3
"""
REAL MediaPipe Hand Landmark Feature Extractor

This runs inside the MediaPipe Docker container (Python 3.12) and provides
the actual MediaPipe hand landmarks for maximum accuracy.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, List, Tuple, Dict
import logging
import json
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class MediaPipeHandLandmarks:
    """Real MediaPipe hand landmarks (21 points × 3 coordinates = 63 features)."""
    landmarks: List[Tuple[float, float, float]]  # (x, y, z) for 21 landmarks

    # Landmark indices for reference
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

    def to_vector(self) -> np.ndarray:
        """Convert landmarks to flat feature vector (63 features)."""
        return np.array([coord for landmark in self.landmarks for coord in landmark], dtype=np.float32)

    def get_fingertips(self) -> List[Tuple[float, float, float]]:
        """Get all fingertip positions."""
        tip_indices = [self.THUMB_TIP, self.INDEX_FINGER_TIP, self.MIDDLE_FINGER_TIP,
                      self.RING_FINGER_TIP, self.PINKY_TIP]
        return [self.landmarks[i] for i in tip_indices]

    def get_finger_lengths(self) -> List[float]:
        """Calculate finger lengths from MCP to tip."""
        finger_pairs = [
            (self.THUMB_MCP, self.THUMB_TIP),
            (self.INDEX_FINGER_MCP, self.INDEX_FINGER_TIP),
            (self.MIDDLE_FINGER_MCP, self.MIDDLE_FINGER_TIP),
            (self.RING_FINGER_MCP, self.RING_FINGER_TIP),
            (self.PINKY_MCP, self.PINKY_TIP)
        ]

        lengths = []
        for mcp_idx, tip_idx in finger_pairs:
            mcp = np.array(self.landmarks[mcp_idx])
            tip = np.array(self.landmarks[tip_idx])
            length = np.linalg.norm(tip - mcp)
            lengths.append(float(length))

        return lengths

    def get_finger_angles(self) -> List[float]:
        """Calculate finger bend angles at PIP joints."""
        finger_joints = [
            (self.THUMB_MCP, self.THUMB_IP, self.THUMB_TIP),
            (self.INDEX_FINGER_MCP, self.INDEX_FINGER_PIP, self.INDEX_FINGER_TIP),
            (self.MIDDLE_FINGER_MCP, self.MIDDLE_FINGER_PIP, self.MIDDLE_FINGER_TIP),
            (self.RING_FINGER_MCP, self.RING_FINGER_PIP, self.RING_FINGER_TIP),
            (self.PINKY_MCP, self.PINKY_PIP, self.PINKY_TIP)
        ]

        angles = []
        for joint1_idx, joint2_idx, joint3_idx in finger_joints:
            p1 = np.array(self.landmarks[joint1_idx])
            p2 = np.array(self.landmarks[joint2_idx])  # Vertex
            p3 = np.array(self.landmarks[joint3_idx])

            # Vectors from vertex
            v1 = p1 - p2
            v2 = p3 - p2

            # Calculate angle
            dot_product = np.dot(v1, v2)
            norms = np.linalg.norm(v1) * np.linalg.norm(v2)

            if norms > 0:
                cos_angle = np.clip(dot_product / norms, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                angles.append(float(angle))
            else:
                angles.append(0.0)

        return angles

@dataclass
class MediaPipeAdvancedFeatures:
    """Advanced features derived from MediaPipe landmarks."""
    # Raw landmarks (63 features)
    landmarks: MediaPipeHandLandmarks

    # Derived geometric features (40+ features)
    finger_lengths: List[float]           # 5 features
    finger_angles: List[float]            # 5 features
    inter_finger_distances: List[float]   # 10 features (C(5,2) combinations)
    hand_orientation: float               # 1 feature
    hand_span: Tuple[float, float]        # 2 features (width, height)
    palm_center: Tuple[float, float, float] # 3 features

    def to_vector(self) -> np.ndarray:
        """Convert all features to vector."""
        features = []

        # Raw landmarks (63)
        features.extend(self.landmarks.to_vector())

        # Derived features
        features.extend(self.finger_lengths)          # 5
        features.extend(self.finger_angles)           # 5
        features.extend(self.inter_finger_distances)  # 10
        features.append(self.hand_orientation)        # 1
        features.extend(self.hand_span)               # 2
        features.extend(self.palm_center)             # 3

        return np.array(features, dtype=np.float32)   # Total: 89 features

@dataclass
class TwoHandDetection:
    """Detection result for both hands."""
    left_hand: Optional[MediaPipeHandLandmarks]
    right_hand: Optional[MediaPipeHandLandmarks]
    num_hands_detected: int

    def has_both_hands(self) -> bool:
        """Check if both hands are detected."""
        return self.left_hand is not None and self.right_hand is not None

    def has_any_hand(self) -> bool:
        """Check if at least one hand is detected."""
        return self.left_hand is not None or self.right_hand is not None

@dataclass
class TwoHandAdvancedFeatures:
    """Advanced features for both hands combined."""
    left_hand_features: Optional[MediaPipeAdvancedFeatures]
    right_hand_features: Optional[MediaPipeAdvancedFeatures]

    # Inter-hand features (when both hands present)
    hand_distance: Optional[float]        # Distance between palm centers
    hand_relative_position: Optional[Tuple[float, float]]  # Relative positioning
    hand_symmetry: Optional[float]        # Symmetry measure

    def to_vector(self) -> np.ndarray:
        """Convert to feature vector."""
        features = []

        # Left hand features (89 features or zeros)
        if self.left_hand_features:
            features.extend(self.left_hand_features.to_vector())
        else:
            features.extend([0.0] * 89)  # Zero padding for missing hand

        # Right hand features (89 features or zeros)
        if self.right_hand_features:
            features.extend(self.right_hand_features.to_vector())
        else:
            features.extend([0.0] * 89)  # Zero padding for missing hand

        # Inter-hand features (3 features)
        features.append(self.hand_distance if self.hand_distance else 0.0)
        if self.hand_relative_position:
            features.extend(self.hand_relative_position)
        else:
            features.extend([0.0, 0.0])
        features.append(self.hand_symmetry if self.hand_symmetry else 0.0)

        return np.array(features, dtype=np.float32)  # Total: 181 features (89+89+3)

class RealMediaPipeExtractor:
    """Real MediaPipe hand landmark extractor."""

    def __init__(self):
        """Initialize MediaPipe hands."""
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        # For single images (dataset processing)
        self.hands_static = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,  # Enable both hands
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        # For video streams (real-time)
        self.hands_video = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Enable both hands
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        logger.info("Real MediaPipe extractor initialized")

    def extract_both_hands_from_image(self, image: np.ndarray) -> TwoHandDetection:
        """Extract landmarks from both hands in image."""
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process image
            results = self.hands_static.process(rgb_image)

            left_hand = None
            right_hand = None
            num_hands = 0

            if results.multi_hand_landmarks and results.multi_handedness:
                h, w = image.shape[:2]

                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # Extract normalized coordinates
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        x = landmark.x * w
                        y = landmark.y * h
                        z = landmark.z * w
                        landmarks.append((x, y, z))

                    hand_data = MediaPipeHandLandmarks(landmarks=landmarks)

                    # Determine if left or right hand
                    hand_label = handedness.classification[0].label
                    if hand_label == "Left":
                        left_hand = hand_data
                    else:  # "Right"
                        right_hand = hand_data

                    num_hands += 1

            return TwoHandDetection(
                left_hand=left_hand,
                right_hand=right_hand,
                num_hands_detected=num_hands
            )

        except Exception as e:
            logger.error(f"Two-hand landmark extraction failed: {e}")
            return TwoHandDetection(left_hand=None, right_hand=None, num_hands_detected=0)

    def extract_both_hands_from_frame(self, frame: np.ndarray) -> TwoHandDetection:
        """Extract landmarks from both hands in video frame."""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frame
            results = self.hands_video.process(rgb_frame)

            left_hand = None
            right_hand = None
            num_hands = 0

            if results.multi_hand_landmarks and results.multi_handedness:
                h, w = frame.shape[:2]

                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # Extract coordinates
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        x = landmark.x * w
                        y = landmark.y * h
                        z = landmark.z * w
                        landmarks.append((x, y, z))

                    hand_data = MediaPipeHandLandmarks(landmarks=landmarks)

                    # Determine if left or right hand
                    hand_label = handedness.classification[0].label
                    if hand_label == "Left":
                        left_hand = hand_data
                    else:  # "Right"
                        right_hand = hand_data

                    num_hands += 1

            return TwoHandDetection(
                left_hand=left_hand,
                right_hand=right_hand,
                num_hands_detected=num_hands
            )

        except Exception as e:
            logger.error(f"Two-hand frame extraction failed: {e}")
            return TwoHandDetection(left_hand=None, right_hand=None, num_hands_detected=0)

    def extract_landmarks_from_image(self, image: np.ndarray) -> Optional[MediaPipeHandLandmarks]:
        """Extract MediaPipe landmarks from single image."""
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process image
            results = self.hands_static.process(rgb_image)

            if results.multi_hand_landmarks:
                # Get first hand
                hand_landmarks = results.multi_hand_landmarks[0]

                # Extract normalized coordinates
                h, w = image.shape[:2]
                landmarks = []

                for landmark in hand_landmarks.landmark:
                    # Convert normalized coordinates to pixel coordinates
                    x = landmark.x * w
                    y = landmark.y * h
                    z = landmark.z * w  # MediaPipe z is relative to wrist
                    landmarks.append((x, y, z))

                return MediaPipeHandLandmarks(landmarks=landmarks)

            return None

        except Exception as e:
            logger.error(f"MediaPipe landmark extraction failed: {e}")
            return None

    def extract_landmarks_from_frame(self, frame: np.ndarray) -> Optional[MediaPipeHandLandmarks]:
        """Extract MediaPipe landmarks from video frame."""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frame
            results = self.hands_video.process(rgb_frame)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                # Extract coordinates
                h, w = frame.shape[:2]
                landmarks = []

                for landmark in hand_landmarks.landmark:
                    x = landmark.x * w
                    y = landmark.y * h
                    z = landmark.z * w
                    landmarks.append((x, y, z))

                return MediaPipeHandLandmarks(landmarks=landmarks)

            return None

        except Exception as e:
            logger.error(f"Real-time landmark extraction failed: {e}")
            return None

    def extract_advanced_features(self, landmarks: MediaPipeHandLandmarks) -> MediaPipeAdvancedFeatures:
        """Extract advanced geometric features from MediaPipe landmarks."""
        # Finger lengths
        finger_lengths = landmarks.get_finger_lengths()

        # Finger angles
        finger_angles = landmarks.get_finger_angles()

        # Inter-finger distances (all combinations)
        fingertips = landmarks.get_fingertips()
        inter_distances = []
        for i in range(len(fingertips)):
            for j in range(i + 1, len(fingertips)):
                tip1 = np.array(fingertips[i])
                tip2 = np.array(fingertips[j])
                distance = np.linalg.norm(tip1 - tip2)
                inter_distances.append(float(distance))

        # Hand orientation (wrist to middle finger MCP)
        wrist = np.array(landmarks.landmarks[landmarks.WRIST])
        middle_mcp = np.array(landmarks.landmarks[landmarks.MIDDLE_FINGER_MCP])
        orientation_vector = middle_mcp - wrist
        hand_orientation = float(np.arctan2(orientation_vector[1], orientation_vector[0]))

        # Hand span (thumb tip to pinky tip, wrist to middle tip)
        thumb_tip = np.array(landmarks.landmarks[landmarks.THUMB_TIP])
        pinky_tip = np.array(landmarks.landmarks[landmarks.PINKY_TIP])
        middle_tip = np.array(landmarks.landmarks[landmarks.MIDDLE_FINGER_TIP])

        hand_width = float(np.linalg.norm(thumb_tip - pinky_tip))
        hand_length = float(np.linalg.norm(wrist - middle_tip))
        hand_span = (hand_width, hand_length)

        # Palm center (average of wrist and MCP joints)
        mcp_indices = [landmarks.INDEX_FINGER_MCP, landmarks.MIDDLE_FINGER_MCP,
                      landmarks.RING_FINGER_MCP, landmarks.PINKY_MCP]
        palm_points = [wrist] + [np.array(landmarks.landmarks[i]) for i in mcp_indices]
        palm_center = tuple(np.mean(palm_points, axis=0))

        return MediaPipeAdvancedFeatures(
            landmarks=landmarks,
            finger_lengths=finger_lengths,
            finger_angles=finger_angles,
            inter_finger_distances=inter_distances,
            hand_orientation=hand_orientation,
            hand_span=hand_span,
            palm_center=palm_center
        )

    def extract_two_hand_advanced_features(self, two_hand_detection: TwoHandDetection) -> TwoHandAdvancedFeatures:
        """Extract advanced features for both hands."""
        left_features = None
        right_features = None

        # Extract features for each hand
        if two_hand_detection.left_hand:
            left_features = self.extract_advanced_features(two_hand_detection.left_hand)

        if two_hand_detection.right_hand:
            right_features = self.extract_advanced_features(two_hand_detection.right_hand)

        # Calculate inter-hand features if both hands present
        hand_distance = None
        hand_relative_position = None
        hand_symmetry = None

        if left_features and right_features:
            # Distance between palm centers
            left_palm = np.array(left_features.palm_center)
            right_palm = np.array(right_features.palm_center)
            hand_distance = float(np.linalg.norm(right_palm - left_palm))

            # Relative position (right hand relative to left hand)
            relative_pos = right_palm - left_palm
            hand_relative_position = (float(relative_pos[0]), float(relative_pos[1]))

            # Hand symmetry measure (compare finger lengths)
            left_lengths = np.array(left_features.finger_lengths)
            right_lengths = np.array(right_features.finger_lengths)
            hand_symmetry = float(1.0 - np.mean(np.abs(left_lengths - right_lengths)) / np.mean(left_lengths + right_lengths))

        return TwoHandAdvancedFeatures(
            left_hand_features=left_features,
            right_hand_features=right_features,
            hand_distance=hand_distance,
            hand_relative_position=hand_relative_position,
            hand_symmetry=hand_symmetry
        )

    def visualize_landmarks(self, image: np.ndarray, landmarks: MediaPipeHandLandmarks) -> np.ndarray:
        """Draw MediaPipe landmarks on image."""
        if landmarks is None:
            return image

        annotated_image = image.copy()
        h, w = image.shape[:2]

        # Draw landmarks manually since we have the coordinates
        for i, (x, y, z) in enumerate(landmarks.landmarks):
            # Draw landmark point
            cv2.circle(annotated_image, (int(x), int(y)), 5, (0, 0, 255), -1)

            # Draw landmark number
            cv2.putText(annotated_image, str(i), (int(x) + 5, int(y) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        # Draw connections manually (simplified)
        connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index finger
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle finger
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring finger
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20),
            # Palm connections
            (5, 9), (9, 13), (13, 17)
        ]

        for start_idx, end_idx in connections:
            if start_idx < len(landmarks.landmarks) and end_idx < len(landmarks.landmarks):
                start_pos = (int(landmarks.landmarks[start_idx][0]), int(landmarks.landmarks[start_idx][1]))
                end_pos = (int(landmarks.landmarks[end_idx][0]), int(landmarks.landmarks[end_idx][1]))
                cv2.line(annotated_image, start_pos, end_pos, (0, 255, 0), 2)

        return annotated_image

    def visualize_both_hands(self, image: np.ndarray, two_hand_detection: TwoHandDetection) -> np.ndarray:
        """Draw landmarks for both hands on image."""
        annotated_image = image.copy()

        # Draw left hand in blue
        if two_hand_detection.left_hand:
            annotated_image = self.draw_hand_with_color(annotated_image, two_hand_detection.left_hand,
                                                       (255, 0, 0), "L")  # Blue for left

        # Draw right hand in red
        if two_hand_detection.right_hand:
            annotated_image = self.draw_hand_with_color(annotated_image, two_hand_detection.right_hand,
                                                       (0, 0, 255), "R")  # Red for right

        # Add hand count info
        cv2.putText(annotated_image, f"Hands detected: {two_hand_detection.num_hands_detected}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return annotated_image

    def draw_hand_with_color(self, image: np.ndarray, landmarks: MediaPipeHandLandmarks,
                           color: tuple, label: str) -> np.ndarray:
        """Draw hand landmarks with specific color and label."""
        # Draw landmarks
        for i, (x, y, z) in enumerate(landmarks.landmarks):
            cv2.circle(image, (int(x), int(y)), 4, color, -1)

        # Draw connections
        connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index finger
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle finger
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring finger
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20),
            # Palm connections
            (5, 9), (9, 13), (13, 17)
        ]

        for start_idx, end_idx in connections:
            if start_idx < len(landmarks.landmarks) and end_idx < len(landmarks.landmarks):
                start_pos = (int(landmarks.landmarks[start_idx][0]), int(landmarks.landmarks[start_idx][1]))
                end_pos = (int(landmarks.landmarks[end_idx][0]), int(landmarks.landmarks[end_idx][1]))
                cv2.line(image, start_pos, end_pos, color, 2)

        # Add hand label
        if landmarks.landmarks:
            wrist_pos = landmarks.landmarks[0]  # Wrist position
            cv2.putText(image, label, (int(wrist_pos[0]) - 10, int(wrist_pos[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        return image

    def process_dataset_image(self, image_path: str) -> Optional[Dict]:
        """Process single dataset image and return features."""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return None

            # Extract landmarks
            landmarks = self.extract_landmarks_from_image(image)
            if landmarks is None:
                return None

            # Extract advanced features
            advanced_features = self.extract_advanced_features(landmarks)

            # Return feature data
            return {
                'image_path': image_path,
                'landmarks': [list(lm) for lm in landmarks.landmarks],  # Convert to serializable format
                'features': advanced_features.to_vector().tolist(),
                'feature_count': len(advanced_features.to_vector())
            }

        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return None

    def cleanup(self):
        """Clean up MediaPipe resources."""
        if hasattr(self, 'hands_static'):
            self.hands_static.close()
        if hasattr(self, 'hands_video'):
            self.hands_video.close()

def main():
    """Test real MediaPipe extraction."""
    print("Real MediaPipe Hand Landmark Extractor")
    print("=" * 45)

    extractor = RealMediaPipeExtractor()

    # Test with webcam if available
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("Testing with webcam (press 'q' to quit)...")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Flip frame
                frame = cv2.flip(frame, 1)

                # Extract landmarks
                landmarks = extractor.extract_landmarks_from_frame(frame)

                if landmarks is not None:
                    # Extract advanced features
                    advanced_features = extractor.extract_advanced_features(landmarks)
                    feature_vector = advanced_features.to_vector()

                    # Visualize
                    annotated_frame = extractor.visualize_landmarks(frame, landmarks)

                    # Add text
                    cv2.putText(annotated_frame, f"MediaPipe: {len(feature_vector)} features",
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, "Real MediaPipe Landmarks!",
                              (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                    cv2.imshow('Real MediaPipe Test', annotated_frame)
                else:
                    cv2.imshow('Real MediaPipe Test', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

        else:
            print("No webcam available, testing with synthetic image...")

            # Create test image
            test_image = np.zeros((480, 640, 3), dtype=np.uint8)
            # Add some content for MediaPipe to detect
            cv2.circle(test_image, (320, 240), 100, (255, 255, 255), -1)

            landmarks = extractor.extract_landmarks_from_image(test_image)
            if landmarks:
                print(f"✅ Extracted {len(landmarks.landmarks)} landmarks")
            else:
                print("⚠️  No landmarks detected in test image")

    except Exception as e:
        print(f"Test error: {e}")

    finally:
        extractor.cleanup()

if __name__ == "__main__":
    main()