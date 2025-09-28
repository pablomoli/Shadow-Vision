#!/usr/bin/env python3
"""
MediaPipe Hand Landmark Feature Extractor

This is the CORRECT approach for hand gesture recognition using precise
hand landmarks instead of pixel-based contour detection.

MediaPipe provides 21 precise hand landmarks with (x, y, z) coordinates,
giving us 63 features + derived geometric relationships.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, List, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class HandLandmarks:
    """Hand landmarks with 21 keypoints."""
    landmarks: np.ndarray  # Shape: (21, 3) for (x, y, z) coordinates

    def to_vector(self) -> np.ndarray:
        """Convert landmarks to feature vector."""
        return self.landmarks.flatten()  # 63 features

class MediaPipeFeatureExtractor:
    """Extract precise hand features using MediaPipe landmarks."""

    def __init__(self):
        """Initialize MediaPipe hands."""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,     # For single images
            max_num_hands=1,            # Only detect one hand
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # For real-time video
        self.hands_video = self.mp_hands.Hands(
            static_image_mode=False,    # For video streams
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        self.mp_drawing = mp.solutions.drawing_utils
        logger.info("MediaPipe hands initialized")

    def extract_landmarks_from_image(self, image: np.ndarray) -> Optional[HandLandmarks]:
        """Extract hand landmarks from a single image."""
        try:
            # Convert BGR to RGB (MediaPipe requirement)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process image
            results = self.hands.process(rgb_image)

            if results.multi_hand_landmarks:
                # Get first (and only) hand
                hand_landmarks = results.multi_hand_landmarks[0]

                # Extract landmark coordinates
                landmarks = []
                h, w = image.shape[:2]

                for landmark in hand_landmarks.landmark:
                    # Convert normalized coordinates to pixel coordinates
                    x = landmark.x * w
                    y = landmark.y * h
                    z = landmark.z  # Depth (relative)
                    landmarks.append([x, y, z])

                landmarks_array = np.array(landmarks, dtype=np.float32)
                return HandLandmarks(landmarks=landmarks_array)

            return None

        except Exception as e:
            logger.error(f"MediaPipe landmark extraction failed: {e}")
            return None

    def extract_landmarks_from_frame(self, frame: np.ndarray) -> Optional[HandLandmarks]:
        """Extract hand landmarks from video frame (real-time optimized)."""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frame
            results = self.hands_video.process(rgb_frame)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                # Extract landmarks
                landmarks = []
                h, w = frame.shape[:2]

                for landmark in hand_landmarks.landmark:
                    x = landmark.x * w
                    y = landmark.y * h
                    z = landmark.z
                    landmarks.append([x, y, z])

                landmarks_array = np.array(landmarks, dtype=np.float32)
                return HandLandmarks(landmarks=landmarks_array)

            return None

        except Exception as e:
            logger.error(f"Real-time landmark extraction failed: {e}")
            return None

    def extract_advanced_features(self, landmarks: HandLandmarks) -> np.ndarray:
        """Extract advanced geometric features from hand landmarks."""
        if landmarks is None:
            return None

        coords = landmarks.landmarks  # Shape: (21, 3)
        features = []

        # 1. Raw landmark coordinates (63 features)
        features.extend(coords.flatten())

        # 2. Finger tip positions (15 features: 5 fingertips × 3 coords)
        fingertips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
        for tip_idx in fingertips:
            features.extend(coords[tip_idx])

        # 3. Finger distances from wrist (5 features)
        wrist = coords[0]
        for tip_idx in fingertips:
            distance = np.linalg.norm(coords[tip_idx] - wrist)
            features.append(distance)

        # 4. Inter-finger distances (10 features: 5C2 combinations)
        for i in range(len(fingertips)):
            for j in range(i + 1, len(fingertips)):
                tip1, tip2 = coords[fingertips[i]], coords[fingertips[j]]
                distance = np.linalg.norm(tip1 - tip2)
                features.append(distance)

        # 5. Hand orientation (3 features)
        # Vector from wrist to middle finger MCP
        orientation_vector = coords[9] - coords[0]  # Middle finger MCP - Wrist
        features.extend(orientation_vector)

        # 6. Hand span (4 features)
        # Width: Thumb tip to pinky tip
        hand_width = np.linalg.norm(coords[4] - coords[20])
        # Length: Wrist to middle finger tip
        hand_length = np.linalg.norm(coords[12] - coords[0])
        # Diagonal 1: Thumb tip to middle finger tip
        diagonal1 = np.linalg.norm(coords[4] - coords[12])
        # Diagonal 2: Index tip to pinky tip
        diagonal2 = np.linalg.norm(coords[8] - coords[20])
        features.extend([hand_width, hand_length, diagonal1, diagonal2])

        # 7. Finger bend angles (5 features)
        # Calculate angle at each finger's PIP joint
        finger_joints = [
            (2, 3, 4),    # Thumb: MCP -> IP -> TIP
            (5, 6, 8),    # Index: MCP -> PIP -> TIP
            (9, 10, 12),  # Middle: MCP -> PIP -> TIP
            (13, 14, 16), # Ring: MCP -> PIP -> TIP
            (17, 18, 20)  # Pinky: MCP -> PIP -> TIP
        ]

        for joint1, joint2, joint3 in finger_joints:
            # Calculate angle at joint2
            v1 = coords[joint1] - coords[joint2]
            v2 = coords[joint3] - coords[joint2]

            # Angle between vectors
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            features.append(angle)

        # 8. Palm center and area (4 features)
        # Palm landmarks: wrist(0), thumb_cmc(1), index_mcp(5), pinky_mcp(17)
        palm_points = coords[[0, 1, 5, 17]]
        palm_center = np.mean(palm_points, axis=0)
        features.extend(palm_center)

        # Approximate palm area using convex hull of palm points
        try:
            # Project to 2D for area calculation
            palm_2d = palm_points[:, :2]
            hull = cv2.convexHull(palm_2d.astype(np.float32))
            palm_area = cv2.contourArea(hull)
            features.append(palm_area)
        except:
            features.append(0.0)

        return np.array(features, dtype=np.float32)

    def visualize_landmarks(self, image: np.ndarray, landmarks: HandLandmarks) -> np.ndarray:
        """Draw hand landmarks on image for visualization."""
        if landmarks is None:
            return image

        annotated_image = image.copy()
        h, w = image.shape[:2]

        # Convert landmarks back to MediaPipe format for drawing
        mp_landmarks = self.mp_hands.HandLandmarks()
        for i, (x, y, z) in enumerate(landmarks.landmarks):
            landmark = mp_landmarks.landmark.add()
            landmark.x = x / w
            landmark.y = y / h
            landmark.z = z

        # Draw landmarks
        self.mp_drawing.draw_landmarks(
            annotated_image,
            mp_landmarks,
            self.mp_hands.HAND_CONNECTIONS
        )

        return annotated_image

    def get_feature_names(self) -> List[str]:
        """Get descriptive names for all extracted features."""
        names = []

        # Raw coordinates (63)
        landmarks_names = [
            "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
            "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
            "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
            "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
            "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
        ]

        for landmark in landmarks_names:
            names.extend([f"{landmark}_X", f"{landmark}_Y", f"{landmark}_Z"])

        # Fingertip positions (15)
        fingertips = ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]
        for finger in fingertips:
            names.extend([f"{finger}_TIP_X", f"{finger}_TIP_Y", f"{finger}_TIP_Z"])

        # Distances from wrist (5)
        for finger in fingertips:
            names.append(f"{finger}_DISTANCE_FROM_WRIST")

        # Inter-finger distances (10)
        for i in range(len(fingertips)):
            for j in range(i + 1, len(fingertips)):
                names.append(f"{fingertips[i]}_{fingertips[j]}_DISTANCE")

        # Hand orientation (3)
        names.extend(["ORIENTATION_X", "ORIENTATION_Y", "ORIENTATION_Z"])

        # Hand span (4)
        names.extend(["HAND_WIDTH", "HAND_LENGTH", "DIAGONAL_1", "DIAGONAL_2"])

        # Finger bend angles (5)
        for finger in fingertips:
            names.append(f"{finger}_BEND_ANGLE")

        # Palm features (4)
        names.extend(["PALM_CENTER_X", "PALM_CENTER_Y", "PALM_CENTER_Z", "PALM_AREA"])

        return names

    def cleanup(self):
        """Clean up MediaPipe resources."""
        if hasattr(self, 'hands'):
            self.hands.close()
        if hasattr(self, 'hands_video'):
            self.hands_video.close()

def main():
    """Test MediaPipe feature extraction."""
    extractor = MediaPipeFeatureExtractor()

    # Test with webcam
    cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Flip to match dataset format
            frame = cv2.flip(frame, 1)

            # Extract landmarks
            landmarks = extractor.extract_landmarks_from_frame(frame)

            if landmarks is not None:
                # Extract advanced features
                features = extractor.extract_advanced_features(landmarks)
                print(f"Extracted {len(features)} features")

                # Visualize
                annotated_frame = extractor.visualize_landmarks(frame, landmarks)
                cv2.imshow('MediaPipe Hand Landmarks', annotated_frame)
            else:
                cv2.imshow('MediaPipe Hand Landmarks', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.cleanup()

if __name__ == "__main__":
    main()