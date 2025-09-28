#!/usr/bin/env python3
"""
Enhanced Hand Detection Without MediaPipe

Since MediaPipe doesn't support Python 3.13, this implements
advanced hand detection using OpenCV with landmark-like features.
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class EnhancedHandFeatures:
    """Enhanced hand features that simulate MediaPipe-style landmarks."""
    # Fingertip positions (10 features: 5 tips × 2 coords)
    fingertips: np.ndarray

    # Hand center and orientation (3 features)
    palm_center: np.ndarray
    hand_orientation: float

    # Finger measurements (15 features)
    finger_lengths: np.ndarray      # 5 finger lengths
    finger_angles: np.ndarray       # 5 finger angles from palm
    inter_finger_distances: np.ndarray  # 5 distances between adjacent fingers

    # Hand geometry (10 features)
    hand_width: float
    hand_length: float
    palm_area: float
    convexity_ratio: float
    solidity: float
    aspect_ratio: float
    compactness: float
    eccentricity: float
    circularity: float
    rectangularity: float

    # Advanced shape analysis (12 features)
    hu_moments: np.ndarray          # 7 Hu moments
    fourier_descriptors: np.ndarray # 5 Fourier descriptors

    def to_vector(self) -> np.ndarray:
        """Convert all features to a single vector."""
        features = []
        features.extend(self.fingertips.flatten())        # 10
        features.extend(self.palm_center)                 # 2
        features.append(self.hand_orientation)            # 1
        features.extend(self.finger_lengths)              # 5
        features.extend(self.finger_angles)               # 5
        features.extend(self.inter_finger_distances)      # 5
        features.extend([
            self.hand_width, self.hand_length, self.palm_area,
            self.convexity_ratio, self.solidity, self.aspect_ratio,
            self.compactness, self.eccentricity, self.circularity,
            self.rectangularity
        ])                                                # 10
        features.extend(self.hu_moments)                  # 7
        features.extend(self.fourier_descriptors)         # 5

        return np.array(features, dtype=np.float32)       # Total: 50 features

class EnhancedHandDetector:
    """Advanced hand detection using enhanced OpenCV techniques."""

    def __init__(self):
        """Initialize enhanced hand detector."""
        self.min_hand_area = 1000
        self.max_hand_area = 100000

        # Background subtraction
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=False,
            varThreshold=50,
            history=500
        )

        # Skin detection parameters (YCrCb color space)
        self.skin_lower = np.array([0, 133, 77], dtype=np.uint8)
        self.skin_upper = np.array([255, 173, 127], dtype=np.uint8)

        logger.info("Enhanced hand detector initialized")

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Advanced preprocessing for hand detection."""
        # 1. Enhance contrast using CLAHE
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:,:,0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(lab[:,:,0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # 2. Skin color detection
        ycrcb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2YCrCb)
        skin_mask = cv2.inRange(ycrcb, self.skin_lower, self.skin_upper)

        # 3. Background subtraction
        fg_mask = self.bg_subtractor.apply(enhanced)

        # 4. Edge detection
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # 5. Combine all detection methods
        combined_mask = cv2.bitwise_or(skin_mask, fg_mask)
        combined_mask = cv2.bitwise_or(combined_mask, edges)

        # 6. Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        return enhanced, combined_mask

    def find_hand_contour(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """Find the best hand contour."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Filter contours by area and aspect ratio
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_hand_area < area < self.max_hand_area:
                # Check aspect ratio
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                if min(width, height) > 0:
                    aspect_ratio = max(width, height) / min(width, height)
                    if aspect_ratio < 3.0:  # Hands shouldn't be too elongated
                        valid_contours.append((contour, area))

        if not valid_contours:
            return None

        # Return largest valid contour
        return max(valid_contours, key=lambda x: x[1])[0]

    def detect_fingertips(self, contour: np.ndarray) -> np.ndarray:
        """Detect fingertip positions using convexity defects."""
        try:
            # Get convex hull
            hull = cv2.convexHull(contour, returnPoints=False)
            if len(hull) < 4:
                return np.zeros((5, 2))

            # Get convexity defects
            defects = cv2.convexityDefects(contour, hull)
            if defects is None:
                return np.zeros((5, 2))

            # Find fingertips (convex points)
            fingertips = []
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = contour[s][0]
                end = contour[e][0]
                far = contour[f][0]

                # Calculate angle at defect point
                a = np.linalg.norm(start - far)
                b = np.linalg.norm(end - far)
                c = np.linalg.norm(start - end)

                if b > 0 and c > 0:
                    angle = np.arccos((b**2 + c**2 - a**2) / (2*b*c))

                    # If angle is acute and depth is significant, it's between fingers
                    if angle <= np.pi/2 and d > 8000:
                        # The start and end points are potential fingertips
                        fingertips.extend([start, end])

            # Remove duplicates and sort by y-coordinate (top to bottom)
            if fingertips:
                fingertips = np.array(fingertips)
                unique_tips = []
                for tip in fingertips:
                    # Check if this tip is too close to existing ones
                    is_duplicate = False
                    for existing in unique_tips:
                        if np.linalg.norm(tip - existing) < 30:
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        unique_tips.append(tip)

                # Sort by y-coordinate and take top 5
                if unique_tips:
                    unique_tips = sorted(unique_tips, key=lambda x: x[1])[:5]

                    # Pad with zeros if we have fewer than 5
                    while len(unique_tips) < 5:
                        unique_tips.append([0, 0])

                    return np.array(unique_tips[:5])

            return np.zeros((5, 2))

        except Exception as e:
            logger.warning(f"Fingertip detection failed: {e}")
            return np.zeros((5, 2))

    def extract_enhanced_features(self, image: np.ndarray) -> Optional[EnhancedHandFeatures]:
        """Extract enhanced hand features."""
        try:
            # Preprocess image
            enhanced, mask = self.preprocess_image(image)

            # Find hand contour
            contour = self.find_hand_contour(mask)
            if contour is None:
                return None

            # Basic measurements
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            if area == 0 or perimeter == 0:
                return None

            # Detect fingertips
            fingertips = self.detect_fingertips(contour)

            # Calculate palm center
            M = cv2.moments(contour)
            if M["m00"] != 0:
                palm_center = np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]])
            else:
                palm_center = np.array([0.0, 0.0])

            # Hand orientation (angle of major axis)
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                hand_orientation = ellipse[2] * np.pi / 180.0  # Convert to radians
            else:
                hand_orientation = 0.0

            # Finger measurements
            finger_lengths = np.zeros(5)
            finger_angles = np.zeros(5)
            for i, tip in enumerate(fingertips):
                if tip[0] != 0 or tip[1] != 0:  # Valid fingertip
                    finger_lengths[i] = np.linalg.norm(tip - palm_center)
                    # Angle from palm center to fingertip
                    vector = tip - palm_center
                    finger_angles[i] = np.arctan2(vector[1], vector[0])

            # Inter-finger distances
            inter_finger_distances = np.zeros(5)
            for i in range(4):
                if (fingertips[i][0] != 0 or fingertips[i][1] != 0) and \
                   (fingertips[i+1][0] != 0 or fingertips[i+1][1] != 0):
                    inter_finger_distances[i] = np.linalg.norm(fingertips[i] - fingertips[i+1])

            # Hand geometry
            rect = cv2.minAreaRect(contour)
            width, height = rect[1]
            hand_width = max(width, height)
            hand_length = min(width, height)

            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)

            convexity_ratio = area / hull_area if hull_area > 0 else 0
            solidity = area / hull_area if hull_area > 0 else 0
            aspect_ratio = hand_width / hand_length if hand_length > 0 else 0
            compactness = (perimeter * perimeter) / area if area > 0 else 0

            # Fit ellipse for eccentricity
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                a, b = ellipse[1][0] / 2, ellipse[1][1] / 2  # Semi-major and semi-minor axes
                if a > 0:
                    eccentricity = np.sqrt(1 - (min(a, b) / max(a, b))**2) if max(a, b) > 0 else 0
                else:
                    eccentricity = 0
            else:
                eccentricity = 0

            circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0

            # Bounding rectangle for rectangularity
            x, y, w, h = cv2.boundingRect(contour)
            rect_area = w * h
            rectangularity = area / rect_area if rect_area > 0 else 0

            # Hu moments
            moments = cv2.moments(contour)
            hu_moments = cv2.HuMoments(moments).flatten()
            # Log transform to handle large values
            hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

            # Fourier descriptors (simplified)
            # Sample points from contour
            if len(contour) > 10:
                # Resample contour to fixed number of points
                contour_points = contour.reshape(-1, 2)
                indices = np.linspace(0, len(contour_points)-1, 64, dtype=int)
                sampled_points = contour_points[indices]

                # Convert to complex numbers
                complex_contour = sampled_points[:, 0] + 1j * sampled_points[:, 1]

                # FFT
                fft_coeffs = np.fft.fft(complex_contour)

                # Take magnitude of first 5 coefficients (excluding DC)
                fourier_descriptors = np.abs(fft_coeffs[1:6])
            else:
                fourier_descriptors = np.zeros(5)

            return EnhancedHandFeatures(
                fingertips=fingertips,
                palm_center=palm_center,
                hand_orientation=hand_orientation,
                finger_lengths=finger_lengths,
                finger_angles=finger_angles,
                inter_finger_distances=inter_finger_distances,
                hand_width=hand_width,
                hand_length=hand_length,
                palm_area=area,
                convexity_ratio=convexity_ratio,
                solidity=solidity,
                aspect_ratio=aspect_ratio,
                compactness=compactness,
                eccentricity=eccentricity,
                circularity=circularity,
                rectangularity=rectangularity,
                hu_moments=hu_moments,
                fourier_descriptors=fourier_descriptors
            )

        except Exception as e:
            logger.error(f"Enhanced feature extraction failed: {e}")
            return None

def main():
    """Test enhanced hand detection."""
    detector = EnhancedHandDetector()
    cap = cv2.VideoCapture(0)

    try:
        print("Enhanced Hand Detection Test")
        print("Press 'q' to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Flip to match dataset format
            frame = cv2.flip(frame, 1)

            # Extract enhanced features
            features = detector.extract_enhanced_features(frame)

            if features is not None:
                # Draw fingertips
                for i, tip in enumerate(features.fingertips):
                    if tip[0] != 0 or tip[1] != 0:
                        cv2.circle(frame, tuple(tip.astype(int)), 8, (0, 255, 0), -1)
                        cv2.putText(frame, str(i), tuple(tip.astype(int)),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Draw palm center
                palm = features.palm_center.astype(int)
                cv2.circle(frame, tuple(palm), 10, (255, 0, 0), -1)

                # Show feature count
                feature_vector = features.to_vector()
                cv2.putText(frame, f"Features: {len(feature_vector)}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.putText(frame, f"Hand Size: {features.hand_width:.0f}x{features.hand_length:.0f}",
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow('Enhanced Hand Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()