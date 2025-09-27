#!/usr/bin/env python3
"""
Shadow-Vision Feature Extraction Utility (Utility #2)

OpenCV-based hand detection and feature extraction for shadow puppet recognition.
Extracts geometric features from hand contours for stable live demo performance.

This approach is MORE stable than MediaPipe for live demos because:
- No complex ML dependencies
- Faster processing
- More consistent results
- Better error handling

Features extracted:
- Hand contour geometry (area, perimeter, convexity)
- Finger detection and counting
- Hand orientation and shape ratios
- Bounding box properties
- Hu moments (rotation invariant)
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import json
from PIL import Image
import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class HandFeatures:
    """Container for extracted hand features."""
    # Basic geometry
    area: float
    perimeter: float
    convexity_ratio: float
    solidity: float

    # Shape properties
    aspect_ratio: float
    extent: float
    bounding_box_area: float

    # Finger detection
    finger_count: int
    convexity_defects: int

    # Orientation
    orientation_angle: float
    major_axis_length: float
    minor_axis_length: float

    # Hu moments (rotation invariant)
    hu_moments: List[float]

    # Position (normalized)
    centroid_x: float
    centroid_y: float

    def to_vector(self) -> np.ndarray:
        """Convert features to numpy vector for ML."""
        features = [
            self.area, self.perimeter, self.convexity_ratio, self.solidity,
            self.aspect_ratio, self.extent, self.bounding_box_area,
            self.finger_count, self.convexity_defects,
            self.orientation_angle, self.major_axis_length, self.minor_axis_length,
            self.centroid_x, self.centroid_y
        ]
        # Add Hu moments
        features.extend(self.hu_moments)
        return np.array(features, dtype=np.float32)


class HandDetector:
    """OpenCV-based hand detection optimized for shadow puppets."""

    def __init__(self):
        self.min_hand_area = 1000  # Minimum hand area in pixels
        self.max_hand_area = 50000  # Maximum hand area in pixels

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for hand detection."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply threshold to create binary image
        # For shadow puppets, we want to detect dark shapes on light background
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Invert if needed (hand should be white on black background)
        if np.mean(thresh) > 127:
            thresh = cv2.bitwise_not(thresh)

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        return thresh

    def find_hand_contour(self, binary_image: np.ndarray) -> Optional[np.ndarray]:
        """Find the main hand contour in the image."""
        # Find contours
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Filter contours by area
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_hand_area <= area <= self.max_hand_area:
                valid_contours.append(contour)

        if not valid_contours:
            return None

        # Return the largest valid contour
        return max(valid_contours, key=cv2.contourArea)

    def count_fingers(self, contour: np.ndarray) -> Tuple[int, int]:
        """Count fingers using convexity defects."""
        # Find convex hull
        hull = cv2.convexHull(contour, returnPoints=False)

        if len(hull) < 4:
            return 0, 0

        # Find convexity defects
        defects = cv2.convexityDefects(contour, hull)

        if defects is None:
            return 0, 0

        finger_count = 0
        defect_count = 0

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            # Calculate lengths of sides
            a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

            # Calculate angle using cosine rule
            if b != 0 and c != 0:
                angle = np.arccos((b**2 + c**2 - a**2) / (2*b*c))

                # If angle is less than 90 degrees and depth is significant
                if angle <= np.pi/2 and d > 10000:  # depth threshold
                    finger_count += 1
                    defect_count += 1

        return finger_count, defect_count


class FeatureExtractor:
    """Extract comprehensive hand features for ML classification."""

    def __init__(self):
        self.detector = HandDetector()
        self.logger = logging.getLogger(__name__)

    def extract_features(self, image_path: str) -> Optional[HandFeatures]:
        """Extract features from a single image."""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"Could not load image: {image_path}")
                return None

            # Preprocess
            binary = self.detector.preprocess_image(image)

            # Find hand contour
            contour = self.detector.find_hand_contour(binary)
            if contour is None:
                self.logger.warning(f"No hand detected in: {image_path}")
                return None

            # Extract basic geometry
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            # Convex hull properties
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            convexity_ratio = area / hull_area if hull_area > 0 else 0

            # Bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            bounding_box_area = w * h
            aspect_ratio = w / h if h > 0 else 0
            extent = area / bounding_box_area if bounding_box_area > 0 else 0

            # Solidity
            solidity = area / hull_area if hull_area > 0 else 0

            # Finger counting
            finger_count, defect_count = self.detector.count_fingers(contour)

            # Orientation using fitted ellipse
            if len(contour) >= 5:  # Need at least 5 points for ellipse
                ellipse = cv2.fitEllipse(contour)
                orientation_angle = ellipse[2]
                major_axis = max(ellipse[1])
                minor_axis = min(ellipse[1])
            else:
                orientation_angle = 0
                major_axis = 0
                minor_axis = 0

            # Centroid (normalized to image size)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                centroid_x = M["m10"] / M["m00"] / image.shape[1]  # Normalize by width
                centroid_y = M["m01"] / M["m00"] / image.shape[0]  # Normalize by height
            else:
                centroid_x = 0.5
                centroid_y = 0.5

            # Hu moments (rotation invariant)
            hu_moments = cv2.HuMoments(M).flatten()
            # Log transform for better numerical properties
            hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

            return HandFeatures(
                area=area,
                perimeter=perimeter,
                convexity_ratio=convexity_ratio,
                solidity=solidity,
                aspect_ratio=aspect_ratio,
                extent=extent,
                bounding_box_area=bounding_box_area,
                finger_count=finger_count,
                convexity_defects=defect_count,
                orientation_angle=orientation_angle,
                major_axis_length=major_axis,
                minor_axis_length=minor_axis,
                centroid_x=centroid_x,
                centroid_y=centroid_y,
                hu_moments=hu_moments.tolist()
            )

        except Exception as e:
            self.logger.error(f"Feature extraction failed for {image_path}: {e}")
            return None


def load_config(config_path="config/config.yaml"):
    """Load configuration from YAML file."""
    config_file = Path(__file__).parent.parent.parent / config_path
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def extract_features_from_dataset(config):
    """Extract features from the organized dataset."""
    print("Shadow-Vision Feature Extraction (Utility #2)")
    print("=" * 50)

    base_dir = Path(__file__).parent.parent.parent
    splits_dir = base_dir / config['dataset']['splits_dir']
    extractor = FeatureExtractor()

    # Process both splits
    for split_name in ['train', 'val']:
        print(f"\nProcessing {split_name} split...")

        # Load CSV
        csv_path = splits_dir / f"{split_name}.csv"
        if not csv_path.exists():
            print(f"ERROR: {csv_path} not found. Run dataset utility first.")
            continue

        df = pd.read_csv(csv_path)

        # Extract features
        features_list = []
        labels_list = []
        paths_list = []

        for idx, row in df.iterrows():
            image_path = base_dir / row['path']

            features = extractor.extract_features(str(image_path))
            if features is not None:
                features_list.append(features.to_vector())
                labels_list.append(row['label'])
                paths_list.append(row['path'])

            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(df)} images...")

        if features_list:
            # Convert to numpy arrays
            X = np.stack(features_list)
            y = np.array(labels_list)

            # Save features
            features_path = splits_dir / f"{split_name}_features.npz"
            np.savez(features_path, X=X, y=y, paths=paths_list)

            print(f"  Saved {len(X)} feature vectors to {features_path}")
            print(f"  Feature vector size: {X.shape[1]}")
            print(f"  Classes: {len(np.unique(y))}")
        else:
            print(f"  ERROR: No features extracted for {split_name} split!")

    print("\n=== Feature Extraction Summary ===")

    # Load and summarize features
    for split_name in ['train', 'val']:
        features_path = splits_dir / f"{split_name}_features.npz"
        if features_path.exists():
            data = np.load(features_path)
            X, y = data['X'], data['y']

            print(f"\n{split_name.upper()} features:")
            print(f"  Samples: {len(X)}")
            print(f"  Features: {X.shape[1]}")
            print(f"  Classes: {len(np.unique(y))}")

            # Class distribution
            unique, counts = np.unique(y, return_counts=True)
            for class_idx, count in zip(unique, counts):
                class_name = config['project']['classes'][class_idx]
                print(f"    {class_name}: {count}")


def main():
    """Main function to run feature extraction."""
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Load config
    config = load_config()

    # Extract features
    extract_features_from_dataset(config)

    print("\n🎉 Feature extraction completed!")
    print("\nNext: Utility #3 - Training (k-NN classifier)")

    return 0


if __name__ == "__main__":
    exit(main())