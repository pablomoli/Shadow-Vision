#!/usr/bin/env python3
"""
Advanced Feature Extractor for Real-World Shadow Puppet Recognition

Major improvements for live demo robustness:
1. Advanced hand detection with background subtraction
2. Multiple feature types: geometric, texture, edge, gradient
3. Data augmentation for background robustness
4. Real-world preprocessing (skin detection, motion, etc.)
5. Robust to lighting and background variations
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import json
import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
import skimage.feature as skfeat
import skimage.segmentation as skseg
from scipy import ndimage


@dataclass
class AdvancedHandFeatures:
    """Container for comprehensive hand features."""
    # Basic geometry (improved)
    area: float
    perimeter: float
    convexity_ratio: float
    solidity: float
    aspect_ratio: float
    extent: float

    # Advanced shape features
    eccentricity: float
    compactness: float
    circularity: float
    rectangularity: float

    # Finger/contour analysis
    finger_count: int
    convexity_defects: int
    contour_energy: float
    bending_energy: float

    # Orientation and position
    orientation_angle: float
    centroid_x: float
    centroid_y: float

    # Hu moments (rotation invariant)
    hu_moments: List[float]

    # NEW: Texture features
    lbp_uniformity: float
    lbp_contrast: float
    glcm_contrast: float
    glcm_homogeneity: float

    # NEW: Edge features
    edge_density: float
    edge_orientation_hist: List[float]  # 8 bins

    # NEW: Gradient features
    gradient_magnitude_mean: float
    gradient_magnitude_std: float

    # NEW: Fourier descriptors
    fourier_descriptors: List[float]  # 10 descriptors

    def to_vector(self) -> np.ndarray:
        """Convert features to numpy vector for ML."""
        features = [
            # Basic geometry (6)
            self.area, self.perimeter, self.convexity_ratio, self.solidity,
            self.aspect_ratio, self.extent,

            # Advanced shape (4)
            self.eccentricity, self.compactness, self.circularity, self.rectangularity,

            # Finger/contour (4)
            self.finger_count, self.convexity_defects, self.contour_energy, self.bending_energy,

            # Orientation (3)
            self.orientation_angle, self.centroid_x, self.centroid_y,

            # Texture (4)
            self.lbp_uniformity, self.lbp_contrast, self.glcm_contrast, self.glcm_homogeneity,

            # Edge (2)
            self.edge_density, self.gradient_magnitude_mean, self.gradient_magnitude_std,
        ]

        # Add variable-length features
        features.extend(self.hu_moments)           # 7 features
        features.extend(self.edge_orientation_hist) # 8 features
        features.extend(self.fourier_descriptors)   # 10 features

        return np.array(features, dtype=np.float32)


class AdvancedHandDetector:
    """Advanced hand detection for real-world backgrounds."""

    def __init__(self):
        self.min_hand_area = 800   # Smaller minimum for distant hands
        self.max_hand_area = 80000 # Larger maximum for close hands

        # Background subtraction
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=False, varThreshold=50
        )

        # Skin color detection (YCrCb color space)
        self.skin_lower = np.array([0, 133, 77], dtype=np.uint8)
        self.skin_upper = np.array([255, 173, 127], dtype=np.uint8)

        self.logger = logging.getLogger(__name__)

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Advanced preprocessing for real-world conditions."""
        original = image.copy()

        # 1. Enhance contrast
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:,:,0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(lab[:,:,0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # 2. Skin color detection
        ycrcb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2YCrCb)
        skin_mask = cv2.inRange(ycrcb, self.skin_lower, self.skin_upper)

        # 3. Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

        # 4. Background subtraction (if we have background model)
        fg_mask = self.bg_subtractor.apply(enhanced)

        # 5. Combine masks
        combined_mask = cv2.bitwise_or(skin_mask, fg_mask)

        # 6. Edge detection for additional hand boundary information
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # 7. Combine with edge information
        final_mask = cv2.bitwise_or(combined_mask, edges)

        # 8. Final cleanup
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

        return final_mask, gray

    def find_hand_contour(self, binary_image: np.ndarray, gray_image: np.ndarray) -> Optional[np.ndarray]:
        """Find the main hand contour with advanced filtering."""
        # Find contours
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Advanced contour filtering
        valid_contours = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # Area filter
            if not (self.min_hand_area <= area <= self.max_hand_area):
                continue

            # Aspect ratio filter (hands are not extremely elongated)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            if not (0.3 <= aspect_ratio <= 3.0):
                continue

            # Extent filter (hand should fill reasonable portion of bounding box)
            extent = area / (w * h)
            if extent < 0.3:
                continue

            # Solidity filter (hand shouldn't be too concave)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            if solidity < 0.6:
                continue

            valid_contours.append((contour, area))

        if not valid_contours:
            return None

        # Return the largest valid contour
        return max(valid_contours, key=lambda x: x[1])[0]

    def count_fingers_advanced(self, contour: np.ndarray) -> Tuple[int, int, float]:
        """Advanced finger counting with multiple methods."""
        try:
            # Method 1: Convexity defects
            hull_indices = cv2.convexHull(contour, returnPoints=False)
            if len(hull_indices) < 4:
                return 0, 0, 0.0

            # Handle self-intersecting contours
            defects = cv2.convexityDefects(contour, hull_indices)
            if defects is None:
                return 0, 0, 0.0
        except cv2.error as e:
            # Handle convexity defects errors (self-intersecting contours)
            self.logger.debug(f"Convexity defects failed: {e}")
            return 0, 0, 0.0

        finger_count = 0
        defect_count = 0
        total_defect_depth = 0

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            # Calculate angle between vectors
            a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

            if b != 0 and c != 0:
                angle = np.arccos((b**2 + c**2 - a**2) / (2*b*c))

                # More sophisticated finger detection
                if angle <= np.pi/2 and d > 8000:  # Adjusted threshold
                    finger_count += 1
                    defect_count += 1
                    total_defect_depth += d

        avg_defect_depth = total_defect_depth / max(1, defect_count)

        return finger_count, defect_count, avg_defect_depth


class AdvancedFeatureExtractor:
    """Extract comprehensive features for robust classification."""

    def __init__(self):
        self.detector = AdvancedHandDetector()
        self.logger = logging.getLogger(__name__)

    def extract_texture_features(self, roi: np.ndarray) -> Tuple[float, float, float, float]:
        """Extract Local Binary Pattern and GLCM texture features."""
        if roi.size == 0:
            return 0.0, 0.0, 0.0, 0.0

        # Resize ROI for consistent texture analysis
        roi_resized = cv2.resize(roi, (64, 64))

        # Local Binary Pattern
        lbp = skfeat.local_binary_pattern(roi_resized, P=8, R=1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 9))
        lbp_hist = lbp_hist.astype(float)
        lbp_uniformity = np.sum(lbp_hist**2) / (np.sum(lbp_hist)**2) if np.sum(lbp_hist) > 0 else 0
        lbp_contrast = np.var(lbp_hist)

        # GLCM features
        try:
            # Quantize to reduce computation
            roi_quantized = (roi_resized / 32).astype(np.uint8)
            glcm = skfeat.greycomatrix(roi_quantized, distances=[1], angles=[0], levels=8)
            glcm_contrast = skfeat.greycoprops(glcm, 'contrast')[0, 0]
            glcm_homogeneity = skfeat.greycoprops(glcm, 'homogeneity')[0, 0]
        except:
            glcm_contrast = 0.0
            glcm_homogeneity = 0.0

        return lbp_uniformity, lbp_contrast, glcm_contrast, glcm_homogeneity

    def extract_edge_features(self, roi: np.ndarray) -> Tuple[float, List[float]]:
        """Extract edge density and orientation histogram."""
        if roi.size == 0:
            return 0.0, [0.0] * 8

        # Edge detection
        edges = cv2.Canny(roi, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        # Edge orientation histogram
        sobelx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)

        # Compute angles
        angles = np.arctan2(sobely, sobelx)
        angles = np.degrees(angles) % 180  # 0-180 degrees

        # Histogram of orientations (8 bins of 22.5 degrees each)
        hist, _ = np.histogram(angles, bins=8, range=(0, 180))
        hist = hist.astype(float)
        hist = hist / (np.sum(hist) + 1e-10)  # Normalize

        return edge_density, hist.tolist()

    def extract_gradient_features(self, roi: np.ndarray) -> Tuple[float, float]:
        """Extract gradient magnitude statistics."""
        if roi.size == 0:
            return 0.0, 0.0

        # Gradient magnitude
        sobelx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)

        return float(np.mean(magnitude)), float(np.std(magnitude))

    def extract_fourier_descriptors(self, contour: np.ndarray, num_descriptors: int = 10) -> List[float]:
        """Extract Fourier descriptors for shape analysis."""
        if len(contour) < 4:
            return [0.0] * num_descriptors

        # Convert contour to complex representation
        contour_complex = contour[:, 0, 0] + 1j * contour[:, 0, 1]

        # Apply FFT
        fourier_result = np.fft.fft(contour_complex)

        # Take magnitude and normalize
        fourier_descriptors = np.abs(fourier_result)

        # Normalize by first descriptor (for translation invariance)
        if fourier_descriptors[0] != 0:
            fourier_descriptors = fourier_descriptors / fourier_descriptors[0]

        # Take first num_descriptors (excluding DC component)
        descriptors = fourier_descriptors[1:num_descriptors+1]

        # Pad if necessary
        if len(descriptors) < num_descriptors:
            descriptors = np.pad(descriptors, (0, num_descriptors - len(descriptors)))

        return descriptors.real.tolist()  # Take real part only

    def extract_features(self, image_path: str) -> Optional[AdvancedHandFeatures]:
        """Extract comprehensive features from image."""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"Could not load image: {image_path}")
                return None

            # Advanced preprocessing
            binary_mask, gray_image = self.detector.preprocess_image(image)

            # Find hand contour
            contour = self.detector.find_hand_contour(binary_mask, gray_image)
            if contour is None:
                return None

            # Basic geometry
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            # Convex hull properties
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            convexity_ratio = area / hull_area if hull_area > 0 else 0
            solidity = area / hull_area if hull_area > 0 else 0

            # Bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            extent = area / (w * h) if (w * h) > 0 else 0

            # Advanced shape features
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                major_axis = max(ellipse[1])
                minor_axis = min(ellipse[1])
                eccentricity = np.sqrt(1 - (minor_axis/major_axis)**2) if major_axis > 0 else 0
                orientation_angle = ellipse[2]
            else:
                eccentricity = 0
                orientation_angle = 0
                major_axis = 0
                minor_axis = 0

            compactness = (perimeter**2) / (4 * np.pi * area) if area > 0 else 0
            circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
            rectangularity = area / (w * h) if (w * h) > 0 else 0

            # Advanced finger counting
            finger_count, defect_count, avg_defect_depth = self.detector.count_fingers_advanced(contour)

            # Contour energy features
            contour_energy = np.sum(np.diff(contour.reshape(-1, 2), axis=0)**2)
            bending_energy = 0.0  # Simplified for now

            # Centroid (normalized)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                centroid_x = M["m10"] / M["m00"] / image.shape[1]
                centroid_y = M["m01"] / M["m00"] / image.shape[0]
            else:
                centroid_x = 0.5
                centroid_y = 0.5

            # Hu moments
            hu_moments = cv2.HuMoments(M).flatten()
            hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

            # Extract ROI for texture/edge analysis
            mask = np.zeros(gray_image.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [contour], 255)
            roi = cv2.bitwise_and(gray_image, mask)

            # Texture features
            lbp_uniformity, lbp_contrast, glcm_contrast, glcm_homogeneity = self.extract_texture_features(roi)

            # Edge features
            edge_density, edge_orientation_hist = self.extract_edge_features(roi)

            # Gradient features
            grad_mean, grad_std = self.extract_gradient_features(roi)

            # Fourier descriptors
            fourier_descriptors = self.extract_fourier_descriptors(contour)

            return AdvancedHandFeatures(
                # Basic geometry
                area=area,
                perimeter=perimeter,
                convexity_ratio=convexity_ratio,
                solidity=solidity,
                aspect_ratio=aspect_ratio,
                extent=extent,

                # Advanced shape
                eccentricity=eccentricity,
                compactness=compactness,
                circularity=circularity,
                rectangularity=rectangularity,

                # Finger/contour
                finger_count=finger_count,
                convexity_defects=defect_count,
                contour_energy=contour_energy,
                bending_energy=bending_energy,

                # Orientation
                orientation_angle=orientation_angle,
                centroid_x=centroid_x,
                centroid_y=centroid_y,

                # Hu moments
                hu_moments=hu_moments.tolist(),

                # Texture
                lbp_uniformity=lbp_uniformity,
                lbp_contrast=lbp_contrast,
                glcm_contrast=glcm_contrast,
                glcm_homogeneity=glcm_homogeneity,

                # Edge
                edge_density=edge_density,
                edge_orientation_hist=edge_orientation_hist,

                # Gradient
                gradient_magnitude_mean=grad_mean,
                gradient_magnitude_std=grad_std,

                # Fourier
                fourier_descriptors=fourier_descriptors
            )

        except Exception as e:
            self.logger.error(f"Feature extraction failed for {image_path}: {e}")
            return None


def load_config(config_path="config/config.yaml"):
    """Load configuration from YAML file."""
    config_file = Path(__file__).parent.parent.parent / config_path
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def extract_advanced_features_from_dataset(config):
    """Extract advanced features from the organized dataset."""
    print("Advanced Shadow-Vision Feature Extraction")
    print("=" * 50)
    print("Improvements:")
    print("- Advanced hand detection for complex backgrounds")
    print("- Comprehensive feature set (50+ features)")
    print("- Texture, edge, and gradient analysis")
    print("- Fourier descriptors for shape analysis")
    print()

    base_dir = Path(__file__).parent.parent.parent
    splits_dir = base_dir / config['dataset']['splits_dir']
    extractor = AdvancedFeatureExtractor()

    # Process both splits
    for split_name in ['train', 'val']:
        print(f"Processing {split_name} split with advanced features...")

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
            features_path = splits_dir / f"{split_name}_advanced_features.npz"
            np.savez(features_path, X=X, y=y, paths=paths_list)

            print(f"  Saved {len(X)} advanced feature vectors to {features_path}")
            print(f"  Feature vector size: {X.shape[1]} (vs 21 in basic version)")
            print(f"  Classes: {len(np.unique(y))}")

            # Show class distribution
            unique, counts = np.unique(y, return_counts=True)
            for class_idx, count in zip(unique, counts):
                class_name = config['project']['classes'][class_idx]
                print(f"    {class_name}: {count}")
        else:
            print(f"  ERROR: No features extracted for {split_name} split!")

    print()
    print("Advanced feature extraction completed!")
    print("Next: Retrain model with comprehensive feature set")


def main():
    """Main function to run advanced feature extraction."""
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Load config
    config = load_config()

    # Extract features
    extract_advanced_features_from_dataset(config)

    return 0


if __name__ == "__main__":
    exit(main())