#!/usr/bin/env python3
"""
Computer Vision Utilities
Helper functions for image processing and frame manipulation
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
import math

def resize_with_aspect_ratio(image: np.ndarray, width: int = None, height: int = None,
                           inter=cv2.INTER_AREA) -> np.ndarray:
    """Resize image while maintaining aspect ratio"""
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        # Calculate width based on height
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        # Calculate height based on width
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

def center_crop(image: np.ndarray, crop_size: Tuple[int, int]) -> np.ndarray:
    """Center crop image to specified size"""
    h, w = image.shape[:2]
    crop_h, crop_w = crop_size

    # Calculate crop coordinates
    start_y = max(0, (h - crop_h) // 2)
    start_x = max(0, (w - crop_w) // 2)
    end_y = min(h, start_y + crop_h)
    end_x = min(w, start_x + crop_w)

    return image[start_y:end_y, start_x:end_x]

def pad_to_square(image: np.ndarray, pad_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """Pad image to make it square"""
    h, w = image.shape[:2]

    if h == w:
        return image

    # Determine padding needed
    if h > w:
        # Pad width
        pad_width = (h - w) // 2
        pad_left = pad_width
        pad_right = h - w - pad_left
        pad_top = pad_bottom = 0
    else:
        # Pad height
        pad_height = (w - h) // 2
        pad_top = pad_height
        pad_bottom = w - h - pad_top
        pad_left = pad_right = 0

    return cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right,
                             cv2.BORDER_CONSTANT, value=pad_color)

def enhance_contrast(image: np.ndarray, alpha: float = 1.5, beta: int = 10) -> np.ndarray:
    """Enhance image contrast and brightness"""
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply Gaussian blur to image"""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def detect_edges(image: np.ndarray, low_threshold: int = 50,
                high_threshold: int = 150) -> np.ndarray:
    """Detect edges using Canny edge detection"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return cv2.Canny(gray, low_threshold, high_threshold)

def remove_background_simple(image: np.ndarray, threshold: int = 50) -> np.ndarray:
    """Simple background removal using thresholding"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Create mask
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Apply mask
    result = image.copy()
    result[mask == 0] = [0, 0, 0]  # Set background to black

    return result

def extract_skin_regions(image: np.ndarray) -> np.ndarray:
    """Extract skin-colored regions from image"""
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define skin color range in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create mask
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Morphological operations to clean up
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Apply mask
    result = cv2.bitwise_and(image, image, mask=mask)

    return result, mask

def find_largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    """Find the largest contour in a binary mask"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    return max(contours, key=cv2.contourArea)

def get_contour_bbox(contour: np.ndarray, padding: int = 10) -> Tuple[int, int, int, int]:
    """Get bounding box of contour with padding"""
    x, y, w, h = cv2.boundingRect(contour)

    return (
        max(0, x - padding),
        max(0, y - padding),
        w + 2 * padding,
        h + 2 * padding
    )

def draw_gesture_overlay(image: np.ndarray, gesture: str, confidence: float,
                        position: Tuple[int, int] = (10, 30),
                        font_scale: float = 1.0, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """Draw gesture prediction overlay on image"""
    result = image.copy()

    # Prepare text
    if gesture:
        text = f"{gesture}: {confidence:.2f}"
    else:
        text = "No gesture detected"

    # Draw text background
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, 2)[0]

    # Background rectangle
    cv2.rectangle(result,
                 (position[0] - 5, position[1] - text_size[1] - 10),
                 (position[0] + text_size[0] + 5, position[1] + 5),
                 (0, 0, 0), -1)

    # Draw text
    cv2.putText(result, text, position, font, font_scale, color, 2)

    return result

def draw_confidence_bar(image: np.ndarray, confidence: float,
                       position: Tuple[int, int] = (10, 60),
                       size: Tuple[int, int] = (200, 20)) -> np.ndarray:
    """Draw confidence level as a progress bar"""
    result = image.copy()
    x, y = position
    w, h = size

    # Background bar
    cv2.rectangle(result, (x, y), (x + w, y + h), (50, 50, 50), -1)

    # Confidence bar
    confidence_width = int(w * confidence)
    color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255) if confidence > 0.5 else (0, 0, 255)
    cv2.rectangle(result, (x, y), (x + confidence_width, y + h), color, -1)

    # Border
    cv2.rectangle(result, (x, y), (x + w, y + h), (255, 255, 255), 1)

    # Text
    cv2.putText(result, f"{confidence:.1%}", (x + w + 10, y + h - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return result

def create_gesture_debug_view(image: np.ndarray, processed_image: np.ndarray,
                             gesture: str, confidence: float,
                             probabilities: dict = None) -> np.ndarray:
    """Create a debug view showing original, processed, and predictions"""

    # Resize images to same height
    target_height = 240
    img1 = resize_with_aspect_ratio(image, height=target_height)
    img2 = resize_with_aspect_ratio(processed_image, height=target_height)

    # Convert processed image if needed
    if img2.dtype == np.float32:
        img2 = (img2 * 255).astype(np.uint8)
    if len(img2.shape) == 3 and img2.shape[2] == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

    # Create side-by-side view
    combined = np.hstack([img1, img2])

    # Add gesture overlay
    combined = draw_gesture_overlay(combined, gesture, confidence)
    combined = draw_confidence_bar(combined, confidence)

    # Add probabilities if provided
    if probabilities:
        y_offset = 100
        for i, (gesture_name, prob) in enumerate(probabilities.items()):
            text = f"{gesture_name}: {prob:.3f}"
            cv2.putText(combined, text, (10, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return combined

def apply_motion_blur(image: np.ndarray, kernel_size: int = 15, angle: float = 0) -> np.ndarray:
    """Apply motion blur to simulate hand movement"""
    # Create motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size))

    # Calculate kernel based on angle
    center = kernel_size // 2
    cos_val = math.cos(math.radians(angle))
    sin_val = math.sin(math.radians(angle))

    for i in range(kernel_size):
        offset = i - center
        x = int(center + offset * cos_val)
        y = int(center + offset * sin_val)

        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1

    # Normalize kernel
    kernel = kernel / np.sum(kernel)

    # Apply convolution
    return cv2.filter2D(image, -1, kernel)

def add_noise(image: np.ndarray, noise_type: str = "gaussian", intensity: float = 0.1) -> np.ndarray:
    """Add noise to image for data augmentation"""
    result = image.copy().astype(np.float32)

    if noise_type == "gaussian":
        noise = np.random.normal(0, intensity * 255, image.shape)
        result += noise
    elif noise_type == "salt_pepper":
        noise = np.random.random(image.shape[:2])
        result[noise < intensity/2] = 0
        result[noise > 1 - intensity/2] = 255

    return np.clip(result, 0, 255).astype(np.uint8)

def calculate_frame_difference(frame1: np.ndarray, frame2: np.ndarray, threshold: int = 30) -> float:
    """Calculate difference between two frames"""
    if frame1.shape != frame2.shape:
        return 1.0

    # Convert to grayscale if needed
    if len(frame1.shape) == 3:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    else:
        gray1, gray2 = frame1, frame2

    # Calculate absolute difference
    diff = cv2.absdiff(gray1, gray2)

    # Threshold and count changed pixels
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    changed_pixels = np.sum(thresh > 0)
    total_pixels = thresh.size

    return changed_pixels / total_pixels

def main():
    """Test utility functions"""
    print("Testing CV utilities...")

    # Create test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Test resize
    resized = resize_with_aspect_ratio(test_image, width=320)
    print(f"Original: {test_image.shape}, Resized: {resized.shape}")

    # Test padding
    padded = pad_to_square(test_image)
    print(f"Padded to square: {padded.shape}")

    # Test skin detection
    skin_regions, mask = extract_skin_regions(test_image)
    print(f"Skin mask non-zero pixels: {np.sum(mask > 0)}")

    print("All tests completed successfully!")

if __name__ == "__main__":
    main()