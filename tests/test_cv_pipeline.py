#!/usr/bin/env python3
"""
Tests for Computer Vision Pipeline
"""

import pytest
import numpy as np
import cv2
import torch
from pathlib import Path
import sys

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from cv_pipeline.camera_handler import CameraHandler
from cv_pipeline.inference_engine import GestureInferenceEngine
from cv_pipeline.utils import *

class TestCameraHandler:
    """Test camera handling functionality"""

    def test_camera_initialization(self):
        """Test camera initialization"""
        camera = CameraHandler(camera_index=-1)  # Use invalid index to avoid actual camera

        # Should handle invalid camera gracefully
        assert camera.cap is None
        assert not camera.is_running

    def test_frame_preprocessing(self):
        """Test frame preprocessing"""
        camera = CameraHandler()

        # Create dummy frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Test preprocessing
        processed = camera.preprocess_frame(frame, target_size=(224, 224))

        assert processed is not None
        assert processed.shape == (224, 224, 3)
        assert processed.dtype == np.float32
        assert np.all(processed >= 0) and np.all(processed <= 1)

    def test_hand_detection(self):
        """Test simple hand detection"""
        camera = CameraHandler()

        # Create frame with skin-colored region
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[200:300, 250:350] = [180, 120, 100]  # Skin-like color

        bbox = camera.detect_hand_region_simple(frame)

        # Should detect some region
        if bbox:
            x, y, w, h = bbox
            assert isinstance(x, int) and isinstance(y, int)
            assert isinstance(w, int) and isinstance(h, int)
            assert w > 0 and h > 0

    def test_statistics(self):
        """Test statistics collection"""
        camera = CameraHandler()
        stats = camera.get_statistics()

        assert isinstance(stats, dict)
        assert "frames_captured" in stats
        assert "frames_dropped" in stats
        assert "current_fps" in stats
        assert "is_running" in stats

class TestInferenceEngine:
    """Test inference engine functionality"""

    def test_frame_preprocessing(self):
        """Test inference preprocessing"""
        # Skip if no model available
        model_path = Path("backend/trained_models/efficient_best.pth")
        if not model_path.exists():
            pytest.skip("No trained model available")

        engine = GestureInferenceEngine(str(model_path))

        # Create dummy frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Test preprocessing
        tensor = engine.preprocess_frame(frame)

        if tensor is not None:
            assert tensor.shape == (1, 3, 224, 224)
            assert tensor.dtype == torch.float32

    def test_prediction_structure(self):
        """Test prediction result structure"""
        model_path = Path("backend/trained_models/efficient_best.pth")
        if not model_path.exists():
            pytest.skip("No trained model available")

        engine = GestureInferenceEngine(str(model_path))

        if engine.model is None:
            pytest.skip("Model loading failed")

        # Create dummy frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        result = engine.predict_gesture(frame)

        assert isinstance(result, dict)
        assert "gesture" in result
        assert "confidence" in result
        assert "status" in result
        assert "timestamp" in result

    def test_statistics(self):
        """Test inference statistics"""
        model_path = Path("backend/trained_models/efficient_best.pth")
        if not model_path.exists():
            pytest.skip("No trained model available")

        engine = GestureInferenceEngine(str(model_path))
        stats = engine.get_statistics()

        assert isinstance(stats, dict)
        assert "total_predictions" in stats
        assert "successful_predictions" in stats
        assert "confidence_threshold" in stats

class TestCVUtils:
    """Test computer vision utilities"""

    def test_resize_with_aspect_ratio(self):
        """Test aspect ratio preserving resize"""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Resize by width
        resized = resize_with_aspect_ratio(image, width=320)
        assert resized.shape[1] == 320
        assert resized.shape[0] == 240  # Maintains aspect ratio

        # Resize by height
        resized = resize_with_aspect_ratio(image, height=240)
        assert resized.shape[0] == 240
        assert resized.shape[1] == 320

    def test_center_crop(self):
        """Test center cropping"""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        cropped = center_crop(image, (224, 224))
        assert cropped.shape == (224, 224, 3)

    def test_pad_to_square(self):
        """Test padding to square"""
        # Rectangular image
        image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)

        padded = pad_to_square(image)
        assert padded.shape[0] == padded.shape[1]  # Square
        assert padded.shape[0] == 400  # Size of larger dimension

    def test_enhance_contrast(self):
        """Test contrast enhancement"""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        enhanced = enhance_contrast(image, alpha=1.5, beta=10)
        assert enhanced.shape == image.shape
        assert enhanced.dtype == np.uint8

    def test_skin_detection(self):
        """Test skin region detection"""
        # Create image with skin-colored regions
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[30:70, 30:70] = [180, 120, 100]  # Skin-like color

        skin_regions, mask = extract_skin_regions(image)

        assert skin_regions.shape == image.shape
        assert mask.shape == image.shape[:2]
        assert np.any(mask > 0)  # Should detect some skin

    def test_contour_processing(self):
        """Test contour finding and processing"""
        # Create binary mask with a shape
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(mask, (20, 20), (80, 80), 255, -1)

        contour = find_largest_contour(mask)

        if contour is not None:
            bbox = get_contour_bbox(contour, padding=5)
            x, y, w, h = bbox

            assert isinstance(x, int) and isinstance(y, int)
            assert isinstance(w, int) and isinstance(h, int)
            assert w > 0 and h > 0

    def test_frame_difference(self):
        """Test frame difference calculation"""
        frame1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        frame2 = frame1.copy()
        frame3 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Identical frames should have minimal difference
        diff1 = calculate_frame_difference(frame1, frame2)
        assert diff1 < 0.1

        # Different frames should have higher difference
        diff2 = calculate_frame_difference(frame1, frame3)
        assert diff2 > diff1

    def test_gesture_overlay(self):
        """Test gesture overlay drawing"""
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

        # Draw gesture overlay
        result = draw_gesture_overlay(image, "dog", 0.85)

        assert result.shape == image.shape
        assert result.dtype == np.uint8

        # Image should be modified (overlay added)
        assert not np.array_equal(result, image)

    def test_confidence_bar(self):
        """Test confidence bar drawing"""
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

        result = draw_confidence_bar(image, 0.75)

        assert result.shape == image.shape
        assert result.dtype == np.uint8

    def test_debug_view_creation(self):
        """Test debug view creation"""
        original = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        processed = np.random.rand(224, 224, 3).astype(np.float32)

        debug_view = create_gesture_debug_view(
            original, processed, "dog", 0.85,
            probabilities={"dog": 0.85, "cat": 0.15}
        )

        assert len(debug_view.shape) == 3  # Should be a color image
        assert debug_view.dtype == np.uint8

def test_integration():
    """Test integration between components"""
    # Test that components can work together
    camera = CameraHandler()

    # Create dummy frame
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Process frame
    processed = camera.preprocess_frame(frame)
    assert processed is not None

    # Apply utilities
    enhanced = enhance_contrast(processed)
    assert enhanced.shape == processed.shape

if __name__ == "__main__":
    pytest.main([__file__, "-v"])