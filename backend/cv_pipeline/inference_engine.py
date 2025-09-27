#!/usr/bin/env python3
"""
Real-time Gesture Inference Engine
Handles model loading and real-time gesture prediction
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import time
import json
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import logging
from collections import deque

from models.gesture_classifier import load_model_from_config
from data.preprocess_data import DataPreprocessor

class GestureInferenceEngine:
    """Real-time gesture recognition inference engine"""

    def __init__(self, model_path: str, config_path: str = "config/model_config.yaml",
                 confidence_threshold: float = 0.8, smoothing_window: int = 5):

        self.model_path = model_path
        self.config_path = config_path
        self.confidence_threshold = confidence_threshold
        self.smoothing_window = smoothing_window

        # Model and preprocessing
        self.model = None
        self.preprocessor = DataPreprocessor()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Gesture mapping
        self.idx_to_gesture = {}
        self.gesture_to_idx = {}

        # Prediction smoothing
        self.prediction_history = deque(maxlen=smoothing_window)
        self.confidence_history = deque(maxlen=smoothing_window)

        # Statistics
        self.total_predictions = 0
        self.successful_predictions = 0
        self.inference_times = deque(maxlen=100)

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Load model
        self.load_model()

    def load_model(self) -> bool:
        """Load trained model from checkpoint"""
        try:
            model_path = Path(self.model_path)
            if not model_path.exists():
                self.logger.error(f"Model file not found: {model_path}")
                return False

            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)

            # Get model type and configuration
            model_type = checkpoint.get('model_type', 'efficient')
            self.idx_to_gesture = checkpoint.get('idx_to_gesture', {})
            self.gesture_to_idx = {v: k for k, v in self.idx_to_gesture.items()}

            # Create model
            self.model = load_model_from_config(self.config_path, model_type)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()

            self.logger.info(f"Model loaded successfully: {model_type}")
            self.logger.info(f"Gesture classes: {list(self.idx_to_gesture.values())}")
            self.logger.info(f"Using device: {self.device}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False

    def preprocess_frame(self, frame: np.ndarray) -> Optional[torch.Tensor]:
        """Preprocess frame for model inference"""
        try:
            if frame is None:
                return None

            # Resize and normalize
            frame_resized = cv2.resize(frame, (224, 224))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frame_normalized = frame_rgb.astype(np.float32) / 255.0

            # Convert to tensor
            tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1)

            # Normalize with ImageNet stats
            normalize = torch.nn.functional.normalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

            tensor = (tensor - mean) / std

            # Add batch dimension
            return tensor.unsqueeze(0).to(self.device)

        except Exception as e:
            self.logger.error(f"Frame preprocessing failed: {e}")
            return None

    def predict_gesture(self, frame: np.ndarray) -> Dict:
        """Predict gesture from frame"""
        start_time = time.time()

        try:
            # Preprocess frame
            input_tensor = self.preprocess_frame(frame)
            if input_tensor is None:
                return self._create_prediction_result(None, 0.0, "preprocessing_failed")

            # Model inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)

                predicted_idx = predicted_idx.item()
                confidence = confidence.item()

            # Get gesture name
            gesture_name = self.idx_to_gesture.get(predicted_idx, "unknown")

            # Update statistics
            self.total_predictions += 1
            inference_time = (time.time() - start_time) * 1000  # ms
            self.inference_times.append(inference_time)

            # Apply confidence threshold
            if confidence < self.confidence_threshold:
                return self._create_prediction_result(None, confidence, "low_confidence")

            # Apply smoothing
            smoothed_gesture, smoothed_confidence = self._apply_smoothing(gesture_name, confidence)

            if smoothed_gesture:
                self.successful_predictions += 1

            return self._create_prediction_result(
                smoothed_gesture, smoothed_confidence, "success",
                raw_gesture=gesture_name, raw_confidence=confidence,
                inference_time=inference_time
            )

        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return self._create_prediction_result(None, 0.0, "inference_failed")

    def _apply_smoothing(self, gesture: str, confidence: float) -> Tuple[Optional[str], float]:
        """Apply temporal smoothing to predictions"""
        self.prediction_history.append(gesture)
        self.confidence_history.append(confidence)

        if len(self.prediction_history) < self.smoothing_window:
            return None, confidence

        # Count occurrences of each gesture in history
        gesture_counts = {}
        for g in self.prediction_history:
            gesture_counts[g] = gesture_counts.get(g, 0) + 1

        # Find most frequent gesture
        most_frequent = max(gesture_counts.items(), key=lambda x: x[1])
        most_frequent_gesture, count = most_frequent

        # Require majority consensus
        if count >= self.smoothing_window // 2 + 1:
            avg_confidence = np.mean(list(self.confidence_history))
            return most_frequent_gesture, avg_confidence

        return None, confidence

    def _create_prediction_result(self, gesture: Optional[str], confidence: float,
                                status: str, **kwargs) -> Dict:
        """Create standardized prediction result"""
        result = {
            "gesture": gesture,
            "confidence": confidence,
            "status": status,
            "timestamp": time.time(),
            **kwargs
        }
        return result

    def get_gesture_probabilities(self, frame: np.ndarray) -> Dict[str, float]:
        """Get probabilities for all gesture classes"""
        try:
            input_tensor = self.preprocess_frame(frame)
            if input_tensor is None:
                return {}

            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1).squeeze().cpu().numpy()

            # Map to gesture names
            gesture_probs = {}
            for idx, prob in enumerate(probabilities):
                gesture_name = self.idx_to_gesture.get(idx, f"class_{idx}")
                gesture_probs[gesture_name] = float(prob)

            return gesture_probs

        except Exception as e:
            self.logger.error(f"Failed to get probabilities: {e}")
            return {}

    def reset_smoothing(self):
        """Reset prediction smoothing history"""
        self.prediction_history.clear()
        self.confidence_history.clear()

    def get_statistics(self) -> Dict:
        """Get inference engine statistics"""
        success_rate = (self.successful_predictions / max(1, self.total_predictions)) * 100
        avg_inference_time = np.mean(list(self.inference_times)) if self.inference_times else 0

        return {
            "total_predictions": self.total_predictions,
            "successful_predictions": self.successful_predictions,
            "success_rate": success_rate,
            "avg_inference_time_ms": avg_inference_time,
            "confidence_threshold": self.confidence_threshold,
            "smoothing_window": self.smoothing_window,
            "gesture_classes": list(self.idx_to_gesture.values()),
            "device": str(self.device)
        }

    def update_confidence_threshold(self, new_threshold: float):
        """Update confidence threshold"""
        self.confidence_threshold = max(0.0, min(1.0, new_threshold))
        self.logger.info(f"Confidence threshold updated to {self.confidence_threshold}")

    def benchmark_inference(self, num_frames: int = 100) -> Dict:
        """Benchmark inference performance"""
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        start_time = time.time()
        times = []

        for _ in range(num_frames):
            frame_start = time.time()
            _ = self.predict_gesture(dummy_frame)
            frame_time = (time.time() - frame_start) * 1000
            times.append(frame_time)

        total_time = (time.time() - start_time) * 1000

        return {
            "total_time_ms": total_time,
            "frames": num_frames,
            "avg_time_per_frame_ms": np.mean(times),
            "min_time_ms": np.min(times),
            "max_time_ms": np.max(times),
            "fps": 1000 / np.mean(times)
        }

def test_inference_engine():
    """Test the inference engine"""
    print("Testing gesture inference engine...")

    # Check for trained model
    model_path = "backend/trained_models/efficient_best.pth"
    if not Path(model_path).exists():
        print(f"Trained model not found at {model_path}")
        print("Please train a model first using: python backend/train_model.py")
        return

    # Create inference engine
    engine = GestureInferenceEngine(model_path)

    if engine.model is None:
        print("Failed to load model")
        return

    print("Model loaded successfully!")
    print(f"Statistics: {engine.get_statistics()}")

    # Benchmark performance
    print("\nBenchmarking inference performance...")
    benchmark_results = engine.benchmark_inference(50)
    print(f"Benchmark results: {benchmark_results}")

    # Test with random frame
    print("\nTesting with random frame...")
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = engine.predict_gesture(test_frame)
    print(f"Prediction result: {result}")

    # Test probabilities
    probs = engine.get_gesture_probabilities(test_frame)
    print(f"All probabilities: {probs}")

if __name__ == "__main__":
    test_inference_engine()