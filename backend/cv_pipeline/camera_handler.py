#!/usr/bin/env python3
"""
Camera Handler for Real-time Gesture Recognition
Handles webcam capture, preprocessing, and frame management
"""

import cv2
import numpy as np
import threading
import time
from queue import Queue
from typing import Optional, Tuple, Callable
import logging

class CameraHandler:
    """Manages webcam capture and preprocessing for gesture recognition"""

    def __init__(self, camera_index=0, width=640, height=480, fps=30):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps

        # Camera capture
        self.cap = None
        self.is_running = False

        # Threading
        self.capture_thread = None
        self.frame_queue = Queue(maxsize=2)  # Keep only latest frames

        # Frame processing
        self.latest_frame = None
        self.frame_lock = threading.Lock()

        # Statistics
        self.frames_captured = 0
        self.frames_dropped = 0
        self.last_fps_time = time.time()
        self.fps_counter = 0
        self.current_fps = 0

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def initialize_camera(self) -> bool:
        """Initialize camera with specified settings"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)

            if not self.cap.isOpened():
                self.logger.error(f"Failed to open camera {self.camera_index}")
                return False

            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            # Verify settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

            self.logger.info(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps} FPS")

            return True

        except Exception as e:
            self.logger.error(f"Camera initialization failed: {e}")
            return False

    def start_capture(self):
        """Start camera capture in separate thread"""
        if self.is_running:
            self.logger.warning("Camera capture already running")
            return

        if not self.initialize_camera():
            return False

        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

        self.logger.info("Camera capture started")
        return True

    def stop_capture(self):
        """Stop camera capture"""
        self.is_running = False

        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)

        if self.cap:
            self.cap.release()
            self.cap = None

        self.logger.info("Camera capture stopped")

    def _capture_loop(self):
        """Main capture loop running in separate thread"""
        while self.is_running:
            try:
                ret, frame = self.cap.read()

                if not ret:
                    self.logger.warning("Failed to capture frame")
                    continue

                # Update statistics
                self.frames_captured += 1
                self._update_fps_counter()

                # Store latest frame
                with self.frame_lock:
                    self.latest_frame = frame.copy()

                # Add to queue (non-blocking)
                if not self.frame_queue.full():
                    self.frame_queue.put(frame, block=False)
                else:
                    # Drop oldest frame
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put(frame, block=False)
                        self.frames_dropped += 1
                    except:
                        pass

            except Exception as e:
                self.logger.error(f"Error in capture loop: {e}")
                break

    def _update_fps_counter(self):
        """Update FPS statistics"""
        self.fps_counter += 1
        current_time = time.time()

        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.last_fps_time = current_time

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the most recent frame"""
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def get_frame_from_queue(self, timeout=0.1) -> Optional[np.ndarray]:
        """Get frame from queue with timeout"""
        try:
            return self.frame_queue.get(timeout=timeout)
        except:
            return None

    def preprocess_frame(self, frame: np.ndarray,
                        target_size: Tuple[int, int] = (224, 224),
                        normalize: bool = True,
                        flip_horizontal: bool = True) -> np.ndarray:
        """Preprocess frame for model inference"""

        if frame is None:
            return None

        # Flip horizontally for mirror effect
        if flip_horizontal:
            frame = cv2.flip(frame, 1)

        # Resize
        if target_size:
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Normalize
        if normalize:
            frame_rgb = frame_rgb.astype(np.float32) / 255.0

        return frame_rgb

    def extract_hand_region(self, frame: np.ndarray,
                           bbox: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
        """Extract hand region from frame"""
        if frame is None:
            return None

        if bbox is None:
            # Use center region as default
            h, w = frame.shape[:2]
            bbox = (w//4, h//4, w//2, h//2)

        x, y, w, h = bbox

        # Ensure bbox is within frame bounds
        x = max(0, min(x, frame.shape[1] - 1))
        y = max(0, min(y, frame.shape[0] - 1))
        w = max(1, min(w, frame.shape[1] - x))
        h = max(1, min(h, frame.shape[0] - y))

        return frame[y:y+h, x:x+w]

    def detect_hand_region_simple(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Simple hand detection using skin color and contours"""
        if frame is None:
            return None

        # Convert to HSV for better skin detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define skin color range
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        # Create mask
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # Morphological operations to clean up mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Add some padding
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame.shape[1] - x, w + 2 * padding)
        h = min(frame.shape[0] - y, h + 2 * padding)

        # Ensure minimum size
        min_size = 100
        if w < min_size or h < min_size:
            return None

        return (x, y, w, h)

    def get_statistics(self) -> dict:
        """Get capture statistics"""
        return {
            "frames_captured": self.frames_captured,
            "frames_dropped": self.frames_dropped,
            "current_fps": self.current_fps,
            "is_running": self.is_running,
            "camera_index": self.camera_index,
            "resolution": f"{self.width}x{self.height}"
        }

    def __enter__(self):
        """Context manager entry"""
        self.start_capture()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_capture()

def test_camera():
    """Test camera functionality"""
    print("Testing camera handler...")

    with CameraHandler(camera_index=0, width=640, height=480) as camera:
        if not camera.is_running:
            print("Failed to start camera")
            return

        print("Camera started. Press 'q' to quit...")

        while True:
            frame = camera.get_latest_frame()

            if frame is not None:
                # Preprocess frame
                processed = camera.preprocess_frame(frame, target_size=(224, 224))

                # Detect hand region
                hand_bbox = camera.detect_hand_region_simple(frame)

                # Draw bounding box if hand detected
                if hand_bbox:
                    x, y, w, h = hand_bbox
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "Hand Detected", (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Show statistics
                stats = camera.get_statistics()
                cv2.putText(frame, f"FPS: {stats['current_fps']}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Display frames
                cv2.imshow("Camera Feed", frame)

                if processed is not None:
                    # Convert back to BGR for display
                    processed_bgr = cv2.cvtColor((processed * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    cv2.imshow("Processed", processed_bgr)

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        print("Camera test completed")

if __name__ == "__main__":
    test_camera()