#!/usr/bin/env python3
"""
Shadow-Vision Live Demo Simulation

Simulates how the live demo will look with:
- Real-time hand detection
- Feature extraction
- Classification
- Performance monitoring
- TouchDesigner OSC output simulation
"""

import cv2
import numpy as np
import time
import sys
from pathlib import Path
import json

# Add backend paths
backend_path = Path(__file__).parent.parent / "backend"
sys.path.append(str(backend_path / "data"))
sys.path.append(str(backend_path / "cv_pipeline"))

try:
    from realtime_inference import RealTimeClassifier
    import joblib
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure the demo model exists")
    sys.exit(1)


class DemoSimulator:
    """Simulates live demo environment."""

    def __init__(self, model_path="models/demo_classifier.joblib"):
        self.model_path = model_path
        self.classifier = None
        self.demo_shapes = self._create_demo_shapes()
        self.current_shape_idx = 0
        self.frame_count = 0

        # Demo settings
        self.shape_hold_frames = 30  # Hold each shape for 30 frames (~1 second)
        self.window_size = (800, 600)

        # Performance tracking
        self.fps_times = []

    def _create_demo_shapes(self):
        """Create different hand-like shapes for demo."""
        shapes = []

        # Bird (thin, pointed)
        bird = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.ellipse(bird, (200, 250), (30, 100), 45, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(bird, (200, 180), (20, 60), 30, 0, 360, (255, 255, 255), -1)
        shapes.append(("bird", bird))

        # Dog (broad, rounded)
        dog = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.ellipse(dog, (200, 200), (80, 60), 0, 0, 360, (255, 255, 255), -1)
        cv2.circle(dog, (150, 160), 25, (255, 255, 255), -1)  # ear
        cv2.circle(dog, (250, 160), 25, (255, 255, 255), -1)  # ear
        cv2.ellipse(dog, (200, 250), (40, 30), 0, 0, 360, (255, 255, 255), -1)  # snout
        shapes.append(("dog", dog))

        # Rabbit (tall with ears)
        rabbit = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.ellipse(rabbit, (200, 250), (50, 80), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(rabbit, (180, 120), (15, 60), 0, 0, 360, (255, 255, 255), -1)  # ear 1
        cv2.ellipse(rabbit, (220, 120), (15, 60), 0, 0, 360, (255, 255, 255), -1)  # ear 2
        shapes.append(("rabbit", rabbit))

        # Duck (rounded with bill)
        duck = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.circle(duck, (200, 200), 70, (255, 255, 255), -1)
        cv2.ellipse(duck, (130, 200), (40, 20), 0, 0, 360, (255, 255, 255), -1)  # bill
        shapes.append(("duck", duck))

        # Wolf (angular, pointed)
        wolf = np.zeros((400, 400, 3), dtype=np.uint8)
        points = np.array([[200, 150], [150, 200], [180, 280], [220, 280], [250, 200]], np.int32)
        cv2.fillPoly(wolf, [points], (255, 255, 255))
        cv2.circle(wolf, (170, 180), 20, (255, 255, 255), -1)  # ear
        cv2.circle(wolf, (230, 180), 20, (255, 255, 255), -1)  # ear
        shapes.append(("wolf", wolf))

        return shapes

    def initialize_classifier(self):
        """Initialize the real-time classifier."""
        try:
            # Check if demo model exists
            if not Path(self.model_path).exists():
                print(f"Demo model not found: {self.model_path}")
                print("Please run the demo model creation script first")
                return False

            self.classifier = RealTimeClassifier(self.model_path)
            print(f"✅ Classifier initialized with demo model")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize classifier: {e}")
            return False

    def get_current_demo_frame(self):
        """Get current demo frame with synthetic hand shape."""
        shape_name, shape_img = self.demo_shapes[self.current_shape_idx]

        # Create a frame that looks like camera input
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame.fill(30)  # Dark background

        # Place the shape in the center
        y_offset = (frame.shape[0] - shape_img.shape[0]) // 2
        x_offset = (frame.shape[1] - shape_img.shape[1]) // 2

        # Blend the shape onto the frame
        mask = shape_img[:, :, 0] > 0
        frame[y_offset:y_offset+shape_img.shape[0],
              x_offset:x_offset+shape_img.shape[1]][mask] = shape_img[mask]

        return frame, shape_name

    def draw_demo_ui(self, frame, result, expected_class):
        """Draw demo UI overlay."""
        display_frame = cv2.resize(frame, self.window_size)

        # Background for text
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (10, 10), (self.window_size[0]-10, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)

        # Title
        cv2.putText(display_frame, "Shadow-Vision Live Demo", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

        # Expected vs Predicted
        cv2.putText(display_frame, f"Expected: {expected_class.upper()}", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if result.hand_detected:
            color = (0, 255, 0) if result.predicted_class.lower() == expected_class.lower() else (0, 255, 255)
            cv2.putText(display_frame, f"Predicted: {result.predicted_class.upper()}", (20, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(display_frame, f"Confidence: {result.confidence:.1%}", (20, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(display_frame, "Predicted: NO HAND DETECTED", (20, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Performance stats
        if len(self.fps_times) > 1:
            fps = len(self.fps_times) / (self.fps_times[-1] - self.fps_times[0]) if len(self.fps_times) > 1 else 0
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (20, 170),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Instructions
        cv2.putText(display_frame, "Press 'q' to quit, 'n' for next shape",
                   (20, self.window_size[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # OSC Simulation
        if result.hand_detected and result.confidence > 0.3:
            cv2.putText(display_frame, f"OSC -> TouchDesigner: {result.predicted_class}",
                       (20, self.window_size[1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return display_frame

    def next_shape(self):
        """Move to next demo shape."""
        self.current_shape_idx = (self.current_shape_idx + 1) % len(self.demo_shapes)
        self.frame_count = 0
        print(f"Switched to: {self.demo_shapes[self.current_shape_idx][0]}")

    def run_demo(self):
        """Run the live demo simulation."""
        print("Shadow-Vision Live Demo Simulation")
        print("=" * 40)

        if not self.initialize_classifier():
            return

        print("Demo Controls:")
        print("  'q' - Quit demo")
        print("  'n' - Next shape")
        print("  Auto-advance every ~1 second")
        print()
        print("Starting demo...")

        cv2.namedWindow("Shadow-Vision Demo", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Shadow-Vision Demo", self.window_size[0], self.window_size[1])

        try:
            while True:
                start_time = time.time()

                # Get current demo frame
                frame, expected_class = self.get_current_demo_frame()

                # Run inference
                result = self.classifier.predict_from_frame(frame)

                # Draw UI
                display_frame = self.draw_demo_ui(frame, result, expected_class)

                # Show frame
                cv2.imshow("Shadow-Vision Demo", display_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('n'):
                    self.next_shape()

                # Auto-advance shapes
                self.frame_count += 1
                if self.frame_count >= self.shape_hold_frames:
                    self.next_shape()

                # Track FPS
                self.fps_times.append(time.time())
                if len(self.fps_times) > 30:  # Keep last 30 frames
                    self.fps_times.pop(0)

                # Simulate real-time (target ~30 FPS)
                elapsed = time.time() - start_time
                target_frame_time = 1.0 / 30.0
                if elapsed < target_frame_time:
                    time.sleep(target_frame_time - elapsed)

        except KeyboardInterrupt:
            print("\nDemo interrupted by user")
        finally:
            cv2.destroyAllWindows()

            # Print final stats
            if len(self.fps_times) > 1:
                total_time = self.fps_times[-1] - self.fps_times[0]
                avg_fps = len(self.fps_times) / total_time
                print(f"\nDemo completed!")
                print(f"Average FPS: {avg_fps:.1f}")
                print(f"Total frames: {len(self.fps_times)}")


def main():
    """Main demo function."""
    try:
        simulator = DemoSimulator()
        simulator.run_demo()
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())