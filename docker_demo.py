#!/usr/bin/env python3
"""
Docker-optimized Shadow-Vision demo script.

This script is designed to run inside Docker containers with proper
camera access and headless operation capabilities.
"""

import cv2
import numpy as np
import logging
import json
import sys
import os
from pathlib import Path
from backend.data.advanced_feature_extractor import AdvancedFeatureExtractor
import joblib

# Setup logging for container environment
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/app/demo.log') if os.path.exists('/app') else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

class DockerShadowVision:
    """Docker-optimized Shadow-Vision demo."""

    def __init__(self):
        """Initialize Docker demo."""
        self.model = None
        self.extractor = None
        self.classes = ['bird', 'cat', 'llama', 'rabbit', 'deer', 'dog', 'snail', 'swan']
        self.load_model()

    def load_model(self):
        """Load the advanced model."""
        model_path = Path("models/advanced_shadow_puppet_classifier.joblib")
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            logger.info("Available files in models/:")
            if Path("models").exists():
                for f in Path("models").iterdir():
                    logger.info(f"  {f.name}")
            return False

        try:
            self.model = joblib.load(model_path)
            self.extractor = AdvancedFeatureExtractor()

            # Load metrics
            metrics_path = Path("models/advanced_training_metrics.json")
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                logger.info(f"Model loaded - Accuracy: {metrics['accuracy']:.1%}")
            else:
                logger.info("Model loaded successfully")

            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def test_camera(self):
        """Test camera access in container."""
        logger.info("Testing camera access...")

        # Try multiple camera indices
        for camera_id in [0, 1, 2]:
            try:
                cap = cv2.VideoCapture(camera_id)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        logger.info(f"✅ Camera {camera_id} working - Frame: {frame.shape}")
                        cap.release()
                        return camera_id
                    else:
                        logger.warning(f"Camera {camera_id} opened but no frame")
                cap.release()
            except Exception as e:
                logger.warning(f"Camera {camera_id} error: {e}")

        logger.error("❌ No working camera found")
        return None

    def run_headless_demo(self, max_frames=100):
        """Run demo without GUI (for containers)."""
        logger.info("Starting headless Shadow-Vision demo...")

        if not self.model:
            logger.error("Model not loaded")
            return False

        camera_id = self.test_camera()
        if camera_id is None:
            logger.error("Cannot run demo without camera")
            return False

        cap = cv2.VideoCapture(camera_id)
        frame_count = 0
        predictions = []

        try:
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame")
                    break

                # Extract features and predict
                try:
                    features = self.extractor.extract_features(frame)
                    if features is not None:
                        prediction = self.model.predict([features])[0]
                        confidence = max(self.model.predict_proba([features])[0])

                        predictions.append({
                            'frame': frame_count,
                            'prediction': prediction,
                            'confidence': confidence
                        })

                        logger.info(f"Frame {frame_count}: {prediction} ({confidence:.2f})")
                    else:
                        logger.debug(f"Frame {frame_count}: No features extracted")

                except Exception as e:
                    logger.warning(f"Frame {frame_count} processing error: {e}")

                frame_count += 1

                # Brief pause to prevent overwhelming logs
                if frame_count % 10 == 0:
                    import time
                    time.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("Demo stopped by user")
        finally:
            cap.release()

        # Summary
        if predictions:
            unique_predictions = set(p['prediction'] for p in predictions)
            avg_confidence = np.mean([p['confidence'] for p in predictions])

            logger.info(f"\n=== DEMO SUMMARY ===")
            logger.info(f"Processed frames: {frame_count}")
            logger.info(f"Successful predictions: {len(predictions)}")
            logger.info(f"Average confidence: {avg_confidence:.2f}")
            logger.info(f"Detected gestures: {list(unique_predictions)}")

            # Save results
            results_path = Path("/app/demo_results.json")
            with open(results_path, 'w') as f:
                json.dump({
                    'total_frames': frame_count,
                    'predictions': predictions,
                    'summary': {
                        'avg_confidence': avg_confidence,
                        'detected_gestures': list(unique_predictions)
                    }
                }, f, indent=2)
            logger.info(f"Results saved to: {results_path}")
        else:
            logger.warning("No successful predictions made")

        return len(predictions) > 0

    def run_gui_demo(self):
        """Run demo with GUI (for X11 forwarding)."""
        logger.info("Starting GUI Shadow-Vision demo...")

        if not self.model:
            logger.error("Model not loaded")
            return False

        camera_id = self.test_camera()
        if camera_id is None:
            logger.error("Cannot run demo without camera")
            return False

        cap = cv2.VideoCapture(camera_id)

        try:
            logger.info("Demo controls: 'q' to quit, 's' to save frame")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Extract features and predict
                try:
                    features = self.extractor.extract_features(frame)
                    if features is not None:
                        prediction = self.model.predict([features])[0]
                        confidence = max(self.model.predict_proba([features])[0])

                        # Add prediction text to frame
                        text = f"{prediction}: {confidence:.2f}"
                        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                except Exception as e:
                    cv2.putText(frame, "Processing error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Display frame
                cv2.imshow('Shadow-Vision Docker Demo', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    cv2.imwrite('/app/saved_frame.jpg', frame)
                    logger.info("Frame saved to /app/saved_frame.jpg")

        except Exception as e:
            logger.error(f"GUI demo error: {e}")
            return False
        finally:
            cap.release()
            cv2.destroyAllWindows()

        return True

def main():
    """Main function."""
    logger.info("🎭 Shadow-Vision Docker Demo")
    logger.info("=" * 50)

    demo = DockerShadowVision()

    # Check environment
    display = os.environ.get('DISPLAY')
    if display:
        logger.info(f"DISPLAY detected: {display}")
        logger.info("Running GUI demo...")
        success = demo.run_gui_demo()
    else:
        logger.info("No DISPLAY detected - running headless demo")
        success = demo.run_headless_demo()

    if success:
        logger.info("✅ Demo completed successfully")
        return 0
    else:
        logger.error("❌ Demo failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())