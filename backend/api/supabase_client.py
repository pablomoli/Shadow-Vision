#!/usr/bin/env python3
"""
Mock Supabase Client for Local Development
Provides mock classes to maintain backwards compatibility without database dependencies
"""

import time
import uuid
from typing import Dict, Any, Optional
import logging

class MockSupabaseLogger:
    """Mock logger that mimics Supabase logging interface"""

    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.logger = logging.getLogger(__name__)

    def log_training_metrics(self, epoch: int, train_loss: float, train_accuracy: float,
                           val_loss: float, val_accuracy: float, model_version: str) -> bool:
        """Mock training metrics logging"""
        self.logger.info(f"Training metrics - Epoch {epoch}: "
                        f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
                        f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        return True

    def log_gesture_detection(self, gesture: str, confidence: float,
                            animation_triggered: str) -> bool:
        """Mock gesture detection logging"""
        self.logger.info(f"Gesture detected: {gesture} (confidence: {confidence:.2f}) "
                        f"-> {animation_triggered}")
        return True

    def start_demo_session(self) -> str:
        """Mock demo session start"""
        self.logger.info(f"Demo session started: {self.session_id}")
        return self.session_id

    def end_demo_session(self, total_gestures: int, successful_detections: int) -> bool:
        """Mock demo session end"""
        self.logger.info(f"Demo session ended: {total_gestures} gestures, "
                        f"{successful_detections} successful detections")
        return True

class SupabaseLogger(MockSupabaseLogger):
    """Alias for backwards compatibility"""
    pass

def create_supabase_logger() -> MockSupabaseLogger:
    """Create mock Supabase logger instance"""
    return MockSupabaseLogger()

def test_supabase_client():
    """Test mock Supabase client functionality"""
    logger = create_supabase_logger()

    print("Testing mock Supabase client...")

    # Test training metrics
    success = logger.log_training_metrics(
        epoch=1,
        train_loss=0.5,
        train_accuracy=85.0,
        val_loss=0.6,
        val_accuracy=82.0,
        model_version="efficient"
    )
    print(f"Training metrics logged: {success}")

    # Test gesture detection
    success = logger.log_gesture_detection(
        gesture="thumbs_up",
        confidence=0.95,
        animation_triggered="celebration"
    )
    print(f"Gesture detection logged: {success}")

    # Test demo session
    session_id = logger.start_demo_session()
    print(f"Demo session started: {session_id}")

    success = logger.end_demo_session(
        total_gestures=50,
        successful_detections=45
    )
    print(f"Demo session ended: {success}")

    print("Mock Supabase client test completed successfully!")

def setup_database_schema():
    """Mock database schema setup"""
    print("Mock database schema setup completed (no-op)")
    return True

if __name__ == "__main__":
    test_supabase_client()