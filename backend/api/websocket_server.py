#!/usr/bin/env python3
"""
WebSocket Server for Real-time Gesture Recognition
Handles real-time communication between CV backend and frontend
"""

import asyncio
import websockets
import json
import cv2
import numpy as np
import base64
import time
import threading
from typing import Set, Dict, Any, Optional
import logging
from pathlib import Path

from cv_pipeline.camera_handler import CameraHandler
from cv_pipeline.inference_engine import GestureInferenceEngine
from api.supabase_client import create_supabase_logger

class GestureWebSocketServer:
    """WebSocket server for real-time gesture recognition"""

    def __init__(self, host: str = "localhost", port: int = 8000,
                 model_path: str = "backend/trained_models/efficient_best.pth"):
        self.host = host
        self.port = port
        self.model_path = model_path

        # Connected clients
        self.clients: Set[websockets.WebSocketServerProtocol] = set()

        # Components
        self.camera_handler: Optional[CameraHandler] = None
        self.inference_engine: Optional[GestureInferenceEngine] = None
        self.logger = create_supabase_logger()

        # Configuration
        self.config = self._load_config()

        # State
        self.is_running = False
        self.current_gesture = None
        self.last_gesture_time = 0
        self.gesture_cooldown = 2.0  # seconds

        # Statistics
        self.total_connections = 0
        self.total_predictions = 0
        self.start_time = time.time()

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger_std = logging.getLogger(__name__)

    def _load_config(self) -> Dict:
        """Load configuration from files"""
        config = {
            "confidence_threshold": 0.8,
            "camera_index": 0,
            "frame_width": 640,
            "frame_height": 480,
            "fps": 30
        }

        # Load from gesture mapping
        try:
            with open("config/gesture_map.json", 'r') as f:
                gesture_config = json.load(f)
                config.update(gesture_config)
        except:
            self.logger_std.warning("Could not load gesture mapping config")

        return config

    async def initialize(self) -> bool:
        """Initialize camera and inference engine"""
        try:
            # Initialize camera
            self.camera_handler = CameraHandler(
                camera_index=self.config.get("camera_index", 0),
                width=self.config.get("frame_width", 640),
                height=self.config.get("frame_height", 480),
                fps=self.config.get("fps", 30)
            )

            if not self.camera_handler.start_capture():
                self.logger_std.error("Failed to start camera")
                return False

            # Initialize inference engine
            if Path(self.model_path).exists():
                self.inference_engine = GestureInferenceEngine(
                    model_path=self.model_path,
                    confidence_threshold=self.config.get("confidence_threshold", 0.8)
                )

                if self.inference_engine.model is None:
                    self.logger_std.error("Failed to load gesture recognition model")
                    return False
            else:
                self.logger_std.warning(f"Model not found at {self.model_path}")
                self.logger_std.warning("Running in demo mode without gesture recognition")

            self.logger_std.info("Server initialized successfully")
            return True

        except Exception as e:
            self.logger_std.error(f"Initialization failed: {e}")
            return False

    async def register_client(self, websocket: websockets.WebSocketServerProtocol):
        """Register new client connection"""
        self.clients.add(websocket)
        self.total_connections += 1

        # Send initial configuration
        await self.send_to_client(websocket, {
            "type": "config",
            "data": {
                "gesture_classes": list(self.config.get("gestures", {}).keys()),
                "confidence_threshold": self.config.get("confidence_threshold", 0.8),
                "has_model": self.inference_engine is not None,
                "session_id": self.logger.session_id
            }
        })

        self.logger_std.info(f"Client connected. Total clients: {len(self.clients)}")

    async def unregister_client(self, websocket: websockets.WebSocketServerProtocol):
        """Unregister client connection"""
        self.clients.discard(websocket)
        self.logger_std.info(f"Client disconnected. Total clients: {len(self.clients)}")

    async def send_to_client(self, websocket: websockets.WebSocketServerProtocol, message: Dict):
        """Send message to specific client"""
        try:
            await websocket.send(json.dumps(message))
        except websockets.exceptions.ConnectionClosed:
            await self.unregister_client(websocket)
        except Exception as e:
            self.logger_std.error(f"Error sending message to client: {e}")

    async def broadcast_to_all(self, message: Dict):
        """Broadcast message to all connected clients"""
        if not self.clients:
            return

        # Create list of send tasks
        tasks = []
        for client in self.clients.copy():
            tasks.append(self.send_to_client(client, message))

        # Execute all tasks concurrently
        await asyncio.gather(*tasks, return_exceptions=True)

    async def handle_client_message(self, websocket: websockets.WebSocketServerProtocol, message: str):
        """Handle incoming message from client"""
        try:
            data = json.loads(message)
            message_type = data.get("type")

            if message_type == "set_confidence_threshold":
                threshold = data.get("threshold", 0.8)
                if self.inference_engine:
                    self.inference_engine.update_confidence_threshold(threshold)
                await self.send_to_client(websocket, {
                    "type": "config_updated",
                    "data": {"confidence_threshold": threshold}
                })

            elif message_type == "get_statistics":
                stats = await self.get_server_statistics()
                await self.send_to_client(websocket, {
                    "type": "statistics",
                    "data": stats
                })

            elif message_type == "reset_gesture_smoothing":
                if self.inference_engine:
                    self.inference_engine.reset_smoothing()
                await self.send_to_client(websocket, {
                    "type": "smoothing_reset",
                    "data": {"status": "success"}
                })

            elif message_type == "ping":
                await self.send_to_client(websocket, {
                    "type": "pong",
                    "data": {"timestamp": time.time()}
                })

        except Exception as e:
            self.logger_std.error(f"Error handling client message: {e}")

    async def process_frame_loop(self):
        """Main frame processing loop"""
        while self.is_running:
            try:
                if not self.camera_handler or not self.camera_handler.is_running:
                    await asyncio.sleep(0.1)
                    continue

                # Get latest frame
                frame = self.camera_handler.get_latest_frame()
                if frame is None:
                    await asyncio.sleep(0.033)  # ~30 FPS
                    continue

                # Prepare frame data for frontend
                frame_data = {
                    "type": "frame",
                    "data": {
                        "timestamp": time.time(),
                        "frame": self._encode_frame(frame),
                        "camera_stats": self.camera_handler.get_statistics()
                    }
                }

                # Gesture recognition
                if self.inference_engine:
                    gesture_result = await self._process_gesture(frame)
                    if gesture_result:
                        frame_data["data"]["gesture"] = gesture_result

                # Broadcast to clients
                await self.broadcast_to_all(frame_data)

                # Control frame rate
                await asyncio.sleep(0.033)  # ~30 FPS

            except Exception as e:
                self.logger_std.error(f"Error in frame processing loop: {e}")
                await asyncio.sleep(0.1)

    async def _process_gesture(self, frame: np.ndarray) -> Optional[Dict]:
        """Process gesture recognition for frame"""
        try:
            # Run inference
            result = self.inference_engine.predict_gesture(frame)
            self.total_predictions += 1

            # Check for successful detection
            if result["status"] == "success" and result["gesture"]:
                current_time = time.time()

                # Apply cooldown to prevent rapid-fire detections
                if (self.current_gesture != result["gesture"] or
                    current_time - self.last_gesture_time > self.gesture_cooldown):

                    self.current_gesture = result["gesture"]
                    self.last_gesture_time = current_time

                    # Get animation info from config
                    gesture_config = self.config.get("gestures", {}).get(result["gesture"], {})
                    animation_triggered = self._select_random_animation(gesture_config)

                    # Log to database (now uses mock logger)
                    self.logger.log_gesture_detection(
                        gesture=result["gesture"],
                        confidence=result["confidence"],
                        animation_triggered=animation_triggered
                    )

                    # Return complete gesture data
                    return {
                        "gesture": result["gesture"],
                        "confidence": result["confidence"],
                        "animation": animation_triggered,
                        "scene_config": gesture_config,
                        "timestamp": current_time
                    }

            return None

        except Exception as e:
            self.logger_std.error(f"Gesture processing error: {e}")
            return None

    def _select_random_animation(self, gesture_config: Dict) -> str:
        """Select random animation variant for gesture"""
        import random

        animations = gesture_config.get("animations", ["default"])
        return random.choice(animations)

    def _encode_frame(self, frame: np.ndarray) -> str:
        """Encode frame as base64 JPEG"""
        try:
            # Resize frame for transmission
            frame_small = cv2.resize(frame, (320, 240))

            # Encode as JPEG
            _, buffer = cv2.imencode('.jpg', frame_small, [cv2.IMWRITE_JPEG_QUALITY, 70])

            # Convert to base64
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{frame_base64}"

        except Exception as e:
            self.logger_std.error(f"Frame encoding error: {e}")
            return ""

    async def get_server_statistics(self) -> Dict:
        """Get server statistics"""
        uptime = time.time() - self.start_time

        stats = {
            "uptime_seconds": uptime,
            "total_connections": self.total_connections,
            "active_clients": len(self.clients),
            "total_predictions": self.total_predictions,
            "current_gesture": self.current_gesture
        }

        # Add camera stats
        if self.camera_handler:
            stats["camera"] = self.camera_handler.get_statistics()

        # Add inference stats
        if self.inference_engine:
            stats["inference"] = self.inference_engine.get_statistics()

        return stats

    async def client_handler(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """Handle individual client connection"""
        await self.register_client(websocket)

        try:
            async for message in websocket:
                await self.handle_client_message(websocket, message)

        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            self.logger_std.error(f"Client handler error: {e}")
        finally:
            await self.unregister_client(websocket)

    async def start_server(self):
        """Start the WebSocket server"""
        if not await self.initialize():
            self.logger_std.error("Server initialization failed")
            return False

        self.is_running = True

        # Start frame processing loop
        frame_task = asyncio.create_task(self.process_frame_loop())

        # Start demo session
        self.logger.start_demo_session()

        self.logger_std.info(f"Starting WebSocket server on {self.host}:{self.port}")

        try:
            # Start WebSocket server
            async with websockets.serve(self.client_handler, self.host, self.port):
                self.logger_std.info("WebSocket server started successfully")
                await asyncio.Future()  # Run forever

        except Exception as e:
            self.logger_std.error(f"Server error: {e}")
        finally:
            self.is_running = False
            frame_task.cancel()
            await self.cleanup()

    async def cleanup(self):
        """Cleanup resources"""
        self.logger_std.info("Cleaning up server resources...")

        # Stop camera
        if self.camera_handler:
            self.camera_handler.stop_capture()

        # End demo session
        self.logger.end_demo_session(
            total_gestures=self.total_predictions,
            successful_detections=len([c for c in self.clients])
        )

        self.logger_std.info("Server cleanup completed")

def main():
    """Main function to run the server"""
    import argparse

    parser = argparse.ArgumentParser(description='Gesture Recognition WebSocket Server')
    parser.add_argument('--host', default='localhost', help='Server host')
    parser.add_argument('--port', type=int, default=8000, help='Server port')
    parser.add_argument('--model', default='backend/trained_models/efficient_best.pth',
                       help='Path to trained model')

    args = parser.parse_args()

    server = GestureWebSocketServer(args.host, args.port, args.model)

    try:
        asyncio.run(server.start_server())
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Server error: {e}")

if __name__ == "__main__":
    main()