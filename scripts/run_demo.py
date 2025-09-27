#!/usr/bin/env python3
"""
Demo Launcher Script
Quick launcher for the Gesture Puppets demo
"""

import os
import sys
import subprocess
import time
import threading
import webbrowser
from pathlib import Path
import signal
import json

class DemoLauncher:
    """Launches and manages the demo"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.backend_process = None
        self.frontend_process = None
        self.is_running = False

        print("🎭 Gesture Puppets - Demo Launcher")
        print("=" * 50)

    def check_setup(self):
        """Check if the project is properly set up"""
        print("🔍 Checking project setup...")

        # Check virtual environment
        venv_path = self.project_root / "venv"
        if not venv_path.exists():
            print("   ❌ Virtual environment not found")
            print("   Please run: python scripts/setup_environment.py")
            return False
        print("   ✅ Virtual environment found")

        # Check if model exists
        model_path = self.project_root / "backend" / "trained_models" / "efficient_best.pth"
        if not model_path.exists():
            print("   ⚠️  Trained model not found")
            print("   Demo will run without gesture recognition")
        else:
            print("   ✅ Trained model found")

        # Check frontend files
        frontend_path = self.project_root / "frontend" / "public" / "index.html"
        if not frontend_path.exists():
            print("   ❌ Frontend files not found")
            return False
        print("   ✅ Frontend files found")

        return True

    def get_python_executable(self):
        """Get the Python executable from virtual environment"""
        venv_path = self.project_root / "venv"

        if os.name == 'nt':  # Windows
            return venv_path / "Scripts" / "python.exe"
        else:  # Unix-like
            return venv_path / "bin" / "python"

    def start_backend(self):
        """Start the backend server"""
        print("🚀 Starting backend server...")

        python_exec = self.get_python_executable()
        backend_script = self.project_root / "backend" / "main.py"

        try:
            self.backend_process = subprocess.Popen([
                str(python_exec),
                str(backend_script),
                "server",
                "--host", "localhost",
                "--port", "8000"
            ], cwd=self.project_root)

            print("   ✅ Backend server starting on http://localhost:8000")
            return True

        except Exception as e:
            print(f"   ❌ Failed to start backend: {e}")
            return False

    def start_frontend_server(self):
        """Start a simple HTTP server for the frontend"""
        print("🌐 Starting frontend server...")

        try:
            # Use Python's built-in HTTP server
            python_exec = self.get_python_executable()
            frontend_dir = self.project_root / "frontend" / "public"

            self.frontend_process = subprocess.Popen([
                str(python_exec),
                "-m", "http.server",
                "3000",
                "--directory", str(frontend_dir)
            ], cwd=self.project_root)

            print("   ✅ Frontend server starting on http://localhost:3000")
            return True

        except Exception as e:
            print(f"   ❌ Failed to start frontend server: {e}")
            return False

    def wait_for_servers(self):
        """Wait for servers to be ready"""
        print("⏳ Waiting for servers to start...")

        import socket
        import time

        def check_port(host, port):
            try:
                sock = socket.create_connection((host, port), timeout=1)
                sock.close()
                return True
            except:
                return False

        # Wait for backend
        backend_ready = False
        for i in range(30):  # Wait up to 30 seconds
            if check_port("localhost", 8000):
                backend_ready = True
                break
            time.sleep(1)

        if not backend_ready:
            print("   ⚠️  Backend server not responding")
        else:
            print("   ✅ Backend server ready")

        # Wait for frontend
        frontend_ready = False
        for i in range(10):  # Wait up to 10 seconds
            if check_port("localhost", 3000):
                frontend_ready = True
                break
            time.sleep(1)

        if not frontend_ready:
            print("   ⚠️  Frontend server not responding")
        else:
            print("   ✅ Frontend server ready")

        return backend_ready and frontend_ready

    def open_browser(self):
        """Open the demo in a web browser"""
        print("🌐 Opening demo in browser...")

        try:
            webbrowser.open("http://localhost:3000")
            print("   ✅ Browser opened")
        except Exception as e:
            print(f"   ⚠️  Could not open browser: {e}")
            print("   Please manually open: http://localhost:3000")

    def show_demo_info(self):
        """Show demo information and controls"""
        print("\n" + "=" * 50)
        print("🎉 Demo is running!")
        print("=" * 50)

        print("\n📋 Demo Information:")
        print("   Frontend: http://localhost:3000")
        print("   Backend API: http://localhost:8000")
        print("   WebSocket: ws://localhost:8000")

        print("\n🎮 How to use:")
        print("   1. Allow camera access when prompted")
        print("   2. Make hand shadow puppet gestures in front of the camera")
        print("   3. Watch the 3D animations respond to your gestures")

        print("\n🖐️ Supported gestures:")
        gestures = ["dog", "bird", "rabbit", "butterfly", "snake"]
        for gesture in gestures:
            print(f"   • {gesture.capitalize()}")

        print("\n⌨️  Controls:")
        print("   • Press Ctrl+C to stop the demo")
        print("   • Spacebar: Connect/disconnect (in browser)")
        print("   • R: Reset camera view (in browser)")
        print("   • F: Toggle fullscreen (in browser)")

        print("\n🔧 Troubleshooting:")
        print("   • If camera doesn't work, check permissions")
        print("   • If gestures aren't detected, ensure good lighting")
        print("   • If performance is slow, close other applications")

    def setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            print("\n\n🛑 Shutting down demo...")
            self.stop_demo()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, signal_handler)

    def stop_demo(self):
        """Stop all demo processes"""
        self.is_running = False

        if self.backend_process:
            print("   🔄 Stopping backend server...")
            self.backend_process.terminate()
            try:
                self.backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.backend_process.kill()

        if self.frontend_process:
            print("   🔄 Stopping frontend server...")
            self.frontend_process.terminate()
            try:
                self.frontend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.frontend_process.kill()

        print("   ✅ Demo stopped")

    def run_health_check(self):
        """Periodically check if processes are still running"""
        while self.is_running:
            time.sleep(10)  # Check every 10 seconds

            if self.backend_process and self.backend_process.poll() is not None:
                print("   ⚠️  Backend process died unexpectedly")
                self.is_running = False
                break

            if self.frontend_process and self.frontend_process.poll() is not None:
                print("   ⚠️  Frontend process died unexpectedly")
                self.is_running = False
                break

    def run_demo(self):
        """Run the complete demo"""
        if not self.check_setup():
            return False

        self.setup_signal_handlers()

        try:
            # Start servers
            if not self.start_backend():
                return False

            if not self.start_frontend_server():
                self.stop_demo()
                return False

            # Wait for servers to be ready
            if not self.wait_for_servers():
                print("   ⚠️  Some servers may not be fully ready")

            # Open browser
            self.open_browser()

            # Show info
            self.show_demo_info()

            # Set running flag
            self.is_running = True

            # Start health check thread
            health_thread = threading.Thread(target=self.run_health_check, daemon=True)
            health_thread.start()

            # Keep the main thread alive
            try:
                while self.is_running:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass

        except Exception as e:
            print(f"❌ Demo failed: {e}")
            return False
        finally:
            self.stop_demo()

        return True

def show_help():
    """Show help information"""
    print("🎭 Gesture Puppets Demo Launcher")
    print("\nUsage:")
    print("  python scripts/run_demo.py [command]")
    print("\nCommands:")
    print("  run      Start the demo (default)")
    print("  setup    Check setup requirements")
    print("  help     Show this help message")
    print("\nFor more information, see README.md")

def main():
    """Main entry point"""
    command = sys.argv[1] if len(sys.argv) > 1 else "run"

    launcher = DemoLauncher()

    if command == "help":
        show_help()
    elif command == "setup":
        launcher.check_setup()
    elif command == "run":
        launcher.run_demo()
    else:
        print(f"Unknown command: {command}")
        show_help()
        sys.exit(1)

if __name__ == "__main__":
    main()