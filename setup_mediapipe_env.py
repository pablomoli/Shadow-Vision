#!/usr/bin/env python3
"""
MediaPipe Environment Setup Script
Creates a stable, version-locked environment for MediaPipe bridges.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, check=True):
    """Run command and handle errors."""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stderr:
            print(f"Stderr: {e.stderr}")
        return False

def check_python_versions():
    """Check available Python versions."""
    print("Checking available Python versions...")

    # Check for Python 3.11 and 3.12
    for version in ["3.11", "3.12", "3.10"]:
        try:
            result = subprocess.run(f"python{version} --version", shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ Found: {result.stdout.strip()}")
                return f"python{version}"
        except:
            pass

    # Check default python version
    try:
        result = subprocess.run("python --version", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"Default Python: {version}")
            if "3.11" in version or "3.12" in version or "3.10" in version:
                return "python"
    except:
        pass

    return None

def setup_venv_environment():
    """Set up virtual environment with MediaPipe."""
    print("\n" + "="*50)
    print("Setting up MediaPipe Virtual Environment")
    print("="*50)

    # Find compatible Python
    python_cmd = check_python_versions()
    if not python_cmd:
        print("❌ No compatible Python version found (need 3.10, 3.11, or 3.12)")
        print("Please install Python 3.11 or 3.12 for MediaPipe compatibility")
        return False

    # Create virtual environment
    env_name = "mediapipe_env"
    print(f"\nCreating virtual environment: {env_name}")

    if not run_command(f"{python_cmd} -m venv {env_name}"):
        return False

    # Determine activation script
    if os.name == 'nt':  # Windows
        activate_cmd = f"{env_name}\\Scripts\\activate"
        pip_cmd = f"{env_name}\\Scripts\\pip"
        python_env_cmd = f"{env_name}\\Scripts\\python"
    else:  # Unix/Linux/Mac
        activate_cmd = f"source {env_name}/bin/activate"
        pip_cmd = f"{env_name}/bin/pip"
        python_env_cmd = f"{env_name}/bin/python"

    print(f"Virtual environment created: {env_name}")

    # Upgrade pip
    print("\nUpgrading pip...")
    if not run_command(f"{pip_cmd} install --upgrade pip"):
        return False

    # Install MediaPipe requirements
    print("\nInstalling MediaPipe and dependencies...")
    if not run_command(f"{pip_cmd} install -r requirements-mediapipe.txt"):
        return False

    # Test MediaPipe installation
    print("\nTesting MediaPipe installation...")
    test_cmd = f"{python_env_cmd} -c \"import mediapipe; print('MediaPipe version:', mediapipe.__version__)\""
    if not run_command(test_cmd):
        return False

    # Create activation script
    create_activation_scripts(env_name)

    print("\n" + "="*50)
    print("✅ MediaPipe Environment Setup Complete!")
    print("="*50)
    print(f"Environment location: ./{env_name}")
    print("\nTo activate the environment:")
    if os.name == 'nt':  # Windows
        print(f"  {env_name}\\Scripts\\activate")
        print("  # or use: python run_mediapipe_bridge.py")
    else:
        print(f"  source {env_name}/bin/activate")
        print("  # or use: python run_mediapipe_bridge.py")

    print("\nTo run MediaPipe bridge:")
    print("  python mediapipe_touchdesigner_bridge.py")

    return True

def create_activation_scripts(env_name):
    """Create convenient activation scripts."""

    # Windows batch script
    if os.name == 'nt':
        batch_script = f"""@echo off
echo Activating MediaPipe environment...
call {env_name}\\Scripts\\activate
echo MediaPipe environment active!
echo.
echo Available commands:
echo   python mediapipe_touchdesigner_bridge.py
echo   python enhanced_mediapipe_touchdesigner_bridge.py
echo   python test_mediapipe_accuracy.py
echo.
cmd /k
"""

        with open("activate_mediapipe.bat", "w") as f:
            f.write(batch_script)
        print("Created: activate_mediapipe.bat")

    # Cross-platform Python script
    py_script = f"""#!/usr/bin/env python3
import subprocess
import sys
import os

def run_in_mediapipe_env():
    env_name = "{env_name}"

    if os.name == 'nt':  # Windows
        python_cmd = f"{env_name}\\\\Scripts\\\\python"
    else:  # Unix/Linux/Mac
        python_cmd = f"{env_name}/bin/python"

    if len(sys.argv) < 2:
        print("Usage: python run_mediapipe_bridge.py <script_name>")
        print("Examples:")
        print("  python run_mediapipe_bridge.py mediapipe_touchdesigner_bridge.py")
        print("  python run_mediapipe_bridge.py enhanced_mediapipe_touchdesigner_bridge.py")
        return

    script_name = sys.argv[1]
    cmd = [python_cmd, script_name] + sys.argv[2:]

    print(f"Running {{script_name}} in MediaPipe environment...")
    subprocess.run(cmd)

if __name__ == "__main__":
    run_in_mediapipe_env()
"""

    with open("run_mediapipe_bridge.py", "w") as f:
        f.write(py_script)
    print("Created: run_mediapipe_bridge.py")

def setup_docker_environment():
    """Set up Docker environment for MediaPipe."""
    print("\n" + "="*50)
    print("Setting up Docker MediaPipe Environment")
    print("="*50)

    # Check if Docker is available
    if not run_command("docker --version", check=False):
        print("❌ Docker not found. Please install Docker Desktop.")
        return False

    # Update Docker configuration for TouchDesigner bridge
    docker_compose_content = """version: '3.8'

services:
  # MediaPipe TouchDesigner Bridge
  mediapipe-bridge:
    build:
      context: .
      dockerfile: Dockerfile.mediapipe
    ports:
      - "7000:7000"  # OSC port for TouchDesigner
      - "8002:8002"  # Additional port
    volumes:
      - ./backend:/app/backend
      - ./models:/app/models
      - ./data:/app/data
      - .:/app/scripts
    environment:
      - PYTHONPATH=/app
      - DISPLAY=host.docker.internal:0.0  # For camera access
    devices:
      - /dev/video0:/dev/video0  # Camera access (Linux)
    network_mode: "host"  # For OSC communication
    command: python scripts/mediapipe_touchdesigner_bridge.py

volumes:
  models_data:
"""

    with open("docker-compose.bridge.yml", "w") as f:
        f.write(docker_compose_content)

    print("Created: docker-compose.bridge.yml")

    # Create Docker run script
    docker_script = """#!/usr/bin/env python3
import subprocess
import sys

def run_docker_bridge():
    print("Starting MediaPipe Bridge in Docker...")
    print("This will run the TouchDesigner bridge with MediaPipe support")
    print()

    # Build and run the container
    cmd = "docker-compose -f docker-compose.bridge.yml up --build mediapipe-bridge"

    try:
        subprocess.run(cmd, shell=True, check=True)
    except KeyboardInterrupt:
        print("\\nStopping MediaPipe bridge...")
        subprocess.run("docker-compose -f docker-compose.bridge.yml down", shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Docker: {e}")

if __name__ == "__main__":
    run_docker_bridge()
"""

    with open("run_docker_bridge.py", "w") as f:
        f.write(docker_script)

    print("Created: run_docker_bridge.py")
    print("\n✅ Docker setup complete!")
    print("To run: python run_docker_bridge.py")

    return True

def main():
    """Main setup function."""
    print("MediaPipe Bridge Environment Setup")
    print("=" * 40)
    print("This script will set up a stable MediaPipe environment for your bridges.")
    print()

    # Check current directory
    if not Path("mediapipe_touchdesigner_bridge.py").exists():
        print("❌ Please run this script from the gesture-puppets directory")
        return 1

    print("Choose setup method:")
    print("1. Virtual Environment (Recommended for development)")
    print("2. Docker Environment (Recommended for production)")
    print("3. Both")

    choice = input("Enter choice (1/2/3): ").strip()

    success = True

    if choice in ["1", "3"]:
        success &= setup_venv_environment()

    if choice in ["2", "3"]:
        success &= setup_docker_environment()

    if success:
        print("\n🎉 Setup complete! Your MediaPipe bridges are ready.")
        print("\nNext steps:")
        print("1. Test the setup: python test_mediapipe_accuracy.py")
        print("2. Run the bridge: python mediapipe_touchdesigner_bridge.py")
        print("3. Open TouchDesigner with OSC In CHOP (port 7000)")
        return 0
    else:
        print("\n❌ Setup failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())