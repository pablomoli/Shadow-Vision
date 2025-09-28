#!/usr/bin/env python3
"""
Docker Setup Validation Script

Test script to validate Docker configuration and deployment readiness.
Run this script to check if Docker setup will work on target machines.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_docker_availability():
    """Check if Docker is installed and running."""
    print("Docker: Checking Docker availability...")

    try:
        result = subprocess.run(['docker', '--version'],
                              capture_output=True, text=True, check=True)
        print(f"SUCCESS: Docker installed: {result.stdout.strip()}")

        # Check if Docker daemon is running
        result = subprocess.run(['docker', 'info'],
                              capture_output=True, text=True, check=True)
        print("SUCCESS: Docker daemon running")
        return True

    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: Docker not available")
        print("Install Docker: https://docs.docker.com/get-docker/")
        return False

def check_docker_compose():
    """Check if Docker Compose is available."""
    print("\nCompose: Checking Docker Compose...")

    try:
        result = subprocess.run(['docker-compose', '--version'],
                              capture_output=True, text=True, check=True)
        print(f"✅ Docker Compose: {result.stdout.strip()}")
        return True

    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            # Try newer docker compose command
            result = subprocess.run(['docker', 'compose', 'version'],
                                  capture_output=True, text=True, check=True)
            print(f"✅ Docker Compose (plugin): {result.stdout.strip()}")
            return True
        except:
            print("❌ Docker Compose not available")
            return False

def validate_dockerfile():
    """Validate Dockerfile syntax and requirements."""
    print("\n📝 Validating Dockerfile...")

    dockerfile_path = Path("Dockerfile.backend")
    if not dockerfile_path.exists():
        print("❌ Dockerfile.backend not found")
        return False

    print("✅ Dockerfile.backend exists")

    # Check essential components
    with open(dockerfile_path, 'r') as f:
        content = f.read()

    checks = [
        ("FROM python", "Base image specified"),
        ("WORKDIR", "Working directory set"),
        ("requirements.txt", "Requirements copied"),
        ("pip install", "Dependencies installed"),
        ("CMD", "Default command specified")
    ]

    all_good = True
    for check, description in checks:
        if check in content:
            print(f"✅ {description}")
        else:
            print(f"❌ {description}")
            all_good = False

    return all_good

def validate_compose_file():
    """Validate docker-compose.yml configuration."""
    print("\n🐙 Validating docker-compose.yml...")

    compose_path = Path("docker-compose.yml")
    if not compose_path.exists():
        print("❌ docker-compose.yml not found")
        return False

    print("✅ docker-compose.yml exists")

    try:
        # Try to parse as YAML (basic validation)
        import yaml
        with open(compose_path, 'r') as f:
            compose_config = yaml.safe_load(f)

        # Check essential components
        if 'services' in compose_config:
            print("✅ Services defined")

            shadow_vision = compose_config['services'].get('shadow-vision')
            if shadow_vision:
                print("✅ shadow-vision service configured")

                if 'build' in shadow_vision:
                    print("✅ Build configuration present")

                if 'volumes' in shadow_vision:
                    print("✅ Volume mounts configured")

                if 'environment' in shadow_vision:
                    print("✅ Environment variables set")

            else:
                print("❌ shadow-vision service not found")
                return False
        else:
            print("❌ No services defined")
            return False

        return True

    except ImportError:
        print("⚠️  PyYAML not available - basic validation only")
        return True
    except Exception as e:
        print(f"❌ Invalid YAML: {e}")
        return False

def check_required_files():
    """Check if all required files are present."""
    print("\n📁 Checking required files...")

    required_files = [
        "requirements.txt",
        "docker_demo.py",
        "backend/data/advanced_feature_extractor.py",
        "models/advanced_shadow_puppet_classifier.joblib",
        "models/advanced_training_metrics.json"
    ]

    all_present = True
    for file_path in required_files:
        if Path(file_path).exists():
            size = Path(file_path).stat().st_size
            print(f"✅ {file_path} ({size:,} bytes)")
        else:
            print(f"❌ {file_path} missing")
            all_present = False

    return all_present

def test_docker_build():
    """Test Docker build process."""
    print("\n🔨 Testing Docker build...")

    if not check_docker_availability():
        print("⚠️  Skipping build test - Docker not available")
        return False

    try:
        # Test build (dry run)
        cmd = ['docker', 'build', '-f', 'Dockerfile.backend', '-t', 'shadow-vision-test', '.']
        print(f"Running: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            print("✅ Docker build successful")

            # Clean up test image
            subprocess.run(['docker', 'rmi', 'shadow-vision-test'],
                         capture_output=True)
            return True
        else:
            print("❌ Docker build failed:")
            print(result.stderr[-500:])  # Last 500 chars of error
            return False

    except subprocess.TimeoutExpired:
        print("❌ Docker build timed out")
        return False
    except Exception as e:
        print(f"❌ Build test error: {e}")
        return False

def generate_deployment_guide():
    """Generate platform-specific deployment instructions."""
    print("\n📋 Generating deployment guide...")

    guide = {
        "linux": {
            "setup": [
                "sudo apt update",
                "sudo apt install docker.io docker-compose",
                "sudo usermod -aG docker $USER",
                "# Log out and back in"
            ],
            "camera": [
                "# Check camera device",
                "ls /dev/video*",
                "# Ensure camera permissions"
            ],
            "run": [
                "git clone https://github.com/pablomoli/Shadow-Vision.git",
                "cd Shadow-Vision",
                "docker-compose up --build shadow-vision"
            ]
        },
        "macos": {
            "setup": [
                "# Install Docker Desktop from docker.com",
                "# Or: brew install --cask docker"
            ],
            "camera": [
                "# Grant camera permission to Docker Desktop",
                "# System Preferences → Security & Privacy → Camera"
            ],
            "run": [
                "git clone https://github.com/pablomoli/Shadow-Vision.git",
                "cd Shadow-Vision",
                "docker-compose up --build shadow-vision"
            ]
        },
        "windows": {
            "setup": [
                "# Install Docker Desktop from docker.com",
                "# Enable WSL2 if prompted"
            ],
            "camera": [
                "# Grant camera permission to Docker Desktop",
                "# Settings → Privacy → Camera"
            ],
            "run": [
                "git clone https://github.com/pablomoli/Shadow-Vision.git",
                "cd Shadow-Vision",
                "docker-compose up --build shadow-vision"
            ]
        }
    }

    with open("deployment_guide.json", "w") as f:
        json.dump(guide, f, indent=2)

    print("✅ Deployment guide saved to deployment_guide.json")
    return True

def main():
    """Main validation function."""
    print("Shadow-Vision Docker Setup Validation")
    print("=" * 60)

    tests = [
        ("Docker Files", validate_dockerfile),
        ("Compose Config", validate_compose_file),
        ("Required Files", check_required_files),
        ("Docker Available", check_docker_availability),
        ("Docker Compose", check_docker_compose),
        ("Deployment Guide", generate_deployment_guide)
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))

    # Optional build test (if Docker available)
    if any(result for name, result in results if "Docker" in name):
        try:
            build_result = test_docker_build()
            results.append(("Docker Build", build_result))
        except:
            results.append(("Docker Build", False))

    # Summary
    print("\n" + "=" * 60)
    print("📋 DOCKER VALIDATION SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed >= total - 1:  # Allow Docker build to fail if Docker not available
        print("\n🎉 DOCKER SETUP READY!")
        print("Ready for deployment with: docker-compose up --build")
    else:
        print(f"\n⚠️  {total - passed} issues found. Check errors above.")

    return passed >= total - 1

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)