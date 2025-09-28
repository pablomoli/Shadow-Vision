#!/usr/bin/env python3
"""
Simple Docker Setup Validation for Shadow-Vision
"""

import os
import sys
import subprocess
from pathlib import Path

def check_files():
    """Check if required files exist."""
    print("Checking required files...")

    required_files = [
        "Dockerfile.backend",
        "docker-compose.yml",
        "docker_demo.py",
        "requirements.txt",
        "models/advanced_shadow_puppet_classifier.joblib"
    ]

    missing = []
    for file_path in required_files:
        if Path(file_path).exists():
            size = Path(file_path).stat().st_size
            print(f"  FOUND: {file_path} ({size:,} bytes)")
        else:
            print(f"  MISSING: {file_path}")
            missing.append(file_path)

    return len(missing) == 0

def validate_dockerfile():
    """Check Dockerfile content."""
    print("\nValidating Dockerfile...")

    if not Path("Dockerfile.backend").exists():
        print("  ERROR: Dockerfile.backend not found")
        return False

    with open("Dockerfile.backend", 'r') as f:
        content = f.read()

    checks = [
        "FROM python",
        "WORKDIR /app",
        "requirements.txt",
        "pip install",
        "CMD"
    ]

    for check in checks:
        if check in content:
            print(f"  FOUND: {check}")
        else:
            print(f"  MISSING: {check}")
            return False

    return True

def validate_compose():
    """Check docker-compose.yml."""
    print("\nValidating docker-compose.yml...")

    if not Path("docker-compose.yml").exists():
        print("  ERROR: docker-compose.yml not found")
        return False

    with open("docker-compose.yml", 'r') as f:
        content = f.read()

    checks = [
        "shadow-vision:",
        "build:",
        "dockerfile: Dockerfile.backend",
        "volumes:",
        "models:/app/models"
    ]

    for check in checks:
        if check in content:
            print(f"  FOUND: {check}")
        else:
            print(f"  MISSING: {check}")
            return False

    return True

def check_docker():
    """Check if Docker is available."""
    print("\nChecking Docker...")

    try:
        result = subprocess.run(['docker', '--version'],
                              capture_output=True, text=True, check=True)
        print(f"  SUCCESS: {result.stdout.strip()}")
        return True
    except:
        print("  ERROR: Docker not available")
        print("  Install from: https://docs.docker.com/get-docker/")
        return False

def main():
    """Main validation."""
    print("Shadow-Vision Docker Validation")
    print("="*50)

    tests = [
        ("Required Files", check_files),
        ("Dockerfile", validate_dockerfile),
        ("Compose File", validate_compose),
        ("Docker Install", check_docker)
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"  ERROR in {name}: {e}")
            results.append((name, False))

    print("\n" + "="*50)
    print("VALIDATION RESULTS")
    print("="*50)

    passed = 0
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{name:15} {status}")
        if result:
            passed += 1

    print(f"\nPassed: {passed}/{len(results)} tests")

    if passed == len(results):
        print("\nREADY FOR DOCKER DEPLOYMENT!")
        print("Run: docker-compose up --build shadow-vision")
    else:
        print(f"\nFIX {len(results)-passed} ISSUES BEFORE DEPLOYMENT")

    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)