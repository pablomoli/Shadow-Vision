#!/usr/bin/env python3
"""
Shadow-Vision Quick Start Script

One-click setup and demo launcher for new users.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and return success status."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True,
                              capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_virtual_env():
    """Check if we're in a virtual environment."""
    return hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )

def main():
    """Main quick start function."""
    print("🎭 Shadow-Vision Quick Start")
    print("=" * 50)

    # Check if we're in project directory
    if not Path("requirements.txt").exists():
        print("❌ Please run this script from the Shadow-Vision project directory")
        print("cd Shadow-Vision")
        return False

    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False

    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")

    # Check virtual environment
    if not check_virtual_env():
        print("\n⚠️  Virtual environment not detected")
        print("Recommended: Create and activate virtual environment first:")
        print("python -m venv .venv")
        print(".venv\\Scripts\\activate  # Windows")
        print("source .venv/bin/activate  # macOS/Linux")

        response = input("\nContinue anyway? (y/N): ").strip().lower()
        if response != 'y':
            return False

    # Install dependencies
    commands = [
        ("pip install --upgrade pip", "Upgrading pip"),
        ("pip install -r requirements.txt", "Installing main dependencies"),
        ("pip install scikit-image joblib", "Installing additional packages"),
    ]

    for command, description in commands:
        if not run_command(command, description):
            print(f"\n❌ Setup failed at: {description}")
            return False

    # Validate setup
    print("\n🔍 Validating setup...")
    if Path("validate_setup.py").exists():
        if not run_command("python validate_setup.py", "Running validation"):
            print("\n⚠️  Some validation checks failed, but you can still try the demo")

    # Check if models exist
    model_path = Path("models/advanced_shadow_puppet_classifier.joblib")
    if not model_path.exists():
        print("\n⚠️  Advanced model not found")
        print("The repository should include pre-trained models.")
        print("If missing, train with: python backend/data/advanced_train_classifier.py")
        return False

    # Success - ready to demo
    print("\n🎉 SETUP COMPLETE!")
    print("=" * 50)
    print("✅ All dependencies installed")
    print("✅ Advanced model available")
    print("✅ Ready for demo")

    print("\n🚀 LAUNCH OPTIONS:")
    print("1. Advanced Real-Time Demo (Recommended):")
    print("   python backend/cv_pipeline/advanced_realtime_inference.py")
    print("\n2. Validate everything works:")
    print("   python validate_setup.py")
    print("\n3. Test model performance:")
    print("   python test_model_accuracy.py")

    # Ask if user wants to launch demo
    print("\n" + "=" * 50)
    response = input("Launch advanced demo now? (Y/n): ").strip().lower()
    if response in ['', 'y', 'yes']:
        print("\n🎬 Launching Shadow-Vision Advanced Demo...")
        print("Controls: 'q' to quit, 's' to save frame, 'space' to pause")
        print("=" * 50)

        try:
            # Launch the demo
            subprocess.run([sys.executable, "backend/cv_pipeline/advanced_realtime_inference.py"])
        except KeyboardInterrupt:
            print("\n\n👋 Demo stopped by user")
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            return False

    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n📞 Need help? Check SETUP.md or create an issue on GitHub")
    sys.exit(0 if success else 1)