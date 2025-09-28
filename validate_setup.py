#!/usr/bin/env python3
"""
Shadow-Vision Setup Validation Script

Run this script after cloning the repository to ensure everything is set up correctly.
"""

import sys
import importlib
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def check_dependencies():
    """Check if all required dependencies are installed."""
    print("\n📦 Checking dependencies...")

    required_packages = [
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'),
        ('sklearn', 'scikit-learn'),
        ('joblib', 'joblib'),
        ('skimage', 'scikit-image'),
        ('torch', 'torch'),
        ('PIL', 'Pillow')
    ]

    missing_packages = []

    for import_name, package_name in required_packages:
        try:
            importlib.import_module(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} - Missing")
            missing_packages.append(package_name)

    if missing_packages:
        print(f"\n⚠️  Install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    return True

def check_camera():
    """Check if camera is accessible."""
    print("\n📹 Checking camera access...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"✅ Camera working - Frame size: {frame.shape}")
                cap.release()
                return True
            else:
                print("❌ Camera opened but no frame captured")
                cap.release()
                return False
        else:
            print("❌ Cannot open camera")
            return False
    except Exception as e:
        print(f"❌ Camera error: {e}")
        return False

def check_models():
    """Check if trained models are available."""
    print("\n🧠 Checking trained models...")

    model_files = [
        'models/advanced_shadow_puppet_classifier.joblib',
        'models/advanced_training_metrics.json',
        'models/advanced_model_report.txt'
    ]

    all_models_present = True

    for model_file in model_files:
        if Path(model_file).exists():
            size_mb = Path(model_file).stat().st_size / (1024 * 1024)
            print(f"✅ {model_file} ({size_mb:.1f} MB)")
        else:
            print(f"❌ {model_file} - Missing")
            all_models_present = False

    if not all_models_present:
        print("\n⚠️  Train models with:")
        print("python backend/data/advanced_train_classifier.py")

    return all_models_present

def check_dataset():
    """Check if dataset is available."""
    print("\n📊 Checking dataset...")

    data_paths = [
        'data/splits/train.csv',
        'data/splits/val.csv',
        'data/splits/train_advanced_features.npz',
        'data/splits/val_advanced_features.npz'
    ]

    dataset_present = True

    for data_path in data_paths:
        if Path(data_path).exists():
            size_mb = Path(data_path).stat().st_size / (1024 * 1024)
            print(f"✅ {data_path} ({size_mb:.1f} MB)")
        else:
            print(f"❌ {data_path} - Missing")
            dataset_present = False

    if not dataset_present:
        print("\n⚠️  Download dataset with:")
        print("python backend/data/dataset.py")

    return dataset_present

def test_inference():
    """Test if inference system can be imported."""
    print("\n🚀 Testing inference system...")
    try:
        # Test basic imports
        from backend.data.advanced_feature_extractor import AdvancedFeatureExtractor
        from backend.cv_pipeline.advanced_realtime_inference import AdvancedRealTimeClassifier
        print("✅ Inference modules import successfully")

        # Test model loading
        import joblib
        model_path = 'models/advanced_shadow_puppet_classifier.joblib'
        if Path(model_path).exists():
            model = joblib.load(model_path)
            print("✅ Advanced model loads successfully")
            return True
        else:
            print("❌ Cannot load advanced model")
            return False

    except Exception as e:
        print(f"❌ Inference system error: {e}")
        return False

def main():
    """Main validation function."""
    print("🎭 Shadow-Vision Setup Validation")
    print("=" * 50)

    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Camera Access", check_camera),
        ("Trained Models", check_models),
        ("Dataset", check_dataset),
        ("Inference System", test_inference)
    ]

    results = []

    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ {check_name} failed with error: {e}")
            results.append((check_name, False))

    # Summary
    print("\n" + "=" * 50)
    print("📋 VALIDATION SUMMARY")
    print("=" * 50)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name:20} {status}")

    print(f"\nOverall: {passed}/{total} checks passed")

    if passed == total:
        print("\n🎉 ALL CHECKS PASSED!")
        print("Ready to run: python backend/cv_pipeline/advanced_realtime_inference.py")
        return True
    else:
        print(f"\n⚠️  {total - passed} issues found. See instructions above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)