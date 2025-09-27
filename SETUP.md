# 🚀 Shadow-Vision Setup Guide

Complete setup instructions for running the advanced gesture recognition system.

## 📋 Prerequisites

### System Requirements
- **Python 3.8+** (recommended: Python 3.10)
- **4GB RAM minimum** (8GB recommended)
- **Webcam** (for real-time demo)
- **Git** for cloning

### Platform Support
- ✅ **Windows 10/11**
- ✅ **macOS 10.15+**
- ✅ **Ubuntu 18.04+**

## 🎯 Quick Start (Recommended)

### Option 1: Direct Python Setup

```bash
# 1. Clone the repository
git clone https://github.com/pablomoli/Shadow-Vision.git
cd Shadow-Vision

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
pip install scikit-image joblib

# 5. Run the advanced demo (models already trained!)
python backend/cv_pipeline/advanced_realtime_inference.py
```

**That's it!** The advanced model is already trained and ready to use.

### Option 2: Docker Setup (Alternative)

```bash
# 1. Clone the repository
git clone https://github.com/pablomoli/Shadow-Vision.git
cd Shadow-Vision

# 2. Run with Docker Compose
docker-compose up --build

# 3. Access the application
# Backend: http://localhost:8000
# Frontend: http://localhost:3000
```

## 🔧 Detailed Setup Steps

### 1. Environment Setup

#### Windows
```bash
# Install Python 3.10 from python.org
# Clone repository
git clone https://github.com/pablomoli/Shadow-Vision.git
cd Shadow-Vision

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install scikit-image joblib
```

#### macOS
```bash
# Install Python via Homebrew (recommended)
brew install python@3.10

# Clone repository
git clone https://github.com/pablomoli/Shadow-Vision.git
cd Shadow-Vision

# Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install scikit-image joblib
```

#### Ubuntu/Linux
```bash
# Install Python and required packages
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip git

# Clone repository
git clone https://github.com/pablomoli/Shadow-Vision.git
cd Shadow-Vision

# Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install scikit-image joblib
```

### 2. Verify Installation

```bash
# Test camera access
python -c "import cv2; print('OpenCV:', cv2.__version__)"

# Test model loading
python -c "import joblib; print('Models ready:', len(list(joblib.os.listdir('models'))))"

# Quick system test
python test_model_accuracy.py
```

### 3. Run Different Components

#### Advanced Real-Time Demo (Main)
```bash
python backend/cv_pipeline/advanced_realtime_inference.py
```

#### Model Training (if you want to retrain)
```bash
# Download dataset (if not already present)
python backend/data/dataset.py

# Train advanced model (takes 10-15 minutes)
python backend/data/advanced_train_classifier.py
```

#### Performance Testing
```bash
# Test specific gesture classes
python test_model_accuracy.py

# View training metrics
cat models/advanced_model_report.txt
```

## 🐛 Troubleshooting

### Common Issues

#### "Module not found" errors
```bash
# Ensure virtual environment is activated
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
pip install scikit-image joblib
```

#### Camera permission denied
- **Windows**: Check Camera settings in Windows Privacy settings
- **macOS**: System Preferences → Security & Privacy → Camera
- **Linux**: Add user to video group: `sudo usermod -a -G video $USER`

#### OpenCV issues
```bash
# Try alternative OpenCV installation
pip uninstall opencv-python
pip install opencv-python-headless
```

#### Model file missing
```bash
# Check if models exist
ls models/
# Should see: advanced_shadow_puppet_classifier.joblib

# If missing, retrain:
python backend/data/advanced_train_classifier.py
```

### Performance Issues

#### Slow inference
- Ensure no other applications using camera
- Close unnecessary programs
- Try lower resolution: modify camera settings in code

#### Low accuracy
- Ensure good lighting conditions
- Position hand 2-3 feet from camera
- Use contrasting background
- Check if correct model is loaded

## 🎮 Demo Controls

When running the real-time demo:
- **'q'**: Quit application
- **'s'**: Save current frame
- **'space'**: Pause/resume processing
- **ESC**: Exit application

## 📊 Expected Performance

### System Performance
- **Inference Time**: ~30ms per frame
- **FPS**: 25-30 FPS real-time
- **Accuracy**: 81.1% overall
- **Memory Usage**: ~2GB during inference

### Gesture Recognition
- **Excellent (>85%)**: Bird, Cat, Deer
- **Good (70-85%)**: Llama, Dog, Snail, Swan
- **Acceptable (>60%)**: Rabbit

## 🔄 Updates and Maintenance

### Keeping Updated
```bash
# Pull latest changes
git pull origin master

# Update dependencies
pip install --upgrade -r requirements.txt
```

### Retraining Models
```bash
# Full pipeline retraining
python backend/data/dataset.py
python backend/data/advanced_train_classifier.py
```

## 📞 Support

### If You Need Help
1. **Check this guide** for common solutions
2. **View model metrics**: `cat models/advanced_model_report.txt`
3. **Check GitHub Issues**: [Shadow-Vision Issues](https://github.com/pablomoli/Shadow-Vision/issues)
4. **Create new issue** with:
   - Operating system
   - Python version
   - Error message
   - Steps to reproduce

## ✨ Features Ready Out-of-the-Box

- ✅ **Pre-trained advanced model** (81.1% accuracy)
- ✅ **Real-time inference system**
- ✅ **Advanced hand detection**
- ✅ **Background robustness**
- ✅ **Ensemble voting classifier**
- ✅ **Complete documentation**

**Ready for ShellHacks 2025 presentations!** 🏆