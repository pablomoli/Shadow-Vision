# 🐳 Shadow-Vision Docker Deployment Guide

Complete instructions for running Shadow-Vision in Docker containers across different platforms.

## 🚀 Quick Start

### Option 1: Headless Demo (No GUI)
```bash
# Clone repository
git clone https://github.com/pablomoli/Shadow-Vision.git
cd Shadow-Vision

# Run headless demo
docker-compose up --build shadow-vision
```

### Option 2: GUI Demo (with X11 forwarding on Linux)
```bash
# Enable X11 forwarding
xhost +local:docker

# Set display
export DISPLAY=:0

# Run with GUI
docker-compose up --build shadow-vision
```

## 📋 Platform-Specific Setup

### 🐧 Linux (Recommended)

#### Prerequisites
```bash
# Install Docker and Docker Compose
sudo apt update
sudo apt install docker.io docker-compose

# Add user to docker group
sudo usermod -aG docker $USER
# Log out and back in, or run: newgrp docker
```

#### Camera Access
```bash
# Check camera device
ls /dev/video*

# Run with camera access
docker-compose up --build
```

#### GUI Support (Optional)
```bash
# Allow X11 forwarding
xhost +local:docker

# Set display variable
export DISPLAY=:0

# Run with GUI
docker-compose up --build shadow-vision
```

### 🍎 macOS

#### Prerequisites
```bash
# Install Docker Desktop for Mac
# Download from: https://docs.docker.com/desktop/mac/install/

# Or with Homebrew
brew install --cask docker
```

#### Camera Access
```bash
# Docker Desktop needs camera permission
# System Preferences → Security & Privacy → Camera → Docker Desktop

# Run container
docker-compose up --build shadow-vision
```

#### GUI Support
```bash
# Install XQuartz for X11 forwarding
brew install --cask xquartz

# Start XQuartz and enable "Allow connections from network clients"
# Applications → Utilities → XQuartz → Preferences → Security

# Set display
export DISPLAY=host.docker.internal:0

# Run with GUI
docker-compose up --build shadow-vision
```

### 🪟 Windows

#### Prerequisites
```bash
# Install Docker Desktop for Windows
# Download from: https://docs.docker.com/desktop/windows/install/

# Or with Chocolatey
choco install docker-desktop
```

#### Camera Access
```bash
# Docker Desktop needs camera permission
# Settings → Privacy → Camera → Allow apps to access camera

# Run container (camera access limited on Windows)
docker-compose up --build shadow-vision
```

#### GUI Support (Limited)
```bash
# Windows containers don't support GUI well
# Recommend using WSL2 with Linux approach instead

# Alternative: Use VcXsrv or similar X11 server
# Download VcXsrv and configure for network access
```

## 🔧 Configuration Options

### Environment Variables

Create `.env` file in project root:
```bash
# Display settings
DISPLAY=:0

# Camera settings
CAMERA_INDEX=0

# Demo settings
MAX_DEMO_FRAMES=1000
CONFIDENCE_THRESHOLD=0.7

# Logging
LOG_LEVEL=INFO
```

### Camera Configuration

#### Linux
```yaml
# docker-compose.yml
devices:
  - /dev/video0:/dev/video0  # Default camera
  - /dev/video1:/dev/video1  # Secondary camera
```

#### macOS/Windows
```yaml
# Camera access through Docker Desktop
# No device mapping needed, but requires permissions
```

## 🧪 Testing Docker Setup

### 1. Build Test
```bash
# Test if container builds successfully
docker-compose build shadow-vision
```

### 2. Camera Test
```bash
# Test camera access
docker-compose run --rm shadow-vision python -c "
import cv2
cap = cv2.VideoCapture(0)
print('Camera working:', cap.isOpened())
cap.release()
"
```

### 3. Model Test
```bash
# Test model loading
docker-compose run --rm shadow-vision python -c "
import joblib
model = joblib.load('models/advanced_shadow_puppet_classifier.joblib')
print('Model loaded successfully')
"
```

### 4. Full Demo Test
```bash
# Run short demo
docker-compose run --rm shadow-vision python docker_demo.py
```

## 📊 Demo Modes

### Headless Mode (Default)
- No GUI window
- Processes frames and logs predictions
- Saves results to `/app/demo_results.json`
- Perfect for servers and testing

### GUI Mode (X11 Required)
- Shows camera feed with predictions
- Interactive controls ('q' to quit, 's' to save)
- Requires X11 forwarding setup

## 🐛 Troubleshooting

### Common Issues

#### Camera Not Working
```bash
# Check camera permissions
ls -l /dev/video*

# Test camera outside Docker
v4l2-ctl --list-devices

# Run with privileged mode
docker-compose run --privileged shadow-vision
```

#### GUI Not Showing
```bash
# Check X11 forwarding
echo $DISPLAY
xhost +local:docker

# Test X11 connection
docker run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix alpine sh -c "apk add --no-cache xeyes && xeyes"
```

#### Model Not Found
```bash
# Ensure models are included in build
docker-compose build --no-cache shadow-vision

# Check models in container
docker-compose run --rm shadow-vision ls -la models/
```

#### Performance Issues
```bash
# Increase container resources
# Docker Desktop: Settings → Resources → Advanced

# Use lighter demo mode
docker-compose run shadow-vision python docker_demo.py --max-frames 50
```

### Platform-Specific Issues

#### Linux
```bash
# SELinux issues
sudo setsebool -P container_use_devices 1

# AppArmor issues
sudo aa-disable docker-default
```

#### macOS
```bash
# XQuartz connection issues
export DISPLAY=host.docker.internal:0
/opt/X11/bin/xhost +

# Camera permission issues
# System Preferences → Security & Privacy → Camera → Docker Desktop
```

#### Windows
```bash
# WSL2 recommended for better compatibility
wsl --install
# Then follow Linux instructions inside WSL2
```

## 🚀 Production Deployment

### Cloud Deployment
```bash
# AWS/Azure/GCP with GPU support
docker run --gpus all -p 8000:8000 shadow-vision

# Kubernetes deployment
kubectl apply -f k8s/shadow-vision-deployment.yaml
```

### Performance Optimization
```bash
# Use multi-stage builds for smaller images
# Optimize for specific architectures
docker buildx build --platform linux/amd64,linux/arm64 .
```

## 📈 Monitoring

### Container Health
```bash
# Check container health
docker-compose ps

# View logs
docker-compose logs shadow-vision

# Monitor resources
docker stats
```

### Demo Results
```bash
# Access demo results
docker-compose exec shadow-vision cat /app/demo_results.json

# Copy results to host
docker cp $(docker-compose ps -q shadow-vision):/app/demo_results.json ./
```

## 🔄 Updates

### Update Container
```bash
# Pull latest code
git pull origin master

# Rebuild container
docker-compose build --no-cache shadow-vision

# Restart with new version
docker-compose up --build shadow-vision
```

## 📞 Support

- **Container Issues**: Check logs with `docker-compose logs`
- **Camera Issues**: Verify device permissions
- **GUI Issues**: Confirm X11 forwarding setup
- **Performance Issues**: Monitor with `docker stats`

---

**Ready for deployment across any platform!** 🏆