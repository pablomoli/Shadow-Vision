# MediaPipe TouchDesigner Bridge - Future-Proof Deployment Guide

## 🎯 Problem Solved
This guide ensures your MediaPipe bridges work reliably regardless of:
- Python version updates
- MediaPipe compatibility changes
- System environment differences
- Future dependency conflicts

## 🚀 Quick Start Options

### Option 1: Virtual Environment (Development)
```bash
# Run the automated setup
python setup_mediapipe_env.py

# Choose option 1 for virtual environment
# This creates a locked environment with MediaPipe
```

### Option 2: Docker (Production/Demos)
```bash
# Run the automated setup
python setup_mediapipe_env.py

# Choose option 2 for Docker
# Or manually:
docker-compose -f docker-compose.touchdesigner.yml up mediapipe-bridge
```

### Option 3: Both (Recommended)
```bash
# Run the automated setup
python setup_mediapipe_env.py

# Choose option 3 for both environments
```

## 📋 What Gets Created

### Virtual Environment Setup
- `mediapipe_env/` - Isolated Python 3.11 environment
- `requirements-mediapipe.txt` - Version-locked dependencies
- `activate_mediapipe.bat` - Windows activation script
- `run_mediapipe_bridge.py` - Cross-platform runner

### Docker Setup
- `Dockerfile.bridge` - MediaPipe bridge container
- `docker-compose.touchdesigner.yml` - TouchDesigner integration
- `run_docker_bridge.py` - Docker runner script

## 🔧 Version Management Strategy

### Locked Dependencies
```
mediapipe==0.10.9        # Stable version
opencv-python==4.8.1.78  # Compatible with MediaPipe
python-osc==1.9.3        # TouchDesigner communication
numpy==1.24.3            # Stable NumPy
scikit-learn==1.3.2      # ML model compatibility
```

### Python Version Strategy
- **Development**: Virtual environment with Python 3.11
- **Production**: Docker container with Python 3.11
- **Future**: Easy upgrade path when MediaPipe supports 3.13+

## 🎮 Running Your Bridges

### Method 1: Virtual Environment
```bash
# Activate environment
activate_mediapipe.bat  # Windows
source mediapipe_env/bin/activate  # Linux/Mac

# Run enhanced bridge
python enhanced_mediapipe_touchdesigner_bridge.py

# Or run via wrapper
python run_mediapipe_bridge.py enhanced_mediapipe_touchdesigner_bridge.py
```

### Method 2: Docker
```bash
# Start enhanced bridge
docker-compose -f docker-compose.touchdesigner.yml up mediapipe-bridge

# Or use wrapper script
python run_docker_bridge.py

# Start standard bridge (alternative)
docker-compose -f docker-compose.touchdesigner.yml --profile standard up standard-bridge
```

## 🧪 Testing Setup

### 1. Test Environment
```bash
# Virtual environment test
mediapipe_env/Scripts/python -c "import mediapipe; print('OK:', mediapipe.__version__)"

# Docker test
docker run --rm mediapipe_touchdesigner_bridge python -c "import mediapipe; print('OK:', mediapipe.__version__)"
```

### 2. Test OSC Communication
```bash
# Test OSC without camera
python simple_osc_test.py

# Test enhanced bridge simulation
python test_enhanced_bridge.py
```

### 3. Test Full Bridge
```bash
# Virtual environment
python run_mediapipe_bridge.py mediapipe_touchdesigner_bridge.py

# Docker
docker-compose -f docker-compose.touchdesigner.yml up mediapipe-bridge
```

## 🔄 Future Upgrade Strategy

### When MediaPipe Supports Python 3.13+
1. **Update requirements-mediapipe.txt**:
   ```
   mediapipe==0.11.0  # New version
   ```

2. **Test in virtual environment first**:
   ```bash
   # Create test environment
   python3.13 -m venv test_env
   test_env/Scripts/activate
   pip install -r requirements-mediapipe.txt
   ```

3. **Update Docker when ready**:
   ```dockerfile
   FROM python:3.13-slim
   ```

4. **Gradual migration**:
   - Keep old environment as backup
   - Test thoroughly before switching
   - Use Docker for production stability

### Dependency Updates
1. **Always test in isolated environment first**
2. **Update one dependency at a time**
3. **Keep working versions locked**
4. **Use Docker for consistent deployment**

## 🛠️ Troubleshooting

### MediaPipe Won't Install
```bash
# Check Python version
python --version  # Should be 3.10, 3.11, or 3.12

# Try specific MediaPipe version
pip install mediapipe==0.10.9

# Use Docker as fallback
docker-compose -f docker-compose.touchdesigner.yml up mediapipe-bridge
```

### Camera Access Issues
```bash
# Windows: Check camera permissions
# Linux: Add user to video group
sudo usermod -a -G video $USER

# Docker: Enable camera device
# Uncomment device mapping in docker-compose.touchdesigner.yml
```

### OSC Communication Problems
```bash
# Test OSC connection
python simple_osc_test.py

# Check TouchDesigner OSC In CHOP:
# - Port: 7000
# - IP: 127.0.0.1
# - Auto Update: On
```

### Performance Issues
```bash
# Reduce landmark streaming frequency
# In enhanced bridge, modify:
# - stream_landmarks = False  # Disable temporarily
# - Lower confidence_threshold = 0.5
# - Increase stability_duration = 2.0
```

## 📊 Environment Comparison

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **Virtual Env** | Fast setup, easy debugging | Version conflicts possible | Development |
| **Docker** | Completely isolated, reproducible | Larger overhead | Production/Demos |
| **Both** | Maximum flexibility | More complex setup | Professional use |

## 🎯 Recommended Workflow

### Development
1. Use virtual environment for fast iteration
2. Test changes quickly
3. Debug easily with IDE integration

### Live Demos
1. Use Docker for reliability
2. No dependency surprises
3. Consistent performance

### Production
1. Docker with health checks
2. Automatic restart on failure
3. Monitoring and logging

## 🔒 Backup Strategy

### Configuration Backup
- `requirements-mediapipe.txt` - Exact dependencies
- `docker-compose.touchdesigner.yml` - Container config
- `enhanced_mediapipe_touchdesigner_bridge.py` - Working bridge

### Model Backup
- `models/` directory - Trained models
- `config/` directory - Configuration files
- Environment setup scripts

## ✅ Success Checklist

- [ ] **Environment created**: Virtual env or Docker working
- [ ] **MediaPipe installed**: Import successful
- [ ] **OSC tested**: Messages sending to TouchDesigner
- [ ] **Camera working**: Video feed displaying
- [ ] **Models loaded**: Gesture recognition functional
- [ ] **Bridge running**: TouchDesigner receiving data
- [ ] **Backup created**: Configuration saved

## 🎉 You're Future-Proof!

With this setup, your MediaPipe bridges will:
- ✅ Work reliably across different systems
- ✅ Survive Python/MediaPipe updates
- ✅ Provide consistent live demo performance
- ✅ Allow easy troubleshooting and maintenance
- ✅ Scale from development to production seamlessly