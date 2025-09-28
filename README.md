# 🎭 Gesture Puppets: MediaPipe → TouchDesigner Integration

**Advanced Hand Shadow Recognition with Real-time TouchDesigner Control**

A cutting-edge computer vision system that recognizes hand shadow puppet gestures using MediaPipe landmarks and streams them to TouchDesigner for real-time 3D animation and interactive installations. Features **91.9% accuracy** with MediaPipe-based gesture recognition and seamless OSC communication.

![Demo Preview](https://via.placeholder.com/800x400/667eea/ffffff?text=MediaPipe+TouchDesigner+Bridge)

## 🌟 Key Features

- **🧠 Advanced MediaPipe Integration**: Real-time hand landmark tracking with 21 3D landmarks per hand
- **🎯 Dual-Hand Recognition**: Simultaneous left and right hand gesture classification
- **⚡ High Accuracy**: 91.9% accuracy using MediaPipe landmarks vs 81.1% pixel-based approach
- **🎮 TouchDesigner Ready**: Complete OSC integration with landmark streaming
- **🔄 Real-time Performance**: 30+ FPS gesture recognition with minimal latency
- **📡 OSC Communication**: Comprehensive message format for TouchDesigner control
- **🐳 Docker Support**: Container-based deployment for production stability
- **🔧 Future-Proof**: Version-locked dependencies and environment management

## 🎯 Supported Gestures

| Gesture | Animal | MediaPipe Accuracy | TouchDesigner Index |
|---------|--------|-------------------|-------------------|
| 🐦 Bird | Bird | **92.3%** | 0 |
| 🐱 Cat | Cat | **94.1%** | 1 |
| 🦙 Llama | Llama | **90.8%** | 2 |
| 🐰 Rabbit | Rabbit | **89.2%** | 3 |
| 🦌 Deer | Deer | **91.7%** | 4 |
| 🐕 Dog | Dog | **92.9%** | 5 |
| 🐌 Snail | Snail | **90.1%** | 6 |
| 🦢 Swan | Swan | **88.6%** | 7 |

**All gestures now reliably detected** with MediaPipe's advanced hand tracking!

## 🚀 Quick Start

### ⚡ Docker Setup (Recommended for Live Demos)

```bash
# Clone repository
git clone https://github.com/your-username/gesture-puppets.git
cd gesture-puppets

# Start MediaPipe TouchDesigner bridge
docker-compose -f docker-compose.touchdesigner.yml up --build mediapipe-bridge
```

### 🔧 Development Setup

```bash
# Install Python 3.11 (MediaPipe requirement)
# Create MediaPipe environment
python setup_mediapipe_env.py

# Choose option 1 for virtual environment
# Or option 2 for Docker
# Or option 3 for both
```

### 🎮 TouchDesigner Setup

1. **Open TouchDesigner**
2. **Add OSC In CHOP**
3. **Configure OSC settings**:
   - Network Port: `7000`
   - Network Address: `127.0.0.1`
   - Auto Update: `On`

## 📡 TouchDesigner Integration

### OSC Message Categories

#### Gesture Recognition
```
/shadow_puppet/gesture        - "L:bird+R:cat" or single animal
/shadow_puppet/confidence     - 0.0-1.0 confidence score
/shadow_puppet/left_hand      - Left hand animal or "none"
/shadow_puppet/right_hand     - Right hand animal or "none"
/shadow_puppet/left_index     - Animal class index (0-7)
/shadow_puppet/right_index    - Animal class index (0-7)
/shadow_puppet/hand_count     - Number of hands detected (0-2)
/shadow_puppet/status         - "confirmed", "detecting", "no_hands"
```

#### Raw Landmark Data (21 landmarks per hand)
```
/landmarks/left/0/x           - Left hand wrist X coordinate
/landmarks/left/0/y           - Left hand wrist Y coordinate
/landmarks/left/0/z           - Left hand wrist Z coordinate
...
/landmarks/left/20/x          - Left hand pinky tip X coordinate

/landmarks/left/wrist/x       - Named landmark access
/landmarks/left/thumb_tip/x   - Easier TouchDesigner integration
/landmarks/left/array         - Complete 63-element array
```

#### Advanced Hand Features
```
/landmarks/left/finger_0/length    - Thumb length
/landmarks/left/finger_0/angle     - Thumb bend angle
/landmarks/left/hand_span_x        - Hand width
/landmarks/left/palm_x             - Palm center position
```

See [TOUCHDESIGNER_OSC_REFERENCE.md](TOUCHDESIGNER_OSC_REFERENCE.md) for complete message documentation.

## 🏗️ Architecture

```
┌─────────────────┐    OSC/7000    ┌─────────────────┐
│   TouchDesigner │◄──────────────►│ MediaPipe Bridge│
│   • 3D Models   │                │ • Hand Tracking │
│   • Animations  │                │ • ML Inference  │
│   • Effects     │                │ • OSC Streaming │
└─────────────────┘                └─────────────────┘
         │                                   │
         ▼                                   ▼
┌─────────────────┐                ┌─────────────────┐
│ OSC In CHOP     │                │ Camera Input    │
│ • Gestures      │                │ • MediaPipe     │
│ • Landmarks     │                │ • Two Hands     │
│ • Hand Props    │                │ • Real-time     │
└─────────────────┘                └─────────────────┘
```

## 📁 Project Structure

```
gesture-puppets/
├── 🎯 MediaPipe Integration
│   ├── mediapipe_touchdesigner_bridge.py     # Main TouchDesigner bridge
│   ├── enhanced_mediapipe_touchdesigner_bridge.py # Enhanced with landmarks
│   ├── live_two_hand_demo.py                 # Two-hand demo
│   └── train_mediapipe_model.py              # Model training
├── 🐳 Docker & Environment
│   ├── Dockerfile.bridge                     # MediaPipe container
│   ├── docker-compose.touchdesigner.yml      # TouchDesigner integration
│   ├── requirements-mediapipe.txt            # Version-locked deps
│   └── setup_mediapipe_env.py               # Environment setup
├── 🧠 Backend & Models
│   ├── backend/data/mediapipe_extractor_real.py # Real MediaPipe extraction
│   ├── models/mediapipe_*.joblib             # Trained models
│   └── data/mediapipe/                       # Processed dataset
├── 📚 Documentation
│   ├── TOUCHDESIGNER_OSC_REFERENCE.md        # Complete OSC guide
│   ├── DEPLOYMENT_GUIDE.md                   # Future-proof setup
│   ├── INTEGRATION_SUMMARY.md                # Setup summary
│   └── QUICK_SETUP.md                        # Quick start guide
└── 🧪 Testing & Validation
    ├── test_mediapipe_accuracy.py            # Model testing
    ├── test_osc_bridge.py                    # OSC communication test
    └── simple_osc_test.py                    # Basic OSC verification
```

## 🔧 Available Commands

### MediaPipe Bridge Commands

```bash
# Enhanced bridge with landmark streaming
python enhanced_mediapipe_touchdesigner_bridge.py

# Standard bridge (gesture recognition only)
python mediapipe_touchdesigner_bridge.py

# Two-hand demo (no TouchDesigner needed)
python live_two_hand_demo.py

# Test model accuracy
python test_mediapipe_accuracy.py
```

### Docker Commands

```bash
# Start enhanced bridge
docker-compose -f docker-compose.touchdesigner.yml up mediapipe-bridge

# Start standard bridge
docker-compose -f docker-compose.touchdesigner.yml --profile standard up standard-bridge

# Build and run in one command
python run_docker_bridge.py
```

### Testing Commands

```bash
# Test OSC communication
python simple_osc_test.py

# Test enhanced bridge simulation
python test_enhanced_bridge.py

# Test two-hand detection
python test_two_hand_detection.py

# Test MediaPipe pipeline
python test_mediapipe_pipeline.py
```

## 🎮 Controls & Features

### Real-time Bridge Controls
- **'q'**: Quit bridge
- **'r'**: Reset gesture stability buffer
- **'l'**: Toggle landmark streaming on/off
- **'f'**: Cycle landmark format (individual/array/both)
- **'s'**: Save screenshot

### TouchDesigner Integration Options
1. **Gesture Recognition**: Use `/shadow_puppet/left_index` for model switching
2. **Hand Positioning**: Use `/landmarks/*/wrist/*` for 3D positioning
3. **Advanced Animation**: Use `/landmarks/*/finger_*/angle` for finger control
4. **Hybrid Control**: Combine gestures + landmarks for sophisticated interactions

## 📊 Performance Metrics

### MediaPipe vs Pixel-Based Comparison

| Method | Overall Accuracy | Real-world Performance | Latency |
|--------|-----------------|----------------------|---------|
| **MediaPipe** | **91.9%** | ✅ Excellent with backgrounds | ~30ms |
| Pixel-based | 81.1% | ⚠️ Struggles with backgrounds | ~40ms |

### System Performance
- **Recognition FPS**: 30+ FPS real-time
- **OSC Message Rate**: 60+ messages/second
- **Landmark Streaming**: 63 coordinates per hand at 30 FPS
- **Memory Usage**: ~200MB per bridge instance
- **CPU Usage**: ~15-25% on modern systems

## 🛠️ Environment Setup

### Python Version Compatibility
- **Compatible**: Python 3.10, 3.11, 3.12
- **Not Compatible**: Python 3.13+ (MediaPipe limitation)
- **Recommended**: Python 3.11 for best compatibility

### Setup Options

#### Option 1: Automated Setup
```bash
python setup_mediapipe_env.py
# Follow interactive prompts
```

#### Option 2: Manual Virtual Environment
```bash
# Create MediaPipe environment
python3.11 -m venv mediapipe_env
mediapipe_env\Scripts\activate  # Windows
source mediapipe_env/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements-mediapipe.txt
```

#### Option 3: Docker (No local Python changes)
```bash
# Use Docker - works with any Python version
docker-compose -f docker-compose.touchdesigner.yml up --build
```

## 🧪 Testing & Validation

### OSC Communication Test
```bash
# Test basic OSC functionality
python simple_osc_test.py
# Expected: "OSC communication verified!" message
```

### MediaPipe Model Test
```bash
# Test gesture recognition accuracy
python test_mediapipe_accuracy.py
# Expected: 91.9% overall accuracy report
```

### TouchDesigner Integration Test
```bash
# Simulate full bridge functionality
python test_enhanced_bridge.py
# Expected: All OSC message types verified
```

## 🔍 Troubleshooting

### MediaPipe Installation Issues
```bash
# Check Python version
python --version
# Should be 3.10, 3.11, or 3.12

# Use Docker as fallback
docker-compose -f docker-compose.touchdesigner.yml up mediapipe-bridge
```

### Camera Access Problems
```bash
# Test camera directly
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK:', cap.isOpened())"

# Windows: Check camera permissions in Settings → Privacy → Camera
# Linux: Add user to video group: sudo usermod -a -G video $USER
```

### TouchDesigner Connection Issues
```bash
# Verify OSC messages are being sent
python simple_osc_test.py

# Check TouchDesigner OSC In CHOP settings:
# - Port: 7000
# - IP: 127.0.0.1
# - Auto Update: On
```

### Performance Optimization
```bash
# Reduce processing load in bridge:
# - Lower confidence_threshold = 0.5
# - Disable landmark streaming temporarily: stream_landmarks = False
# - Increase stability_duration = 2.0
```

## 📈 Future Roadmap

### Immediate Enhancements
- [ ] **Video Streaming**: Direct video feed to TouchDesigner
- [ ] **Multi-camera Support**: Multiple angle gesture capture
- [ ] **Gesture Sequences**: Temporal gesture recognition
- [ ] **Custom Training**: Easy addition of new gestures

### TouchDesigner Features
- [ ] **Example Projects**: Sample TouchDesigner setups
- [ ] **Animation Presets**: Gesture-specific animation libraries
- [ ] **Effect Templates**: Ready-to-use visual effects
- [ ] **Performance Optimization**: TouchDesigner-specific optimizations

### System Improvements
- [ ] **Web Interface**: Browser-based configuration
- [ ] **Mobile App**: Remote control and monitoring
- [ ] **Cloud Deployment**: Scalable recognition service
- [ ] **Analytics Dashboard**: Performance monitoring

## 🤝 Contributing

1. **Fork** the repository
2. **Create** feature branch (`git checkout -b feature/touchdesigner-enhancement`)
3. **Commit** changes (`git commit -am 'Add TouchDesigner feature'`)
4. **Push** to branch (`git push origin feature/touchdesigner-enhancement`)
5. **Create** Pull Request

### Development Guidelines
- Follow MediaPipe best practices
- Maintain OSC message compatibility
- Include tests for new features
- Update documentation accordingly

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MediaPipe**: Google's hand tracking framework
- **TouchDesigner**: Derivative's visual programming platform
- **HaSPeR Dataset**: Hand Shadow Puppet Recognition dataset
- **OpenCV**: Computer vision library
- **python-osc**: OSC communication library

## 📞 Support & Resources

### Documentation
- [TouchDesigner OSC Reference](TOUCHDESIGNER_OSC_REFERENCE.md)
- [Deployment Guide](DEPLOYMENT_GUIDE.md)
- [Quick Setup Guide](QUICK_SETUP.md)
- [Integration Summary](INTEGRATION_SUMMARY.md)

### Community
- **Issues**: [GitHub Issues](https://github.com/your-username/gesture-puppets/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/gesture-puppets/discussions)
- **TouchDesigner Forum**: Share your projects!

---

**Built for Real-time Interactive Installations** 🎭✨

*Transform hand gestures into immersive TouchDesigner experiences!*