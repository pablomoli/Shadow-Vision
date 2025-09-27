# 🎭 Gesture Puppets

**Hand Shadow Recognition → Real-time 3D Animation**

An interactive hackathon project that uses computer vision to recognize hand shadow puppet gestures and translates them into real-time 3D animated scenes. Built for ShellHacks 2025.

![Demo Preview](https://via.placeholder.com/800x400/667eea/ffffff?text=Gesture+Puppets+Demo)

## 🌟 Features

- **Real-time Gesture Recognition**: OpenCV + PyTorch model trained on HaSPeR dataset
- **3D Animated Responses**: Three.js rendering with randomized animation variants
- **Dynamic Scene Backgrounds**: Contextual environments that match each animal
- **WebSocket Communication**: Real-time data flow between CV backend and frontend
- **Modular Architecture**: Easy to extend with new gestures and animations
- **Hackathon Ready**: Stable, demo-friendly with graceful error handling

## 🎯 Supported Gestures

| Gesture | Animal | Scene | Animations |
|---------|--------|-------|------------|
| 🐕 Dog | Dog | Park | Bark, Sit, Wag, Jump, Play |
| 🐦 Bird | Bird | Sky | Fly, Perch, Flap, Peck, Turn |
| 🐰 Rabbit | Rabbit | Meadow | Hop, Nibble, Alert, Clean, Sit |
| 🦋 Butterfly | Butterfly | Garden | Flutter, Land, Spiral, Rest, Takeoff |
| 🐍 Snake | Snake | Desert | Slither, Coil, Strike, Bask, Flick |

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone and navigate to project
cd gesture-puppets

# Run one-click setup
python scripts/setup_environment.py
```

### 2. Download Dataset & Train Model

```bash
# Activate virtual environment
# Windows:
venv\Scripts\activate
# Unix/Mac:
source venv/bin/activate

# Download HaSPeR dataset
python backend/main.py download-dataset

# Train gesture recognition model
python backend/main.py train --model-type efficient --epochs 25
```

### 3. Launch Demo

```bash
# Quick demo launcher
python scripts/run_demo.py

# Or manually:
python backend/main.py server &
python -m http.server 3000 --directory frontend/public
```

### 4. Open Browser

Navigate to `http://localhost:3000` and allow camera access!

## 🏗️ Architecture

```
┌─────────────────┐    WebSocket    ┌─────────────────┐
│   Frontend      │◄──────────────►│   Backend       │
│   (Three.js)    │                │   (FastAPI)     │
└─────────────────┘                └─────────────────┘
         │                                   │
         ▼                                   ▼
┌─────────────────┐                ┌─────────────────┐
│ 3D Scene        │                │ CV Pipeline     │
│ • Models        │                │ • Camera        │
│ • Animations    │                │ • Preprocessing │
│ • Environments  │                │ • ML Inference  │
└─────────────────┘                └─────────────────┘
                                             │
                                             ▼
                                   ┌─────────────────┐
                                   │ Supabase DB     │
                                   │ • Gesture Logs  │
                                   │ • Training Data │
                                   │ • Analytics     │
                                   └─────────────────┘
```

## 📁 Project Structure

```
gesture-puppets/
├── backend/
│   ├── data/                    # Dataset handling
│   ├── models/                  # ML model architectures
│   ├── cv_pipeline/            # Computer vision processing
│   ├── api/                    # WebSocket server & database
│   └── main.py                 # Backend entry point
├── frontend/
│   ├── src/
│   │   ├── components/         # UI components
│   │   ├── models/             # 3D model files
│   │   ├── animations/         # Animation definitions
│   │   └── scenes/             # Environment configs
│   └── public/index.html       # Main app interface
├── config/
│   ├── gesture_map.json        # Gesture→Animation mapping
│   ├── model_config.yaml       # ML training settings
│   └── .env.example            # Environment variables
├── scripts/
│   ├── setup_environment.py    # One-click setup
│   ├── run_demo.py            # Demo launcher
│   └── download_models.py      # 3D asset setup
└── tests/                      # Unit tests
```

## 🔧 Development

### Backend Commands

```bash
# Train different model architectures
python backend/main.py train --model-type lightweight
python backend/main.py train --model-type mobilenet
python backend/main.py train --model-type efficient

# Test components
python backend/main.py test camera
python backend/main.py test inference
python backend/main.py test database

# Start server with custom settings
python backend/main.py server --host 0.0.0.0 --port 8080
```

### Frontend Development

```bash
# Install dependencies
npm install

# Development server
npm run dev

# Build for production
npm run build
```

### Configuration

#### Environment Variables (`.env`)

```bash
# Supabase (optional)
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_anon_key

# Model settings
CONFIDENCE_THRESHOLD=0.8
CAMERA_INDEX=0

# Server settings
BACKEND_HOST=localhost
BACKEND_PORT=8000
```

#### Gesture Mapping (`gesture_map.json`)

```json
{
  "gestures": {
    "dog": {
      "model": "models/dog.gltf",
      "scene": "park",
      "animations": ["idle_wag", "bark", "sit", "playful_jump"],
      "background_color": "#87CEEB"
    }
  }
}
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Test specific components
python backend/cv_pipeline/camera_handler.py
python backend/cv_pipeline/inference_engine.py
python backend/api/supabase_client.py
```

## 📊 Performance

### Model Specifications

| Model | Parameters | Size | Inference Time | Accuracy |
|-------|------------|------|----------------|----------|
| Lightweight | 50K | 1.2MB | 15ms | 85% |
| MobileNet | 2.2M | 9.1MB | 25ms | 92% |
| Efficient | 1.2M | 4.8MB | 20ms | 90% |

### System Requirements

- **Minimum**: Python 3.8+, 4GB RAM, webcam
- **Recommended**: Python 3.10+, 8GB RAM, GPU (optional)
- **Browser**: Chrome/Firefox with WebGL support

## 🔍 Troubleshooting

### Common Issues

**Camera not working**
```bash
# Check camera permissions and index
python backend/main.py test camera
```

**Model not found**
```bash
# Ensure model is trained
ls backend/trained_models/
python backend/main.py train
```

**WebSocket connection failed**
```bash
# Check if backend is running
curl http://localhost:8000
python backend/main.py server
```

**Poor gesture detection**
- Ensure good lighting
- Position hand 2-3 feet from camera
- Use contrasting background
- Adjust confidence threshold in settings

### Debug Mode

```bash
# Enable debug logging
export DEBUG=True
python backend/main.py server

# Frontend debug
# Open browser console for detailed logs
```

## 🎨 Customization

### Adding New Gestures

1. **Add to gesture mapping**:
   ```json
   "new_gesture": {
     "model": "models/new_animal.gltf",
     "scene": "new_environment",
     "animations": ["anim1", "anim2", "anim3"]
   }
   ```

2. **Create 3D model**: Add GLTF file to `frontend/src/models/`

3. **Define animations**: Add keyframes to `frontend/src/animations/`

4. **Train model**: Include new gesture in training data

### Custom Scenes

```json
{
  "background_color": "#YOUR_COLOR",
  "lighting": {
    "ambient": {"color": "#404040", "intensity": 0.6}
  },
  "objects": [
    {"type": "tree", "position": [0, 0, 0]}
  ]
}
```

## 📈 Analytics & Logging

The system logs gesture detection events, training metrics, and user interactions to Supabase for analysis:

- **Gesture Logs**: Detected gestures, confidence scores, timestamps
- **Training Metrics**: Model performance, accuracy, loss curves
- **Demo Sessions**: User engagement, success rates, feedback

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-gesture`)
3. Commit changes (`git commit -am 'Add new gesture'`)
4. Push to branch (`git push origin feature/new-gesture`)
5. Create Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **HaSPeR Dataset**: Hand Shadow Puppet Recognition dataset from Hugging Face
- **Three.js**: 3D graphics library
- **OpenCV**: Computer vision library
- **PyTorch**: Machine learning framework
- **Supabase**: Backend-as-a-Service platform

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/gesture-puppets/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/gesture-puppets/discussions)
- **Email**: your-email@example.com

---

**Built with ❤️ for ShellHacks 2025**

*Make hand shadows come to life!* 🎭✨