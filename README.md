# 🎭 Shadow-Vision: Advanced Gesture Puppets

**Hand Shadow Recognition → Real-time 3D Animation**

An advanced computer vision system for ShellHacks 2025 that recognizes hand shadow puppet gestures with **81.1% accuracy** and translates them into real-time 3D animated scenes. Features cutting-edge ML ensemble models and robust real-world performance.

![Demo Preview](https://via.placeholder.com/800x400/667eea/ffffff?text=Shadow-Vision+Advanced+Demo)

## 🌟 Advanced Features

- **🧠 Advanced ML Pipeline**: Ensemble classifier (k-NN + Random Forest + SVM) with 49 comprehensive features
- **🌍 Real-World Robustness**: Works with complex backgrounds and human bodies (perfect for judge demos)
- **⚡ Enhanced Accuracy**: 81.1% overall accuracy with significant improvements for minority classes
- **🎯 Class Balance**: SMOTE technique addresses data imbalance for reliable detection
- **🔧 Sophisticated Feature Engineering**: Texture analysis, edge detection, gradient features, Hu moments
- **📱 Real-time Inference**: Optimized for live demonstrations with prediction smoothing
- **🎮 3D Animated Responses**: Three.js rendering with randomized animation variants
- **🌐 WebSocket Communication**: Real-time data flow between CV backend and frontend
- **🏗️ Modular Architecture**: Easy to extend with new gestures and animations

## 🎯 Supported Gestures (HuggingFace Dataset)

| Gesture | Animal | F1-Score | Previous Issues | Current Status |
|---------|--------|----------|-----------------|----------------|
| 🐦 Bird | Bird | **86.1%** | Good detection | ✅ Excellent |
| 🐱 Cat | Cat | **95.8%** | Good detection | ✅ Outstanding |
| 🦙 Llama | Llama | **82.5%** | False positives | ✅ Improved |
| 🐰 Rabbit | Rabbit | **63.9%** | Minority class | ⚠️ Acceptable |
| 🦌 Deer | Deer | **83.2%** | Good detection | ✅ Excellent |
| 🐕 Dog | Dog | **74.8%** | Confused with llama | ✅ Fixed |
| 🐌 Snail | Snail | **80.0%** | **0% before** | ✅ **FIXED** |
| 🦢 Swan | Swan | **70.5%** | **0% before** | ✅ **FIXED** |

**Key Improvements**: Snail and swan now properly detected! Dog vs llama confusion resolved!

## 🚀 Quick Start (Advanced System)

### ⚡ One-Click Setup (Recommended)

```bash
# Clone and navigate to project
git clone https://github.com/pablomoli/Shadow-Vision.git
cd Shadow-Vision

# One-click setup and demo launcher
python quick_start.py
```

### 🔧 Manual Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Unix/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install scikit-image joblib

# Validate setup
python validate_setup.py
```

### 🐳 Docker Setup

```bash
# Run with Docker (alternative)
docker-compose up --build
```

### 2. Advanced Model Training (Already Completed!)

The advanced ensemble model is **already trained** with 81.1% accuracy! If you want to retrain:

```bash
# Download and organize HuggingFace dataset
python backend/data/dataset.py

# Train advanced ensemble model (takes 10-15 minutes)
python backend/data/advanced_train_classifier.py

# View training results
cat models/advanced_model_report.txt
```

### 3. Run Advanced Real-Time Demo

```bash
# Launch advanced real-time gesture recognition
python backend/cv_pipeline/advanced_realtime_inference.py
```

**Features during demo:**
- Real-time gesture recognition with ensemble voting
- Confidence scores and prediction quality indicators
- Works with complex backgrounds and human bodies
- Press 'q' to quit, 's' to save frame, 'space' to pause

### 4. Test Model Performance

```bash
# Test advanced model on specific problematic classes
python test_model_accuracy.py
```

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

## 📊 Advanced Performance Metrics

### Model Evolution

| Model | Accuracy | Key Features | Status |
|-------|----------|--------------|--------|
| **Basic k-NN** | 88.3% | 21 geometric features | ⚠️ Failed real-world conditions |
| **Advanced Ensemble** | **81.1%** | 49 comprehensive features + SMOTE + ensemble | ✅ **Production Ready** |

### Advanced Model Specifications

- **Architecture**: k-NN + Random Forest + SVM ensemble with soft voting
- **Features**: 49 comprehensive features (geometric, texture, edge, gradient, Hu moments, Fourier)
- **Training Data**: 2,224 samples with SMOTE class balancing
- **Validation**: 456 samples with stratified split
- **Inference Time**: ~30ms per frame (real-time capable)
- **Background Robustness**: Advanced hand detection with skin color + background subtraction

### Per-Class Performance Analysis

**Significantly Improved Classes:**
- **Snail**: 0% → 80.0% (complete fix)
- **Swan**: 0% → 70.5% (complete fix)
- **Dog**: Improved accuracy + reduced llama confusion

**Excellent Performance Classes:**
- **Cat**: 95.8% F1-score (outstanding)
- **Bird**: 86.1% F1-score (excellent)
- **Deer**: 83.2% F1-score (excellent)

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