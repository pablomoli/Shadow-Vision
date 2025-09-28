# TouchDesigner Integration Guide

Complete guide for connecting Shadow-Vision gesture recognition to TouchDesigner for interactive installations.

## Quick Start

### 1. Install OSC Dependency
```bash
pip install python-osc
```

### 2. Run TouchDesigner Bridge
```bash
python touchdesigner_osc_bridge.py
```

### 3. TouchDesigner OSC Setup
1. Open TouchDesigner
2. Add **OSC In CHOP**
3. Set Network Port: `7000`
4. Set Network Address: `127.0.0.1`

## OSC Message Reference

### Gesture Recognition Messages
| OSC Address | Type | Range | Description |
|-------------|------|-------|-------------|
| `/gesture/name` | string | - | Current gesture name |
| `/gesture/confidence` | float | 0.0-1.0 | ML confidence score |
| `/gesture/stability` | int | 0+ | Frames with same prediction |
| `/gesture/frame_count` | int | 0+ | Total processed frames |

### Individual Gesture Triggers
| OSC Address | Type | Value | Description |
|-------------|------|-------|-------------|
| `/gesture/bird` | float | 0.0/1.0 | Bird gesture active |
| `/gesture/cat` | float | 0.0/1.0 | Cat gesture active |
| `/gesture/llama` | float | 0.0/1.0 | Llama gesture active |
| `/gesture/rabbit` | float | 0.0/1.0 | Rabbit gesture active |
| `/gesture/deer` | float | 0.0/1.0 | Deer gesture active |
| `/gesture/dog` | float | 0.0/1.0 | Dog gesture active |
| `/gesture/snail` | float | 0.0/1.0 | Snail gesture active |
| `/gesture/swan` | float | 0.0/1.0 | Swan gesture active |

### Hand Position Data
| OSC Address | Type | Range | Description |
|-------------|------|-------|-------------|
| `/hand/x` | float | 0.0-1.0 | Normalized X position |
| `/hand/y` | float | 0.0-1.0 | Normalized Y position |
| `/hand/size` | float | 0.0-1.0 | Normalized hand size |
| `/hand/detected` | float | 0.0/1.0 | Hand detection status |

### System Information
| OSC Address | Type | Range | Description |
|-------------|------|-------|-------------|
| `/camera/width` | int | - | Camera frame width |
| `/camera/height` | int | - | Camera frame height |
| `/system/fps` | float | 0+ | Processing framerate |

## TouchDesigner Network Setup

### Basic OSC Reception
```
1. OSC In CHOP
   - Network Port: 7000
   - Network Address: 127.0.0.1
   - Auto Update: On

2. Split CHOP (to separate channels)
   - Input: oscin1
   - Split Method: By Channel Names

3. Select CHOP (for individual gestures)
   - Channel Names: gesture/bird gesture/cat gesture/dog etc.
```

### Advanced Processing Network
```
OSC In CHOP → Split CHOP → Logic CHOP → Trigger CHOP
                      ↓
            Select CHOP (confidence) → Math CHOP (threshold)
                      ↓
            Select CHOP (position) → Transform TOP
```

## Creative Applications

### 1. Gesture-Triggered Animations
```python
# TouchDesigner Setup:
# 1. OSC In CHOP receives /gesture/bird, /gesture/cat, etc.
# 2. Trigger CHOP converts 0→1 transitions to triggers
# 3. Animation COMP plays different sequences per gesture

# Example network:
OSC In → Select (gesture/bird) → Trigger → Animation COMP (bird sequence)
      → Select (gesture/cat) → Trigger → Animation COMP (cat sequence)
```

### 2. Hand Position Control
```python
# Use hand position to control visuals:
# 1. OSC In receives /hand/x, /hand/y, /hand/size
# 2. Transform TOP uses position data
# 3. Scale effects based on hand size

# Example network:
OSC In → Select (hand/x) → Transform TOP (translate X)
      → Select (hand/y) → Transform TOP (translate Y)
      → Select (hand/size) → Math CHOP → Transform TOP (scale)
```

### 3. Confidence-Based Effects
```python
# Visual feedback based on ML confidence:
# 1. Low confidence = subtle effects
# 2. High confidence = dramatic visuals

# Example network:
OSC In → Select (gesture/confidence) → Math CHOP (0.7-1.0 range) → Level TOP (opacity)
```

## Advanced Configuration

### Custom OSC Port
```bash
# Use different port
python touchdesigner_osc_bridge.py --td-port 7001
```

### Remote TouchDesigner
```bash
# Send to different machine
python touchdesigner_osc_bridge.py --td-ip 192.168.1.100 --td-port 7000
```

### Debug Mode
```bash
# Enable detailed logging
python touchdesigner_osc_bridge.py --debug
```

## Example TouchDesigner Patches

### Basic Gesture Visualizer
```
1. Create Geometry COMP for each animal
2. OSC In CHOP → Select CHOPs for each gesture
3. Trigger CHOPs detect gesture activation
4. Switch TOP changes between animal visuals
5. Level TOP controls opacity based on confidence
```

### Interactive Particle System
```
1. Particle SOP as base system
2. Hand position (/hand/x, /hand/y) controls Force SOP
3. Hand size (/hand/size) controls particle birth rate
4. Different gestures trigger different particle behaviors:
   - Bird: Upward forces (flying)
   - Cat: Scattered movement (pouncing)
   - Dog: Following behavior (loyal)
```

### Audio-Visual Reactive
```
1. Audio File In SOP triggered by gestures
2. Audio Spectrum CHOP analyzes audio
3. Visual effects respond to both:
   - Gesture type (determines base effect)
   - Audio analysis (modulates effect intensity)
```

## Troubleshooting

### No OSC Messages in TouchDesigner
```bash
# Check Python bridge is running
python touchdesigner_osc_bridge.py --debug

# Verify TouchDesigner OSC In settings:
# - Port: 7000 (default)
# - Address: 127.0.0.1
# - Auto Update: On
```

### Camera Not Working
```bash
# Test camera separately
python backend/cv_pipeline/advanced_realtime_inference.py

# Check camera permissions
# Windows: Settings → Privacy → Camera
```

### Performance Issues
```bash
# Reduce processing load
# In touchdesigner_osc_bridge.py, modify:
# - Lower camera resolution
# - Reduce inference frequency
# - Skip frames if needed
```

### Wrong Gesture Detection
```bash
# Retrain model if needed
python backend/data/advanced_train_classifier.py

# Adjust confidence threshold in TouchDesigner
# Use Math CHOP to filter low-confidence predictions
```

## Creative Ideas

### 1. Shadow Theater Installation
- Project shadow puppet silhouettes
- Gesture controls story progression
- Hand position moves characters

### 2. Interactive Music Performance
- Each gesture triggers different instruments
- Hand position controls pitch/volume
- Confidence affects effect intensity

### 3. Educational Animal Experience
- Gesture recognition activates animal facts
- 3D animal models appear and animate
- Educational content triggered by gestures

### 4. Generative Art System
- Hand gestures control art generation
- Different animals create different patterns
- Hand movement trails create brush strokes

## Performance Optimization

### TouchDesigner Settings
```
# Optimize for real-time performance:
1. Lower resolution textures where possible
2. Use LOD (Level of Detail) for 3D models
3. Limit particle counts
4. Use efficient shader techniques
```

### Python Bridge Optimization
```python
# In touchdesigner_osc_bridge.py:
# - Skip frames if processing too slow
# - Reduce OSC message frequency
# - Use threading for non-blocking OSC sends
```

## Getting Started Checklist

- [ ] Install `python-osc`: `pip install python-osc`
- [ ] Run bridge: `python touchdesigner_osc_bridge.py`
- [ ] Open TouchDesigner
- [ ] Add OSC In CHOP (port 7000)
- [ ] Test with `/gesture/name` messages
- [ ] Build your creative network
- [ ] Create gesture-triggered content
- [ ] Add hand position controls
- [ ] Implement confidence-based effects

## Support

### Common Issues
- **No OSC data**: Check port 7000 in TouchDesigner
- **Lag/delay**: Reduce camera resolution or skip frames
- **Wrong gestures**: Ensure good lighting and clean background

### Resources
- TouchDesigner OSC documentation
- Shadow-Vision model performance metrics
- Example .toe files (coming soon)

---

**Ready to create interactive installations with gesture control**