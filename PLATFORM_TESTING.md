# Platform Testing Status & Requirements

## Testing Status

### ✅ Verified Components
- **Windows virtual environment syntax** - Confirmed correct paths and activation scripts
- **Docker compose file syntax** - All YAML files validated
- **Python launcher commands** - Windows `py` command syntax verified
- **Cross-platform file paths** - Forward/backward slash handling correct

### ❌ Requires Testing
- **macOS Docker camera access** - Need macOS system with Docker Desktop
- **Linux device mapping** - Need Linux system with `/dev/video0` access
- **Windows Docker Desktop camera** - Need Windows with Docker Desktop installed
- **MediaPipe installation** - Cross-platform MediaPipe compatibility testing

## Platform Requirements for Testing

### Windows Testing Requirements
```cmd
# Required software
- Python 3.10, 3.11, or 3.12
- Docker Desktop with camera permissions enabled
- Git for Windows

# Test commands
py --list                                    # Check available Python versions
docker --version                             # Verify Docker Desktop
docker-compose -f docker-compose.windows.yml config  # Validate YAML syntax
```

### macOS Testing Requirements
```bash
# Required software
- Python 3.10+ (via Homebrew recommended)
- Docker Desktop for Mac with camera permissions
- Xcode Command Line Tools

# Test commands
python3 --version                            # Check Python
docker --version                             # Verify Docker Desktop
docker-compose -f docker-compose.macos.yml config   # Validate YAML syntax
```

### Linux Testing Requirements
```bash
# Required software
- Python 3.10+ (system package or pyenv)
- Docker CE or Docker Desktop
- V4L2 camera drivers

# Test commands
python3 --version                            # Check Python
docker --version                             # Verify Docker
ls /dev/video*                               # Check camera devices
docker-compose -f docker-compose.touchdesigner.yml config  # Validate YAML
```

## Known Issues & Workarounds

### Windows Issues
1. **Python 3.11/3.12 not installed**:
   - Install from python.org or Microsoft Store
   - Alternative: Use Docker (works with any Python version)

2. **Docker Desktop camera access**:
   - Enable in Docker Desktop → Settings → Resources → File sharing
   - Grant camera permissions in Windows Privacy settings

### macOS Issues
1. **Camera permissions**:
   - System Preferences → Security & Privacy → Camera → Docker Desktop
   - May require Docker Desktop restart

2. **Python version conflicts**:
   - Use `python3` instead of `python`
   - Consider using `pyenv` for version management

### Linux Issues
1. **Camera device permissions**:
   - Add user to `video` group: `sudo usermod -a -G video $USER`
   - Logout/login required after group change

2. **Docker permissions**:
   - Add user to `docker` group: `sudo usermod -a -G docker $USER`

## Verification Checklist

### Pre-Deployment Testing
- [ ] **Docker syntax validation** - All compose files parse correctly
- [ ] **Python command testing** - Virtual environment creation works
- [ ] **Camera detection test** - `cv2.VideoCapture(0).isOpened()` returns True
- [ ] **OSC communication test** - `python simple_osc_test.py` succeeds

### Platform-Specific Testing
- [ ] **Windows**: Docker Desktop camera access functional
- [ ] **macOS**: Camera permissions granted to Docker Desktop
- [ ] **Linux**: Device mapping `/dev/video0` accessible in container

### Integration Testing
- [ ] **TouchDesigner connection** - OSC messages received on port 7000
- [ ] **Gesture recognition** - Hand detection and classification working
- [ ] **Performance testing** - 30+ FPS sustained recognition

## Testing Commands by Platform

### Windows Verification
```cmd
# Environment setup
py -3.11 -m venv test_env
test_env\Scripts\activate.bat
pip install opencv-python

# Camera test
python -c "import cv2; print('Camera:', cv2.VideoCapture(0).isOpened())"

# Docker test (requires Docker Desktop)
docker-compose -f docker-compose.windows.yml up --build --abort-on-container-exit
```

### macOS Verification
```bash
# Environment setup
python3.11 -m venv test_env
source test_env/bin/activate
pip install opencv-python

# Camera test
python -c "import cv2; print('Camera:', cv2.VideoCapture(0).isOpened())"

# Docker test (requires Docker Desktop)
docker-compose -f docker-compose.macos.yml up --build --abort-on-container-exit
```

### Linux Verification
```bash
# Environment setup
python3.11 -m venv test_env
source test_env/bin/activate
pip install opencv-python

# Camera test
python -c "import cv2; print('Camera:', cv2.VideoCapture(0).isOpened())"

# Docker test
docker-compose -f docker-compose.touchdesigner.yml up --build --abort-on-container-exit
```

## Community Testing Request

**We need community testing for complete platform verification!**

### How to Help Test
1. **Choose your platform** (Windows/macOS/Linux)
2. **Follow the verification commands** above
3. **Report results** via GitHub Issues with:
   - Operating system version
   - Python version used
   - Docker version
   - Success/failure of each test step
   - Any error messages encountered

### Expected Test Results
- **Camera detection**: Should return `True`
- **Docker build**: Should complete without errors
- **Container startup**: Should show MediaPipe initialization messages
- **OSC communication**: Should be testable with TouchDesigner

## Fallback Solutions

If platform-specific issues arise:
1. **Use automated setup script**: `python setup_mediapipe_env.py`
2. **Try Docker approach**: Usually more reliable across platforms
3. **Check dependencies**: Ensure camera drivers and permissions are correct
4. **Community support**: Ask for help via GitHub Discussions

---

**Status**: Requires community testing for full platform validation
**Priority**: High - needed for reliable cross-platform deployment