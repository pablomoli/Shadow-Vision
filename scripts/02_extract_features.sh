#!/usr/bin/env bash

# Shadow-Vision Feature Extraction Script (02_extract_features.sh)
# Part of Utility #2: Feature Extraction with OpenCV
#
# Extracts geometric features from hand contours for ML training
# More stable than MediaPipe for live demos

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}Shadow-Vision Feature Extraction${NC}"
echo -e "${BLUE}================================${NC}"
echo ""
echo "Project root: $PROJECT_ROOT"
echo "Script: $(basename "$0")"
echo ""

# Check if we're in the right directory
if [[ ! -f "$PROJECT_ROOT/config/config.yaml" ]]; then
    echo -e "${RED}ERROR: config/config.yaml not found!${NC}"
    echo "Make sure you're running this from the gesture-puppets directory"
    exit 1
fi

# Check if dataset was prepared
if [[ ! -f "$PROJECT_ROOT/data/splits/train.csv" ]]; then
    echo -e "${RED}ERROR: Dataset not prepared!${NC}"
    echo "Run ./scripts/01_prepare_data.sh first"
    exit 1
fi

# Check Python environment
echo -e "${YELLOW}Checking Python environment...${NC}"
if ! command -v python &> /dev/null; then
    echo -e "${RED}ERROR: Python not found!${NC}"
    exit 1
fi

# Check required packages
echo -e "${YELLOW}Checking required packages...${NC}"
python -c "
import sys
required = ['cv2', 'numpy', 'pandas', 'yaml', 'PIL']
missing = []
for pkg in required:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)

if missing:
    print(f'ERROR: Missing packages: {missing}')
    print('Install with: pip install opencv-python numpy pandas pyyaml pillow')
    sys.exit(1)
else:
    print('OK: All required packages found')
"

if [[ $? -ne 0 ]]; then
    echo -e "${RED}Package check failed!${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Starting feature extraction...${NC}"
echo ""

# Change to project directory
cd "$PROJECT_ROOT"

# Run the feature extraction
echo -e "${YELLOW}Running OpenCV-based feature extraction...${NC}"
python backend/data/feature_extractor.py

# Check exit status
if [[ $? -eq 0 ]]; then
    echo ""
    echo -e "${GREEN}SUCCESS: Feature extraction completed successfully!${NC}"
    echo ""
    echo -e "${BLUE}Features ready for training:${NC}"
    echo "   • data/splits/train_features.npz (training features)"
    echo "   • data/splits/val_features.npz (validation features)"
    echo ""
    echo -e "${BLUE}Next step:${NC}"
    echo "   • ./scripts/03_train_model.sh (train k-NN classifier)"
    echo ""
else
    echo ""
    echo -e "${RED}ERROR: Feature extraction failed!${NC}"
    echo "Check the error messages above for details."
    exit 1
fi