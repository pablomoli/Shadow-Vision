#!/usr/bin/env bash

# Shadow-Vision Model Training Script (03_train_model.sh)
# Part of Utility #3: k-NN Classifier Training
#
# Trains k-NN classifier optimized for live demo stability

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

echo -e "${BLUE}Shadow-Vision Model Training${NC}"
echo -e "${BLUE}============================${NC}"
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

# Check if features were extracted
if [[ ! -f "$PROJECT_ROOT/data/splits/train_features.npz" ]]; then
    echo -e "${RED}ERROR: Training features not found!${NC}"
    echo "Run ./scripts/02_extract_features.sh first"
    exit 1
fi

if [[ ! -f "$PROJECT_ROOT/data/splits/val_features.npz" ]]; then
    echo -e "${RED}ERROR: Validation features not found!${NC}"
    echo "Run ./scripts/02_extract_features.sh first"
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
required = ['sklearn', 'numpy', 'pandas', 'yaml', 'joblib']
missing = []
for pkg in required:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)

if missing:
    print(f'ERROR: Missing packages: {missing}')
    print('Install with: pip install scikit-learn numpy pandas pyyaml joblib')
    sys.exit(1)
else:
    print('OK: All required packages found')
"

if [[ $? -ne 0 ]]; then
    echo -e "${RED}Package check failed!${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Starting model training...${NC}"
echo ""

# Change to project directory
cd "$PROJECT_ROOT"

# Run the training
echo -e "${YELLOW}Training k-NN classifier optimized for live demos...${NC}"
python backend/data/train_classifier.py

# Check exit status
if [[ $? -eq 0 ]]; then
    echo ""
    echo -e "${GREEN}SUCCESS: Model training completed successfully!${NC}"
    echo ""
    echo -e "${BLUE}Model ready for live demo:${NC}"
    echo "   • models/shadow_puppet_classifier.joblib (trained model)"
    echo "   • models/training_metrics.json (performance metrics)"
    echo "   • models/model_report.txt (human-readable report)"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "   • Test real-time inference"
    echo "   • Integrate with TouchDesigner via OSC"
    echo ""
else
    echo ""
    echo -e "${RED}ERROR: Model training failed!${NC}"
    echo "Check the error messages above for details."
    exit 1
fi