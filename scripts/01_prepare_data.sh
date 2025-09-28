#!/usr/bin/env bash

# Shadow-Vision Data Preparation Script (01_prepare_data.sh)
# Part of Utility #1: Dataset Download and Organization
#
# This script downloads the HuggingFace Animal-Puppet-Dataset,
# organizes it locally, and creates CSV splits with integrity checks.
#
# Outputs:
# - data/raw/train/<label_slug>/...
# - data/raw/val/<label_slug>/...
# - data/splits/train.csv
# - data/splits/val.csv
# - data/splits/labels.json

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

echo -e "${BLUE}Shadow-Vision Data Preparation${NC}"
echo -e "${BLUE}==============================${NC}"
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
required = ['datasets', 'PIL', 'pandas', 'yaml']
missing = []
for pkg in required:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)

if missing:
    print(f'ERROR: Missing packages: {missing}')
    print('Install with: pip install datasets pillow pandas pyyaml')
    sys.exit(1)
else:
    print('OK: All required packages found')
"

if [[ $? -ne 0 ]]; then
    echo -e "${RED}Package check failed!${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Starting dataset preparation...${NC}"
echo ""

# Change to project directory
cd "$PROJECT_ROOT"

# Run the dataset utility
echo -e "${YELLOW}Running dataset download and organization...${NC}"
python backend/data/dataset.py

# Check exit status
if [[ $? -eq 0 ]]; then
    echo ""
    echo -e "${GREEN}SUCCESS: Dataset preparation completed successfully!${NC}"
    echo ""
    echo -e "${BLUE}Dataset ready for next steps:${NC}"
    echo "   • Utility #2: Feature Extraction (MediaPipe)"
    echo "   • Utility #3: Training (k-NN classifier)"
    echo ""
    echo -e "${BLUE}Generated files:${NC}"
    echo "   • data/raw/train/<class>/... (training images)"
    echo "   • data/raw/val/<class>/... (validation images)"
    echo "   • data/splits/train.csv (training metadata)"
    echo "   • data/splits/val.csv (validation metadata)"
    echo "   • data/splits/labels.json (class definitions)"
    echo ""
else
    echo ""
    echo -e "${RED}ERROR: Dataset preparation failed!${NC}"
    echo "Check the error messages above for details."
    exit 1
fi