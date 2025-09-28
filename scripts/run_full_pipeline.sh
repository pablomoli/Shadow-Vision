#!/usr/bin/env bash

# Shadow-Vision Complete Pipeline Script
# Runs the entire training pipeline from dataset to live demo

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${CYAN}Shadow-Vision Complete Training Pipeline${NC}"
echo -e "${CYAN}=======================================${NC}"
echo ""
echo "This script will run the complete Shadow-Vision training pipeline:"
echo "  1. Dataset preparation (download from HuggingFace)"
echo "  2. Feature extraction (OpenCV-based)"
echo "  3. Model training (k-NN classifier)"
echo "  4. Pipeline testing (comprehensive validation)"
echo "  5. Live demo readiness check"
echo ""

# Ask for confirmation
read -p "Continue? (y/N): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted by user"
    exit 0
fi

# Change to project directory
cd "$PROJECT_ROOT"

# Check if we're in the right directory
if [[ ! -f "config/config.yaml" ]]; then
    echo -e "${RED}ERROR: config/config.yaml not found!${NC}"
    echo "Make sure you're running this from the gesture-puppets directory"
    exit 1
fi

# Function to check if a step was successful
check_step() {
    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}SUCCESS: $1 completed${NC}"
        echo ""
    else
        echo -e "${RED}ERROR: $1 failed${NC}"
        echo "Check the error messages above for details."
        exit 1
    fi
}

echo -e "${BLUE}Starting complete pipeline...${NC}"
echo ""

# Step 1: Dataset Preparation
echo -e "${YELLOW}STEP 1: Dataset Preparation${NC}"
echo "Downloading and organizing HuggingFace dataset..."
./scripts/01_prepare_data.sh
check_step "Dataset preparation"

# Step 2: Feature Extraction
echo -e "${YELLOW}STEP 2: Feature Extraction${NC}"
echo "Extracting OpenCV-based geometric features..."
./scripts/02_extract_features.sh
check_step "Feature extraction"

# Step 3: Model Training
echo -e "${YELLOW}STEP 3: Model Training${NC}"
echo "Training k-NN classifier optimized for live demos..."
./scripts/03_train_model.sh
check_step "Model training"

# Step 4: Pipeline Testing
echo -e "${YELLOW}STEP 4: Pipeline Testing${NC}"
echo "Running comprehensive validation tests..."
python scripts/test_pipeline.py
check_step "Pipeline testing"

# Final summary
echo -e "${GREEN}🎉 COMPLETE PIPELINE FINISHED SUCCESSFULLY! 🎉${NC}"
echo ""
echo -e "${CYAN}Generated Assets:${NC}"
echo "  📁 data/raw/                          - Organized dataset images"
echo "  📄 data/splits/*.csv                  - Dataset metadata"
echo "  📄 data/splits/*_features.npz         - Extracted features"
echo "  🤖 models/shadow_puppet_classifier.joblib - Trained model"
echo "  📊 models/training_metrics.json       - Performance metrics"
echo "  📋 models/model_report.txt            - Human-readable report"
echo "  🧪 test_report.json                   - Test validation results"
echo ""

echo -e "${CYAN}Next Steps - Live Demo:${NC}"
echo "  🎥 Test with camera:"
echo "     python backend/cv_pipeline/realtime_inference.py"
echo ""
echo "  🔗 TouchDesigner Integration:"
echo "     - Configure OSC in config/config.yaml"
echo "     - Set host/port for TouchDesigner communication"
echo "     - Run real-time inference with OSC output"
echo ""

echo -e "${CYAN}Performance Summary:${NC}"
if [[ -f "test_report.json" ]]; then
    python -c "
import json
try:
    with open('test_report.json', 'r') as f:
        data = json.load(f)

    if 'accuracy' in data:
        print(f\"  🎯 Model Accuracy: {data['accuracy']['overall']:.1%}\")

    if 'performance' in data:
        perf = data['performance']
        print(f\"  ⚡ Processing Speed: {perf['avg_processing_time_ms']:.1f}ms\")
        print(f\"  📈 Estimated FPS: {perf['estimated_fps']:.1f}\")
        print(f\"  👁 Detection Rate: {perf['detection_rate']:.1%}\")

    if 'stability' in data:
        print(f\"  🎲 Prediction Consistency: {data['stability']['consistency']:.1%}\")

except Exception as e:
    print(f\"  📊 Detailed metrics available in test_report.json\")
"
fi

echo ""
echo -e "${GREEN}🚀 Shadow-Vision is ready for your live demo! 🚀${NC}"