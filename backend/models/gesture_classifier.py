#!/usr/bin/env python3
"""
Gesture Recognition CNN Model
Lightweight CNN optimized for real-time hand gesture recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import yaml
from pathlib import Path

class GestureCNN(nn.Module):
    """Lightweight CNN for gesture classification"""

    def __init__(self, num_classes=5, input_size=(224, 224)):
        super(GestureCNN, self).__init__()

        self.num_classes = num_classes
        self.input_size = input_size

        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class MobileNetGestureClassifier(nn.Module):
    """MobileNet-based gesture classifier for better accuracy"""

    def __init__(self, num_classes=5, pretrained=True):
        super(MobileNetGestureClassifier, self).__init__()

        # Use MobileNetV2 as backbone
        self.backbone = models.mobilenet_v2(pretrained=pretrained)

        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.backbone.last_channel, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

class EfficientGestureNet(nn.Module):
    """Efficient gesture recognition network optimized for real-time inference"""

    def __init__(self, num_classes=5):
        super(EfficientGestureNet, self).__init__()

        # Depthwise separable convolutions for efficiency
        self.features = nn.Sequential(
            # Initial conv
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),

            # Depthwise separable blocks
            self._make_depthwise_block(16, 32, stride=1),
            self._make_depthwise_block(32, 64, stride=2),
            self._make_depthwise_block(64, 128, stride=1),
            self._make_depthwise_block(128, 128, stride=2),
            self._make_depthwise_block(128, 256, stride=1),
            self._make_depthwise_block(256, 256, stride=2),

            # Global average pooling
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def _make_depthwise_block(self, in_channels, out_channels, stride=1):
        """Create a depthwise separable convolution block"""
        return nn.Sequential(
            # Depthwise
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            # Pointwise
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def load_model_from_config(config_path="config/model_config.yaml", model_type="efficient"):
    """Load model based on configuration"""

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    num_classes = config['model']['num_classes']

    if model_type == "lightweight":
        model = GestureCNN(num_classes=num_classes)
    elif model_type == "mobilenet":
        model = MobileNetGestureClassifier(num_classes=num_classes)
    elif model_type == "efficient":
        model = EfficientGestureNet(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model

def count_parameters(model):
    """Count the number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size_mb(model):
    """Calculate model size in MB"""
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / (1024 * 1024)
    return size_mb

def test_model_inference_speed(model, device="cpu", input_size=(1, 3, 224, 224), num_runs=100):
    """Test model inference speed"""
    import time

    model.eval()
    model = model.to(device)

    # Warm up
    dummy_input = torch.randn(input_size).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # Time inference
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)

    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs * 1000  # ms

    return avg_time

def main():
    """Test different model architectures"""
    print("Testing gesture recognition models...\n")

    models_to_test = [
        ("lightweight", "GestureCNN"),
        ("mobilenet", "MobileNetGestureClassifier"),
        ("efficient", "EfficientGestureNet")
    ]

    for model_type, model_name in models_to_test:
        print(f"=== {model_name} ===")

        try:
            # For testing, create model without config file
            if model_type == "lightweight":
                model = GestureCNN(num_classes=5)
            elif model_type == "mobilenet":
                model = MobileNetGestureClassifier(num_classes=5, pretrained=False)
            elif model_type == "efficient":
                model = EfficientGestureNet(num_classes=5)

            # Model statistics
            num_params = count_parameters(model)
            model_size = get_model_size_mb(model)
            inference_time = test_model_inference_speed(model)

            print(f"Parameters: {num_params:,}")
            print(f"Model size: {model_size:.2f} MB")
            print(f"Avg inference time: {inference_time:.2f} ms")

            # Test forward pass
            dummy_input = torch.randn(1, 3, 224, 224)
            output = model(dummy_input)
            print(f"Output shape: {output.shape}")
            print(f"Output (softmax): {F.softmax(output, dim=1)}")

        except Exception as e:
            print(f"Error testing {model_name}: {e}")

        print()

if __name__ == "__main__":
    main()