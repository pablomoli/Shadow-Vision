#!/usr/bin/env python3
"""
3D Model Downloader
Downloads sample 3D models for the gesture animations
"""

import os
import urllib.request
import json
import math
from pathlib import Path

class ModelDownloader:
    """Downloads and manages 3D models"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.models_dir = self.project_root / "frontend" / "src" / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Free 3D model sources (using placeholder URLs)
        self.model_sources = {
            "dog": {
                "url": "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/Fox/glTF/Fox.gltf",
                "filename": "dog.gltf",
                "license": "CC0 - Creative Commons"
            },
            "bird": {
                "url": "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/Duck/glTF/Duck.gltf",
                "filename": "bird.gltf",
                "license": "CC0 - Creative Commons"
            },
            "rabbit": {
                "url": "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/BunnyMesh/glTF/BunnyMesh.gltf",
                "filename": "rabbit.gltf",
                "license": "CC0 - Creative Commons"
            }
        }

    def download_file(self, url, destination):
        """Download a file from URL"""
        try:
            print(f"   Downloading {url}...")
            urllib.request.urlretrieve(url, destination)
            return True
        except Exception as e:
            print(f"   ❌ Failed to download {url}: {e}")
            return False

    def create_placeholder_models(self):
        """Create placeholder JSON files for models"""
        print("🎨 Creating placeholder 3D models...")

        placeholder_models = {
            "dog": {
                "type": "procedural",
                "geometry": "box",
                "color": "#8B4513",
                "animations": ["idle", "bark", "sit", "wag", "jump"],
                "scale": [1.0, 1.0, 1.0],
                "position": [0, 0, 0]
            },
            "bird": {
                "type": "procedural",
                "geometry": "sphere",
                "color": "#4169E1",
                "animations": ["fly", "perch", "flap", "peck", "turn"],
                "scale": [0.8, 0.8, 0.8],
                "position": [0, 2, 0]
            },
            "rabbit": {
                "type": "procedural",
                "geometry": "capsule",
                "color": "#FFFFFF",
                "animations": ["hop", "nibble", "alert", "clean", "sit"],
                "scale": [0.9, 0.9, 0.9],
                "position": [0, 0.5, 0]
            },
            "butterfly": {
                "type": "procedural",
                "geometry": "custom",
                "color": "#FF69B4",
                "animations": ["flutter", "land", "spiral", "rest", "takeoff"],
                "scale": [0.5, 0.5, 0.5],
                "position": [0, 1.5, 0]
            },
            "snake": {
                "type": "procedural",
                "geometry": "tube",
                "color": "#228B22",
                "animations": ["slither", "coil", "strike", "bask", "flick"],
                "scale": [2.0, 0.3, 0.3],
                "position": [0, 0.2, 0]
            }
        }

        for model_name, model_data in placeholder_models.items():
            model_file = self.models_dir / f"{model_name}.json"

            try:
                with open(model_file, 'w') as f:
                    json.dump(model_data, f, indent=2)
                print(f"   ✅ Created {model_name}.json")
            except Exception as e:
                print(f"   ❌ Failed to create {model_name}.json: {e}")

    def create_scene_configs(self):
        """Create scene configuration files"""
        print("🌍 Creating scene configurations...")

        scenes_dir = self.project_root / "frontend" / "src" / "scenes"
        scenes_dir.mkdir(parents=True, exist_ok=True)

        scene_configs = {
            "park": {
                "background_color": "#87CEEB",
                "fog": {"color": "#87CEEB", "near": 10, "far": 50},
                "lighting": {
                    "ambient": {"color": "#404040", "intensity": 0.6},
                    "directional": {"color": "#FFFFFF", "intensity": 1.0, "position": [10, 10, 5]}
                },
                "ground": {"color": "#228B22", "texture": "grass"},
                "objects": [
                    {"type": "tree", "position": [-5, 0, -5], "scale": [1, 1, 1]},
                    {"type": "bench", "position": [3, 0, 2], "scale": [1, 1, 1]}
                ]
            },
            "sky": {
                "background_color": "#4169E1",
                "fog": {"color": "#4169E1", "near": 20, "far": 100},
                "lighting": {
                    "ambient": {"color": "#6495ED", "intensity": 0.8},
                    "directional": {"color": "#FFFFFF", "intensity": 1.2, "position": [0, 20, 0]}
                },
                "ground": {"color": "#FFFFFF", "texture": "clouds"},
                "objects": [
                    {"type": "cloud", "position": [-10, 5, -10], "scale": [3, 1, 3]},
                    {"type": "cloud", "position": [8, 8, -15], "scale": [2, 1, 2]}
                ]
            },
            "meadow": {
                "background_color": "#90EE90",
                "fog": {"color": "#90EE90", "near": 15, "far": 60},
                "lighting": {
                    "ambient": {"color": "#404040", "intensity": 0.7},
                    "directional": {"color": "#FFFF88", "intensity": 1.1, "position": [8, 12, 3]}
                },
                "ground": {"color": "#32CD32", "texture": "grass"},
                "objects": [
                    {"type": "flower", "position": [2, 0, 1], "scale": [0.5, 0.5, 0.5]},
                    {"type": "flower", "position": [-3, 0, -2], "scale": [0.5, 0.5, 0.5]},
                    {"type": "rock", "position": [4, 0, -4], "scale": [1.2, 0.8, 1.2]}
                ]
            },
            "garden": {
                "background_color": "#FFB6C1",
                "fog": {"color": "#FFB6C1", "near": 12, "far": 45},
                "lighting": {
                    "ambient": {"color": "#FFF0F5", "intensity": 0.8},
                    "directional": {"color": "#FFFFFF", "intensity": 1.0, "position": [6, 10, 4]}
                },
                "ground": {"color": "#90EE90", "texture": "garden"},
                "objects": [
                    {"type": "flower_bed", "position": [0, 0, -3], "scale": [3, 1, 1]},
                    {"type": "butterfly_bush", "position": [-2, 0, 2], "scale": [1, 1.5, 1]}
                ]
            },
            "desert": {
                "background_color": "#DEB887",
                "fog": {"color": "#DEB887", "near": 18, "far": 70},
                "lighting": {
                    "ambient": {"color": "#8B7355", "intensity": 0.9},
                    "directional": {"color": "#FFFF99", "intensity": 1.3, "position": [15, 15, 0]}
                },
                "ground": {"color": "#F4A460", "texture": "sand"},
                "objects": [
                    {"type": "cactus", "position": [-4, 0, -6], "scale": [1, 2, 1]},
                    {"type": "rock", "position": [3, 0, -3], "scale": [1.5, 1, 1.5]},
                    {"type": "dune", "position": [0, 0, -10], "scale": [8, 2, 4]}
                ]
            }
        }

        for scene_name, scene_data in scene_configs.items():
            scene_file = scenes_dir / f"{scene_name}.json"

            try:
                with open(scene_file, 'w') as f:
                    json.dump(scene_data, f, indent=2)
                print(f"   ✅ Created {scene_name}.json")
            except Exception as e:
                print(f"   ❌ Failed to create {scene_name}.json: {e}")

    def create_animation_configs(self):
        """Create animation configuration files"""
        print("🎬 Creating animation configurations...")

        animations_dir = self.project_root / "frontend" / "src" / "animations"
        animations_dir.mkdir(parents=True, exist_ok=True)

        animation_configs = {
            "dog": {
                "idle": {
                    "type": "loop",
                    "duration": 2000,
                    "keyframes": [
                        {"time": 0, "position": [0, 0, 0], "rotation": [0, 0, 0]},
                        {"time": 1000, "position": [0, 0.1, 0], "rotation": [0, 0, 0]},
                        {"time": 2000, "position": [0, 0, 0], "rotation": [0, 0, 0]}
                    ]
                },
                "bark": {
                    "type": "once",
                    "duration": 1000,
                    "keyframes": [
                        {"time": 0, "scale": [1, 1, 1]},
                        {"time": 200, "scale": [1.1, 1.1, 1.1]},
                        {"time": 400, "scale": [1, 1, 1]},
                        {"time": 600, "scale": [1.05, 1.05, 1.05]},
                        {"time": 1000, "scale": [1, 1, 1]}
                    ]
                }
            },
            "bird": {
                "fly": {
                    "type": "loop",
                    "duration": 1500,
                    "keyframes": [
                        {"time": 0, "position": [0, 2, 0], "rotation": [0, 0, 0]},
                        {"time": 750, "position": [0, 2.5, 0], "rotation": [0, Math.PI, 0]},
                        {"time": 1500, "position": [0, 2, 0], "rotation": [0, 2*Math.PI, 0]}
                    ]
                }
            },
            "rabbit": {
                "hop": {
                    "type": "loop",
                    "duration": 800,
                    "keyframes": [
                        {"time": 0, "position": [0, 0.5, 0]},
                        {"time": 200, "position": [0, 1.2, 0.3]},
                        {"time": 400, "position": [0, 0.5, 0.6]},
                        {"time": 600, "position": [0, 1.2, 0.3]},
                        {"time": 800, "position": [0, 0.5, 0]}
                    ]
                }
            }
        }

        for animal, animations in animation_configs.items():
            animation_file = animations_dir / f"{animal}_animations.json"

            try:
                with open(animation_file, 'w') as f:
                    json.dump(animations, f, indent=2)
                print(f"   ✅ Created {animal}_animations.json")
            except Exception as e:
                print(f"   ❌ Failed to create {animal}_animations.json: {e}")

    def create_model_manifest(self):
        """Create a manifest file listing all available models"""
        print("📋 Creating model manifest...")

        manifest = {
            "version": "1.0",
            "models": {},
            "scenes": {},
            "animations": {},
            "created": "2024-01-01T00:00:00Z",
            "description": "Gesture Puppets 3D Assets"
        }

        # List model files
        for model_file in self.models_dir.glob("*.json"):
            model_name = model_file.stem
            manifest["models"][model_name] = f"models/{model_file.name}"

        # List scene files
        scenes_dir = self.project_root / "frontend" / "src" / "scenes"
        if scenes_dir.exists():
            for scene_file in scenes_dir.glob("*.json"):
                scene_name = scene_file.stem
                manifest["scenes"][scene_name] = f"scenes/{scene_file.name}"

        # List animation files
        animations_dir = self.project_root / "frontend" / "src" / "animations"
        if animations_dir.exists():
            for anim_file in animations_dir.glob("*.json"):
                anim_name = anim_file.stem
                manifest["animations"][anim_name] = f"animations/{anim_file.name}"

        # Save manifest
        manifest_file = self.project_root / "frontend" / "src" / "asset_manifest.json"
        try:
            with open(manifest_file, 'w') as f:
                json.dump(manifest, f, indent=2)
            print(f"   ✅ Created asset_manifest.json")
        except Exception as e:
            print(f"   ❌ Failed to create manifest: {e}")

    def download_all(self):
        """Download and create all assets"""
        print("🎨 Setting up 3D assets...")
        print("=" * 50)

        # Create placeholder models (since we don't have real GLTF models)
        self.create_placeholder_models()

        # Create scene configurations
        self.create_scene_configs()

        # Create animation configurations
        self.create_animation_configs()

        # Create asset manifest
        self.create_model_manifest()

        print("\n✅ Asset setup complete!")
        print("\nNote: This demo uses procedural 3D models.")
        print("For production, replace with proper GLTF models.")

def main():
    """Main entry point"""
    downloader = ModelDownloader()
    downloader.download_all()

if __name__ == "__main__":
    main()