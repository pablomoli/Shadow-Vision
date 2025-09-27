#!/usr/bin/env python3
"""
Environment Setup Script
One-click setup for the Gesture Puppets project
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import json
import shutil

class EnvironmentSetup:
    """Handles project environment setup"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.platform = platform.system().lower()
        self.python_executable = sys.executable

        print("🎭 Gesture Puppets - Environment Setup")
        print("=" * 50)
        print(f"Platform: {platform.system()} {platform.release()}")
        print(f"Python: {sys.version}")
        print(f"Project root: {self.project_root}")
        print()

    def run_command(self, command, description=None, cwd=None):
        """Run a command and handle errors"""
        if description:
            print(f"📦 {description}...")

        try:
            if isinstance(command, str):
                result = subprocess.run(command, shell=True, check=True,
                                      capture_output=True, text=True, cwd=cwd)
            else:
                result = subprocess.run(command, check=True,
                                      capture_output=True, text=True, cwd=cwd)

            if result.stdout:
                print(f"   Output: {result.stdout.strip()}")

            return True

        except subprocess.CalledProcessError as e:
            print(f"   ❌ Error: {e}")
            if e.stderr:
                print(f"   Error details: {e.stderr.strip()}")
            return False

    def check_prerequisites(self):
        """Check system prerequisites"""
        print("🔍 Checking prerequisites...")

        # Check Python version
        if sys.version_info < (3, 8):
            print("   ❌ Python 3.8+ required")
            return False
        print(f"   ✅ Python {sys.version_info.major}.{sys.version_info.minor}")

        # Check pip
        try:
            import pip
            print(f"   ✅ pip available")
        except ImportError:
            print("   ❌ pip not found")
            return False

        # Check git (optional but recommended)
        try:
            subprocess.run(["git", "--version"], check=True,
                         capture_output=True)
            print("   ✅ git available")
        except:
            print("   ⚠️  git not found (optional)")

        # Check Node.js (for frontend)
        try:
            result = subprocess.run(["node", "--version"], check=True,
                                  capture_output=True, text=True)
            version = result.stdout.strip()
            print(f"   ✅ Node.js {version}")
        except:
            print("   ⚠️  Node.js not found (needed for frontend development)")

        return True

    def create_virtual_environment(self):
        """Create Python virtual environment"""
        venv_path = self.project_root / "venv"

        if venv_path.exists():
            print("   ✅ Virtual environment already exists")
            return True

        print("🐍 Creating virtual environment...")
        command = [self.python_executable, "-m", "venv", str(venv_path)]
        return self.run_command(command, "Creating virtual environment")

    def install_python_dependencies(self):
        """Install Python dependencies"""
        print("📦 Installing Python dependencies...")

        # Activate virtual environment
        venv_path = self.project_root / "venv"
        if self.platform == "windows":
            pip_path = venv_path / "Scripts" / "pip"
        else:
            pip_path = venv_path / "bin" / "pip"

        if not pip_path.exists():
            print("   ❌ Virtual environment not found")
            return False

        requirements_path = self.project_root / "requirements.txt"
        if not requirements_path.exists():
            print("   ❌ requirements.txt not found")
            return False

        command = [str(pip_path), "install", "-r", str(requirements_path)]
        return self.run_command(command, "Installing Python packages")

    def install_frontend_dependencies(self):
        """Install frontend dependencies"""
        print("🌐 Installing frontend dependencies...")

        # Check if npm is available
        try:
            subprocess.run(["npm", "--version"], check=True, capture_output=True)
        except:
            print("   ⚠️  npm not found, skipping frontend setup")
            return True

        package_json = self.project_root / "package.json"
        if not package_json.exists():
            print("   ❌ package.json not found")
            return False

        return self.run_command(["npm", "install"], "Installing Node.js packages",
                               cwd=self.project_root)

    def setup_environment_file(self):
        """Set up environment variables file"""
        print("⚙️  Setting up environment configuration...")

        env_example = self.project_root / "config" / ".env.example"
        env_file = self.project_root / "config" / ".env"

        if env_file.exists():
            print("   ✅ Environment file already exists")
            return True

        if not env_example.exists():
            print("   ❌ .env.example not found")
            return False

        try:
            shutil.copy2(env_example, env_file)
            print("   ✅ Environment file created from template")
            print("   ⚠️  Please edit config/.env with your actual values")
            return True
        except Exception as e:
            print(f"   ❌ Failed to create environment file: {e}")
            return False

    def create_data_directories(self):
        """Create necessary data directories"""
        print("📁 Creating data directories...")

        directories = [
            "data",
            "data/hasper",
            "backend/trained_models",
            "logs"
        ]

        for dir_path in directories:
            full_path = self.project_root / dir_path
            try:
                full_path.mkdir(parents=True, exist_ok=True)
                print(f"   ✅ Created {dir_path}")
            except Exception as e:
                print(f"   ❌ Failed to create {dir_path}: {e}")
                return False

        return True

    def test_installation(self):
        """Test the installation"""
        print("🧪 Testing installation...")

        # Test Python imports
        test_imports = [
            "torch",
            "cv2",
            "numpy",
            "fastapi",
            "websockets"
        ]

        venv_path = self.project_root / "venv"
        if self.platform == "windows":
            python_path = venv_path / "Scripts" / "python"
        else:
            python_path = venv_path / "bin" / "python"

        for module in test_imports:
            command = [str(python_path), "-c", f"import {module}; print(f'{module} imported successfully')"]
            if not self.run_command(command, f"Testing {module} import"):
                return False

        print("   ✅ All imports successful")
        return True

    def show_next_steps(self):
        """Show next steps to the user"""
        print("\n" + "=" * 50)
        print("🎉 Setup Complete!")
        print("=" * 50)

        print("\n📋 Next Steps:")
        print("1. Edit config/.env with your Supabase credentials (optional)")
        print("2. Download the dataset:")

        if self.platform == "windows":
            print("   venv\\Scripts\\python backend\\main.py download-dataset")
        else:
            print("   ./venv/bin/python backend/main.py download-dataset")

        print("3. Train the model:")
        if self.platform == "windows":
            print("   venv\\Scripts\\python backend\\main.py train")
        else:
            print("   ./venv/bin/python backend/main.py train")

        print("4. Start the server:")
        if self.platform == "windows":
            print("   venv\\Scripts\\python backend\\main.py server")
        else:
            print("   ./venv/bin/python backend/main.py server")

        print("5. Open frontend/public/index.html in your browser")

        print("\n🔧 Development Commands:")
        print(f"   Activate virtual environment:")
        if self.platform == "windows":
            print("   venv\\Scripts\\activate")
        else:
            print("   source venv/bin/activate")

        print("\n📚 For more information, see README.md")

    def setup(self):
        """Run complete setup process"""
        steps = [
            ("Prerequisites", self.check_prerequisites),
            ("Virtual Environment", self.create_virtual_environment),
            ("Python Dependencies", self.install_python_dependencies),
            ("Frontend Dependencies", self.install_frontend_dependencies),
            ("Environment Configuration", self.setup_environment_file),
            ("Data Directories", self.create_data_directories),
            ("Installation Test", self.test_installation)
        ]

        print("🚀 Starting setup process...\n")

        for step_name, step_func in steps:
            try:
                if not step_func():
                    print(f"\n❌ Setup failed at step: {step_name}")
                    return False
                print()
            except Exception as e:
                print(f"\n❌ Setup failed at step {step_name}: {e}")
                return False

        self.show_next_steps()
        return True

def main():
    """Main entry point"""
    setup = EnvironmentSetup()

    try:
        success = setup.setup()
        exit_code = 0 if success else 1
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup cancelled by user")
        exit_code = 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        exit_code = 1

    sys.exit(exit_code)

if __name__ == "__main__":
    main()