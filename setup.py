#!/usr/bin/env python3
"""
Setup script for the video summarization project.

This script helps set up the development environment and install dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def setup_project():
    """Setup the video summarization project."""
    print("🚀 Setting up Video Summarization Project")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create necessary directories
    directories = ["data", "output", "models", "cache"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing core dependencies"):
        return False
    
    # Install development dependencies (optional)
    install_dev = input("📦 Install development dependencies? (y/n): ").lower().strip()
    if install_dev in ['y', 'yes']:
        if not run_command("pip install -r requirements-dev.txt", "Installing development dependencies"):
            print("⚠️  Development dependencies installation failed, but core setup is complete")
    
    # Run tests
    run_tests = input("🧪 Run tests to verify installation? (y/n): ").lower().strip()
    if run_tests in ['y', 'yes']:
        run_command("python -m pytest tests/ -v", "Running tests")
    
    # Create sample video
    create_sample = input("🎥 Create sample video for testing? (y/n): ").lower().strip()
    if create_sample in ['y', 'yes']:
        try:
            from src.utils import create_sample_video
            sample_path = Path("data/sample_video.mp4")
            create_sample_video(sample_path, duration=10, fps=30)
            print(f"✅ Sample video created: {sample_path}")
        except Exception as e:
            print(f"⚠️  Could not create sample video: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\n📖 Next steps:")
    print("1. Run the example: python example.py")
    print("2. Launch web interface: streamlit run web_app/app.py")
    print("3. Use CLI: python cli.py --help")
    print("4. Read documentation: README.md")
    print("=" * 50)
    
    return True


if __name__ == "__main__":
    success = setup_project()
    sys.exit(0 if success else 1)
