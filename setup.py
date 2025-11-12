#!/usr/bin/env python3
"""
Setup script for Machine Learning Textbook
Run this to install all required packages and set up the environment
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
    except subprocess.CalledProcessError:
        print("❌ Failed to install packages. Please install manually:")
        print("pip install -r requirements.txt")

def setup_jupyter():
    """Set up Jupyter notebook extensions"""
    print("📓 Setting up Jupyter notebook...")
    
    try:
        # Install kernel
        subprocess.check_call([sys.executable, "-m", "ipykernel", "install", "--user", "--name", "ml_textbook"])
        print("✅ Jupyter kernel installed!")
    except subprocess.CalledProcessError:
        print("⚠️ Jupyter kernel setup failed (optional)")

def create_directories():
    """Create necessary directories"""
    directories = [
        "notebooks/exercises",
        "datasets/raw",
        "datasets/processed", 
        "code/examples",
        "images/plots",
        "docs"
    ]
    
    print("📁 Creating directory structure...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  Created: {directory}")

def main():
    """Main setup function"""
    print("🚀 Setting up Machine Learning Textbook Environment")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Create directories
    create_directories()
    
    # Install packages
    install_requirements()
    
    # Setup Jupyter
    setup_jupyter()
    
    print("\n🎉 Setup complete!")
    print("\nTo get started:")
    print("1. cd to the project directory") 
    print("2. Run: jupyter notebook")
    print("3. Open notebooks/chapter_01_introduction.ipynb")
    
    print("\nProject structure:")
    print("├── chapters/          # Markdown chapters")
    print("├── notebooks/         # Jupyter notebooks")
    print("├── code/             # Python utilities")
    print("├── datasets/         # Data files")
    print("├── images/           # Plots and figures")
    print("└── docs/             # Documentation")

if __name__ == "__main__":
    main()
