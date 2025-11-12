# Appendix A: Python Environment Setup

## A.1 Introduction

This appendix provides comprehensive instructions for setting up a Python environment for machine learning development. We'll cover multiple approaches to ensure you can work with the examples and exercises throughout this textbook.

---

## A.2 Anaconda/Miniconda Installation

### A.2.1 Anaconda vs Miniconda

**Anaconda** is a complete Python distribution that includes:
- Python interpreter
- 250+ pre-installed packages for data science
- Conda package manager
- Anaconda Navigator GUI
- Jupyter Notebook, Spyder IDE, and other tools

**Miniconda** is a minimal installer that includes:
- Python interpreter
- Conda package manager
- Basic packages only

**Recommendation**: Use Miniconda for more control over your environment, or Anaconda for convenience.

### A.2.2 Installation Instructions

#### Windows
1. Download the installer from [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)
2. Run the installer as Administrator
3. Choose "Add Anaconda to my PATH environment variable" (optional but recommended)
4. Complete the installation

#### macOS
```bash
# Using Homebrew (recommended)
brew install --cask anaconda

# Or download from website and install manually
# Download .pkg file from anaconda.com and run installer
```

#### Linux
```bash
# Download the installer
wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh

# Make it executable and run
chmod +x Anaconda3-2023.09-0-Linux-x86_64.sh
./Anaconda3-2023.09-0-Linux-x86_64.sh

# Follow the prompts and add conda to PATH
echo 'export PATH="$HOME/anaconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### A.2.3 Verification
```bash
# Check conda installation
conda --version

# Check Python installation
python --version

# List installed packages
conda list
```

---

## A.3 Virtual Environment Management

### A.3.1 Why Use Virtual Environments?

Virtual environments provide:
- **Isolation**: Separate package installations for different projects
- **Reproducibility**: Consistent environments across machines
- **Dependency Management**: Avoid version conflicts
- **Clean System**: Keep base Python installation clean

### A.3.2 Creating Environments with Conda

#### Basic Environment Creation
```bash
# Create a new environment
conda create --name ml-textbook python=3.9

# Activate the environment
conda activate ml-textbook

# Deactivate when done
conda deactivate
```

#### Environment with Specific Packages
```bash
# Create environment with essential packages
conda create --name ml-textbook python=3.9 \
    numpy pandas matplotlib seaborn \
    scikit-learn jupyter notebook

# Create from environment file
conda env create -f environment.yml
```

#### Environment File (environment.yml)
```yaml
name: ml-textbook
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - numpy=1.24.0
  - pandas=2.0.0
  - matplotlib=3.7.0
  - seaborn=0.12.0
  - scikit-learn=1.3.0
  - jupyter=1.0.0
  - notebook=6.5.0
  - ipykernel=6.22.0
  - pip=23.0.0
  - pip:
    - plotly==5.14.0
    - shap==0.41.0
    - lime==0.2.0.1
```

### A.3.3 Managing Environments

```bash
# List all environments
conda env list
conda info --envs

# Clone an environment
conda create --name ml-textbook-copy --clone ml-textbook

# Export environment
conda env export > environment.yml

# Remove environment
conda env remove --name ml-textbook

# Update environment from file
conda env update --name ml-textbook --file environment.yml
```

### A.3.4 Using pip with Virtual Environments

#### Using venv (built-in)
```bash
# Create virtual environment
python -m venv ml-textbook-env

# Activate (Windows)
ml-textbook-env\Scripts\activate

# Activate (macOS/Linux)
source ml-textbook-env/bin/activate

# Install packages
pip install numpy pandas matplotlib scikit-learn jupyter

# Create requirements file
pip freeze > requirements.txt

# Install from requirements
pip install -r requirements.txt

# Deactivate
deactivate
```

---

## A.4 Jupyter Notebook Configuration

### A.4.1 Installation and Setup

```bash
# Install Jupyter (if not already installed)
conda install jupyter notebook

# Or with pip
pip install jupyter notebook

# Install JupyterLab (modern interface)
conda install jupyterlab

# Or with pip
pip install jupyterlab
```

### A.4.2 Jupyter Configuration

#### Generate Configuration File
```bash
# Generate config file
jupyter notebook --generate-config

# Config file location
# Windows: C:\Users\username\.jupyter\jupyter_notebook_config.py
# macOS/Linux: ~/.jupyter/jupyter_notebook_config.py
```

#### Essential Configuration Settings

```python
# ~/.jupyter/jupyter_notebook_config.py

# Set default directory
c.NotebookApp.notebook_dir = '/path/to/your/projects'

# Enable extensions
c.NotebookApp.nbserver_extensions = {
    'jupyter_nbextensions_configurator': True,
    'nbgrader.server_extensions.formgrader': True,
}

# Security settings
c.NotebookApp.token = ''  # Disable token for local use (less secure)
c.NotebookApp.password = ''  # Or set password

# Browser settings
c.NotebookApp.open_browser = True
c.NotebookApp.port = 8888

# Auto-save interval (in seconds)
c.FileContentsManager.autosave_interval = 60
```

### A.4.3 Useful Jupyter Extensions

```bash
# Install nbextensions
conda install -c conda-forge jupyter_contrib_nbextensions
jupyter contrib nbextension install --user

# Install configurator
conda install -c conda-forge jupyter_nbextensions_configurator
jupyter nbextensions_configurator enable --user

# Popular extensions to enable:
# - Variable Inspector
# - Code Folding
# - Table of Contents (2)
# - Autopep8
# - ExecuteTime
```

### A.4.4 Jupyter Kernels

```bash
# Add environment as Jupyter kernel
conda activate ml-textbook
python -m ipykernel install --user --name ml-textbook --display-name "ML Textbook"

# List available kernels
jupyter kernelspec list

# Remove kernel
jupyter kernelspec uninstall ml-textbook
```

---

## A.5 Essential Package Installation

### A.5.1 Complete Package List

```bash
# Data manipulation and analysis
conda install numpy pandas

# Visualization
conda install matplotlib seaborn plotly

# Machine learning
conda install scikit-learn

# Deep learning (optional)
conda install tensorflow pytorch

# Statistical analysis
conda install scipy statsmodels

# Jupyter ecosystem
conda install jupyter notebook jupyterlab ipywidgets

# Development tools
conda install autopep8 black flake8

# Additional ML tools
pip install shap lime xgboost lightgbm catboost
```

### A.5.2 Requirements File for This Textbook

Create `requirements.txt`:
```text
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.1.0
jupyter>=1.0.0
notebook>=6.4.0
ipykernel>=6.0.0
plotly>=5.0.0
scipy>=1.7.0
statsmodels>=0.13.0

# Interpretability
shap>=0.40.0
lime>=0.2.0

# Gradient boosting
xgboost>=1.6.0
lightgbm>=3.3.0
catboost>=1.1.0

# Additional utilities
tqdm>=4.60.0
joblib>=1.1.0
pillow>=8.3.0
```

### A.5.3 Installation Script

Create `setup_environment.py`:
```python
#!/usr/bin/env python3
"""
Setup script for ML Textbook environment
"""
import subprocess
import sys
import os

def run_command(command):
    """Run shell command and handle errors"""
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✓ {command}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"✗ {command}")
        print(f"Error: {e.stderr}")
        return None

def setup_conda_environment():
    """Setup conda environment for ML textbook"""
    print("Setting up ML Textbook Environment...")
    
    # Check if conda is available
    if run_command("conda --version") is None:
        print("Conda not found. Please install Anaconda or Miniconda first.")
        return False
    
    # Create environment
    env_command = """
    conda create --name ml-textbook python=3.9 -y &&
    conda activate ml-textbook &&
    conda install -c conda-forge numpy pandas matplotlib seaborn scikit-learn jupyter notebook ipykernel -y &&
    python -m ipykernel install --user --name ml-textbook --display-name "ML Textbook" &&
    pip install shap lime xgboost lightgbm plotly tqdm
    """
    
    if run_command(env_command):
        print("\n✓ Environment setup complete!")
        print("To activate: conda activate ml-textbook")
        print("To start Jupyter: jupyter notebook")
        return True
    else:
        print("✗ Environment setup failed!")
        return False

def verify_installation():
    """Verify that all packages are properly installed"""
    print("\nVerifying installation...")
    
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'seaborn', 
        'sklearn', 'jupyter', 'shap', 'lime'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - Not installed")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        verify_installation()
    else:
        setup_conda_environment()
```

---

## A.6 Common Troubleshooting

### A.6.1 Installation Issues

#### Conda Command Not Found
```bash
# Add conda to PATH (Linux/macOS)
echo 'export PATH="$HOME/anaconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Windows: Add to system PATH through Environment Variables
# C:\Users\username\Anaconda3\Scripts
```

#### Package Installation Fails
```bash
# Update conda
conda update conda

# Clear package cache
conda clean --all

# Use different channels
conda install -c conda-forge package_name

# Use pip as fallback
pip install package_name
```

#### Permission Errors
```bash
# Linux/macOS: Use --user flag
pip install --user package_name

# Windows: Run as Administrator or use --user flag
```

### A.6.2 Jupyter Issues

#### Jupyter Not Starting
```bash
# Check if running
jupyter notebook list

# Kill existing processes
pkill -f jupyter

# Restart with specific port
jupyter notebook --port=8889

# Reset configuration
jupyter notebook --generate-config
```

#### Kernel Issues
```bash
# Refresh kernel list
jupyter kernelspec list

# Install kernel for current environment
python -m ipykernel install --user --name $(basename $CONDA_DEFAULT_ENV)

# Fix kernel connection
pip install --upgrade jupyter jupyter-client
```

### A.6.3 Import Errors

#### Module Not Found
```python
# Check Python path
import sys
print(sys.path)

# Check installed packages
import pkg_resources
installed_packages = [d.project_name for d in pkg_resources.working_set]
print(sorted(installed_packages))
```

#### Version Conflicts
```bash
# Check package versions
conda list package_name
pip show package_name

# Update specific package
conda update package_name
pip install --upgrade package_name

# Force reinstall
pip install --force-reinstall package_name
```

### A.6.4 Environment Issues

#### Environment Not Activating
```bash
# Reinitialize conda
conda init

# Check environment path
conda info --envs

# Recreate environment
conda env remove --name ml-textbook
conda create --name ml-textbook python=3.9
```

---

## A.7 Development Tools Setup

### A.7.1 IDE Configuration

#### VS Code Setup
```bash
# Install VS Code extensions
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter
code --install-extension ms-python.flake8
code --install-extension ms-python.black-formatter
```

#### PyCharm Configuration
- Configure Python interpreter to use conda environment
- Enable Jupyter notebook support
- Install plugins: R Language, Markdown, Database Tools

### A.7.2 Code Formatting and Linting

```bash
# Install formatting tools
pip install black autopep8 flake8 isort

# Format code
black your_script.py
autopep8 --in-place --aggressive your_script.py

# Check style
flake8 your_script.py

# Sort imports
isort your_script.py
```

### A.7.3 Git Configuration

```bash
# Configure Git for Jupyter notebooks
pip install nbstripout

# Remove output from notebooks before committing
nbstripout --install

# Configure Git
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Create .gitignore for Python projects
echo "*.pyc
__pycache__/
.ipynb_checkpoints/
.env
.venv/
*.egg-info/
build/
dist/" > .gitignore
```

---

## A.8 Performance Optimization

### A.8.1 Memory Management

```python
# Monitor memory usage
import psutil
import os

def memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

# Optimize pandas memory usage
import pandas as pd

def optimize_dataframe(df):
    """Optimize DataFrame memory usage"""
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != 'object':
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
    
    return df
```

### A.8.2 Parallel Processing

```python
# Configure joblib for scikit-learn
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Use all available cores
clf = RandomForestClassifier(n_jobs=-1)
grid_search = GridSearchCV(clf, param_grid, n_jobs=-1)

# Configure number of threads
import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
```

---

## A.9 Quick Reference

### A.9.1 Essential Commands

```bash
# Environment management
conda create --name myenv python=3.9
conda activate myenv
conda deactivate
conda env list
conda env remove --name myenv

# Package management
conda install package_name
pip install package_name
conda update package_name
pip install --upgrade package_name

# Jupyter
jupyter notebook
jupyter lab
jupyter kernelspec list
jupyter nbextension enable --py widgetsnbextension
```

### A.9.2 Import Template

```python
# Standard imports for ML projects
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Jupyter notebook settings
%matplotlib inline
%load_ext autoreload
%autoreload 2
```

### A.9.3 Useful Jupyter Magic Commands

```python
# Time execution
%time code_line
%%time
# cell content

# Memory profiling
%memit code_line
%%memit
# cell content

# Load external Python files
%load filename.py
%run filename.py

# System commands
!pip install package_name
!ls
!pwd

# Variable information
%whos
%who_ls

# Debug mode
%debug
%pdb on
```

---

## A.10 Environment Templates

### A.10.1 Basic ML Environment

```yaml
# basic_ml_environment.yml
name: basic-ml
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - numpy
  - pandas
  - matplotlib
  - scikit-learn
  - jupyter
  - notebook
```

### A.10.2 Advanced ML Environment

```yaml
# advanced_ml_environment.yml
name: advanced-ml
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - numpy=1.24.0
  - pandas=2.0.0
  - matplotlib=3.7.0
  - seaborn=0.12.0
  - scikit-learn=1.3.0
  - scipy=1.10.0
  - statsmodels=0.14.0
  - jupyter=1.0.0
  - notebook=6.5.0
  - jupyterlab=3.6.0
  - ipykernel=6.22.0
  - ipywidgets=8.0.0
  - pip=23.0.0
  - pip:
    - shap>=0.41.0
    - lime>=0.2.0
    - plotly>=5.14.0
    - xgboost>=1.7.0
    - lightgbm>=3.3.0
    - catboost>=1.2.0
    - optuna>=3.1.0
    - mlflow>=2.3.0
```

### A.10.3 Deep Learning Environment

```yaml
# deep_learning_environment.yml
name: deep-learning
channels:
  - conda-forge
  - pytorch
  - defaults
dependencies:
  - python=3.9
  - numpy
  - pandas
  - matplotlib
  - scikit-learn
  - jupyter
  - pytorch
  - torchvision
  - tensorflow
  - keras
  - pip:
    - transformers
    - datasets
    - accelerate
```

This completes Appendix A with comprehensive Python environment setup instructions, troubleshooting guides, and templates for different types of machine learning projects.
