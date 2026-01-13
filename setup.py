"""
Setup script - Install all dependencies
Run this first before running the main pipeline
"""

import subprocess # Run commands
import sys # System operations
import os # File operations
from pathlib import Path # Path utilities

def run_command(command, description): # Run shell command
    """Execute command and show progress"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {command}\n")

    result = subprocess.run(command, shell=True) # Run command

    if result.returncode != 0: # Check if failed
        print(f"\n❌ Failed: {description}")
        return False

    print(f"✓ Success: {description}")
    return True

def main(): # Main setup function
    """Install all dependencies"""
    print("\n" + "="*60)
    print("CONTENT MODERATION SYSTEM - DEPENDENCY INSTALLER")
    print("="*60)

    # Check Python version
    print(f"\nPython version: {sys.version}")
    if sys.version_info < (3, 8): # Check version
        print("❌ Python 3.8+ required!")
        return

    # Backend dependencies
    print("\n" + "="*60)
    print("STEP 1: INSTALLING BACKEND DEPENDENCIES")
    print("="*60)

    if not Path('backend/requirements.txt').exists(): # Check if exists
        print("Creating requirements.txt...")
        requirements = """torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
transformers>=4.30.0
detoxify>=0.5.0
gymnasium>=0.28.0
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
pydantic>=2.0.0
tqdm>=4.65.0
""" # Requirements content
        Path('backend').mkdir(exist_ok=True) # Create backend dir
        Path('backend/requirements.txt').write_text(requirements) # Write file
        print("✓ Created requirements.txt")

    # Install backend packages
    if not run_command( # Install pip packages
        f"{sys.executable} -m pip install --upgrade pip", # Upgrade pip
        "Upgrading pip"
    ):
        return

    if not run_command( # Install requirements
        f"{sys.executable} -m pip install -r backend/requirements.txt", # Install from file
        "Installing Python packages (this may take 5-10 minutes)"
    ):
        return

    # Frontend dependencies
    print("\n" + "="*60)
    print("STEP 2: INSTALLING FRONTEND DEPENDENCIES")
    print("="*60)

    # Check if Node.js is installed
    result = subprocess.run("node --version", shell=True, capture_output=True) # Check node
    if result.returncode != 0: # If failed
        print("❌ Node.js not found!")
        print("Please install Node.js from: https://nodejs.org/")
        print("Then run this script again.")
        return

    node_version = result.stdout.decode().strip() # Get version
    print(f"✓ Node.js version: {node_version}")

    # Install frontend packages
    if Path('frontend').exists(): # Check if exists
        original_dir = os.getcwd() # Save current dir
        os.chdir('frontend') # Change to frontend

        if not run_command( # Install npm packages
            "npm install", # Install command
            "Installing frontend packages (this may take 2-3 minutes)"
        ):
            os.chdir(original_dir) # Return to original
            return

        os.chdir(original_dir) # Return to original
    else:
        print("⚠ Frontend directory not found, skipping")

    # Create necessary directories
    print("\n" + "="*60)
    print("STEP 3: CREATING DIRECTORIES")
    print("="*60)

    directories = [ # Directory list
        'backend/data', # Data directory
        'backend/saved_models', # Models directory
    ]

    for directory in directories: # Create each
        Path(directory).mkdir(parents=True, exist_ok=True) # Create dir
        print(f"✓ Created: {directory}")

    # Final summary
    print("\n" + "="*60)
    print("✓ SETUP COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Place train.csv in backend/data/")
    print("2. Run: python run.py")
    print()

if __name__ == "__main__":
    main() # Run setup
