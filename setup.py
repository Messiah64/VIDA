#!/usr/bin/env python3
"""
Setup script for Video Analysis Dashboard
Helps with initial configuration and dependency installation
"""
import os
import sys
import json
import subprocess
from pathlib import Path

def install_dependencies():
    """Install required Python packages"""
    print("📦 Installing Python dependencies...")
    
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_config():
    """Create configuration file with user input"""
    print("\n⚙️ Setting up configuration...")
    
    print("\nYou'll need credentials from Azure services:")
    print("1. Azure Video Indexer - Account ID and Subscription Key")
    print("2. Azure OpenAI - Endpoint URL and API Key")
    print()
    
    # Video Indexer configuration
    print("Azure Video Indexer Configuration:")
    vi_account_id = input("Enter Video Indexer Account ID: ").strip()
    vi_subscription_key = input("Enter Video Indexer Subscription Key: ").strip()
    vi_location = input("Enter Video Indexer Location (default: trial): ").strip() or "trial"
    
    print("\nAzure OpenAI Configuration:")
    openai_endpoint = input("Enter OpenAI Endpoint URL: ").strip()
    openai_key = input("Enter OpenAI API Key: ").strip()
    openai_deployment = input("Enter GPT-4o Deployment Name (default: gpt-4o): ").strip() or "gpt-4o"
    
    config = {
        "video_indexer": {
            "location": vi_location,
            "account_id": vi_account_id,
            "subscription_key": vi_subscription_key
        },
        "openai": {
            "endpoint": openai_endpoint,
            "model_name": "gpt-4o",
            "deployment": openai_deployment,
            "subscription_key": openai_key,
            "api_version": "2024-12-01-preview"
        },
        "processing": {
            "timeout_seconds": 900,
            "privacy": "Private"
        }
    }
    
    # Save configuration
    try:
        with open('config.json', 'w') as f:
            json.dump(config, f, indent=2)
        print("✅ Configuration saved to config.json")
        return True
    except Exception as e:
        print(f"❌ Failed to save configuration: {e}")
        return False

def create_directories():
    """Create required directories"""
    print("\n📁 Creating required directories...")
    
    directories = ['uploads', 'exports']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created: {directory}/")

def check_existing_config():
    """Check if configuration already exists"""
    if Path('config.json').exists():
        print("\n⚠️  Configuration file already exists!")
        response = input("Do you want to recreate it? (y/n): ")
        return response.lower() != 'y'
    return False

def test_imports():
    """Test if all required packages can be imported"""
    print("\n🔍 Testing package imports...")
    
    packages = [
        ('flask', 'Flask'),
        ('requests', 'requests'),
        ('openai', 'Azure OpenAI'),
        ('docx', 'python-docx'),
        ('pathlib', 'pathlib')
    ]
    
    all_good = True
    for package, name in packages:
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - not installed")
            all_good = False
    
    return all_good

def main():
    """Main setup function"""
    print("🎥 Video Analysis Dashboard - Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed during dependency installation")
        sys.exit(1)
    
    # Test imports
    if not test_imports():
        print("❌ Some packages failed to import. Please check the installation.")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Handle configuration
    if not check_existing_config():
        if not create_config():
            print("❌ Setup failed during configuration")
            sys.exit(1)
    else:
        print("✅ Using existing configuration")
    
    # Final instructions
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Verify your config.json has correct Azure credentials")
    print("2. Run the application with: python run.py")
    print("3. Open http://localhost:8000 in your browser")
    print("\n📖 See README.md for detailed usage instructions")

if __name__ == "__main__":
    main()