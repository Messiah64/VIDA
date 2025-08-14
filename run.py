#!/usr/bin/env python3
"""
Simple run script for the Video Analysis Dashboard
"""
import os
import sys
import json
from pathlib import Path

def check_requirements():
    """Check if all required files and dependencies exist"""
    required_files = [
        'index.html',
        'app.py', 
        'video_analyzer.py',
        'docx_exporter.py',
        'configs.json',
        'requirements.txt'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    return True

def check_config():
    """Check if configuration is properly set up"""
    try:
        with open('configs.json', 'r') as f:
            config = json.load(f)
        
        # Check for placeholder values
        vi_key = config.get('video_indexer', {}).get('subscription_key', '')
        openai_key = config.get('openai', {}).get('subscription_key', '')
        
        if 'YOUR_' in vi_key or 'YOUR_' in openai_key:
            print("⚠️  Warning: Configuration contains placeholder values.")
            print("   Please update config.json with your actual Azure credentials.")
            response = input("   Continue anyway? (y/n): ")
            return response.lower() == 'y'
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading config.json: {e}")
        return False

def create_directories():
    """Create required directories"""
    directories = ['uploads', 'exports']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Directory ready: {directory}/")

def check_dependencies():
    """Check if required Python packages are installed"""
    try:
        import flask
        import requests
        import openai
        import docx
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("   Run: pip install -r requirements.txt")
        return False

def main():
    """Main function to run the application"""
    print("🎥 Video Analysis Dashboard")
    print("=" * 40)
    
    # Check requirements
    print("Checking requirements...")
    
    if not check_requirements():
        print("\n❌ Please ensure all required files are present.")
        sys.exit(1)
    
    if not check_dependencies():
        print("\n❌ Please install required dependencies.")
        sys.exit(1)
    
    if not check_config():
        print("\n❌ Configuration check failed.")
        sys.exit(1)
    
    # Create directories
    print("\nSetting up directories...")
    create_directories()
    
    # Start the application
    print("\n🚀 Starting Video Analysis Dashboard...")
    print("   Server will be available at: http://localhost:8000")
    print("   Press Ctrl+C to stop the server")
    print("-" * 40)
    
    try:
        # Import and run the Flask app
        from app import app
        app.run(debug=True, host='0.0.0.0', port=8000)
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()