#!/usr/bin/env python3
"""
Fix Dependencies Script
Resolves common dependency issues with the Video Analysis Dashboard
"""
import subprocess
import sys
import importlib

def uninstall_and_reinstall_openai():
    """Uninstall and reinstall OpenAI library"""
    print("🔧 Fixing OpenAI library...")
    
    try:
        # Uninstall current version
        print("   Uninstalling current OpenAI library...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', 'openai', '-y'])
        
        # Install latest version
        print("   Installing latest OpenAI library...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'openai>=1.35.0'])
        
        print("✅ OpenAI library updated successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to update OpenAI library: {e}")
        return False

def test_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing imports...")
    
    packages = [
        ('flask', 'Flask'),
        ('flask_cors', 'Flask-CORS'),
        ('requests', 'requests'),
        ('openai', 'OpenAI'),
        ('docx', 'python-docx')
    ]
    
    all_good = True
    for package, name in packages:
        try:
            importlib.import_module(package)
            print(f"✅ {name}")
        except ImportError as e:
            print(f"❌ {name} - {e}")
            all_good = False
    
    return all_good

def test_openai_client():
    """Test OpenAI client initialization"""
    print("🔍 Testing OpenAI client...")
    
    try:
        from openai import AzureOpenAI
        
        # Try to create client (will fail with auth but should not fail with import)
        try:
            client = AzureOpenAI(
                api_version="2024-12-01-preview",
                azure_endpoint="https://test.openai.azure.com/",
                api_key="test-key"
            )
            print("✅ OpenAI client can be initialized")
            return True
        except Exception as e:
            if "proxies" in str(e):
                print(f"❌ OpenAI client has proxies issue: {e}")
                return False
            else:
                print("✅ OpenAI client initialization works (auth error expected)")
                return True
                
    except ImportError as e:
        print(f"❌ Cannot import OpenAI: {e}")
        return False

def reinstall_all_dependencies():
    """Reinstall all dependencies from requirements.txt"""
    print("🔧 Reinstalling all dependencies...")
    
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt', '--upgrade', '--force-reinstall'
        ])
        print("✅ All dependencies reinstalled!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to reinstall dependencies: {e}")
        return False

def main():
    """Main fix function"""
    print("🛠️ Video Analysis Dashboard - Dependency Fixer")
    print("=" * 50)
    
    # Test current state
    print("Step 1: Testing current imports...")
    if test_imports() and test_openai_client():
        print("🎉 All dependencies are working correctly!")
        print("The issue might be in your configuration. Try:")
        print("   1. Check your config.json file")
        print("   2. Test with: python -c 'from video_analyzer import VideoAnalyzer; v = VideoAnalyzer()'")
        return
    
    # Try fixing OpenAI specifically
    print("\nStep 2: Fixing OpenAI library...")
    if not uninstall_and_reinstall_openai():
        print("❌ Failed to fix OpenAI library specifically")
        
        # Try full reinstall
        print("\nStep 3: Reinstalling all dependencies...")
        if not reinstall_all_dependencies():
            print("❌ Failed to fix dependencies")
            print("\nManual fix steps:")
            print("1. pip uninstall openai")
            print("2. pip install openai>=1.35.0")
            print("3. pip install -r requirements.txt --upgrade")
            return
    
    # Test again
    print("\nStep 4: Testing fixes...")
    if test_imports() and test_openai_client():
        print("🎉 Dependencies fixed successfully!")
        print("\nNext steps:")
        print("1. Run: python run.py")
        print("2. Open: http://localhost:8000")
    else:
        print("❌ Some issues remain. Check the output above for specific errors.")

if __name__ == "__main__":
    main()