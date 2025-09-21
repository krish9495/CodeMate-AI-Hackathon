"""
Deep Researcher Agent v2.0 - Launcher Script
"""
import sys
import subprocess
from pathlib import Path

def main():
    """Launch the Deep Researcher Agent"""
    
    print("🔬 Deep Researcher Agent v2.0")
    print("=" * 50)
    
    # Check if config exists
    env_path = Path(".env")
    if not env_path.exists():
        print("⚠️  No .env file found!")
        print("Please create a .env file with your Gemini API key:")
        print("GEMINI_API_KEY=your_api_key_here")
        return
    
    # Verify API key
    try:
        from dotenv import load_dotenv
        import os
        load_dotenv()
        
        if not os.getenv("GEMINI_API_KEY"):
            print("❌ GEMINI_API_KEY not found in .env file")
            print("Please add your Gemini API key to the .env file")
            return
        
    except ImportError:
        print("❌ python-dotenv not installed. Run: pip install python-dotenv")
        return
    
    print("✅ Configuration looks good!")
    print("🚀 Launching Deep Researcher Agent...")
    print("\n" + "=" * 50)
    
    # Launch Streamlit app
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "deep_researcher_app.py",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Thank you for using Deep Researcher Agent!")
    except Exception as e:
        print(f"❌ Error launching application: {e}")

if __name__ == "__main__":
    main()