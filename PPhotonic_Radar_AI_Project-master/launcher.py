#!/usr/bin/env python3
"""
AI Cognitive Photonic Radar - Project Launcher
Main entry point for the radar application.
"""

import subprocess
import sys
import os

# --- SILENCE TENSORFLOW WARNINGS ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide all TF info/warning/error logs


def check_dependencies():
    """Verify required packages are installed."""
    required = ['streamlit', 'torch', 'numpy', 'pandas']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"⚠️  Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    return True


def launch():
    """Launch the Streamlit application."""
    print("🚀 Starting AI Cognitive Photonic Radar Web Interface...")
    print("-" * 60)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU Acceleration Detected: {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ️  Running in CPU mode (No CUDA device found).")
    except ImportError:
        print("ℹ️  Running in CPU optimized mode.")
    
    print("-" * 60)
    print("Opening application at: http://localhost:8501")
    print("Press Ctrl+C to stop the server.\n")
    
    try:
        # Run streamlit app
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "app.py"],
            check=False
        )
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping server...")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error launching application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if check_dependencies():
        launch()
    else:
        sys.exit(1)