#!/usr/bin/env python3
"""
AI Cognitive Photonic Radar - CLI Entry Point
Simple command-line interface for core operations
"""

import os
import sys
import argparse


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='AI Cognitive Photonic Radar - Core CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python core_cli.py validate       - Run core validation checks
  python core_cli.py info           - Show system info
  python core_cli.py signal         - Generate test signal
  python core_cli.py status         - Show application status
        """
    )
    
    parser.add_argument(
        'command',
        nargs='?',
        default='validate',
        choices=['validate', 'info', 'signal', 'status', 'help'],
        help='Command to run'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    if args.command == 'help' or (len(sys.argv) == 1):
        parser.print_help()
        return 0
    
    if args.command == 'validate':
        return cmd_validate(args.verbose)
    elif args.command == 'info':
        return cmd_info(args.verbose)
    elif args.command == 'signal':
        return cmd_signal(args.verbose)
    elif args.command == 'status':
        return cmd_status(args.verbose)
    else:
        parser.print_help()
        return 1


def cmd_validate(verbose):
    """Run core validation."""
    print("🔍 Running core validation...")
    import subprocess
    result = subprocess.run(
        [sys.executable, 'run_core.py'],
        capture_output=not verbose
    )
    return result.returncode


def cmd_info(verbose):
    """Show system information."""
    print("📊 System Information")
    print("=" * 60)
    
    import platform
    
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    
    print("\nPython Packages:")
    packages = {
        'numpy': 'numpy',
        'torch': 'torch',
        'torchvision': 'torchvision',
        'streamlit': 'streamlit',
        'cv2': 'opencv',
        'scipy': 'scipy',
    }
    
    for pkg, display_name in packages.items():
        try:
            mod = __import__(pkg)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  ✅ {display_name}: {version}")
        except ImportError:
            print(f"  ❌ {display_name}: not installed")
    
    print("\nGPU Status:")
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            print(f"  ✅ CUDA Available: {torch.cuda.get_device_name(0)}")
            print(f"  ✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("  ℹ️  Running on CPU (No CUDA device)")
    except ImportError:
        print("  ℹ️  PyTorch not installed - install to check GPU status")
    
    return 0


def cmd_signal(verbose):
    """Generate and analyze test signal."""
    print("🎚️  Generating test signal...")
    
    try:
        from src.signal_generator import generate_radar_signal
        from src.feature_extractor import get_all_features
        import numpy as np  # type: ignore
        
        # Generate test signal
        signal_data = generate_radar_signal(
            duration=0.1,
            sample_rate=2000e6,  # 2 GHz
            target_distance=1000,  # 1 km
            target_velocity=100,  # 100 m/s
            noise_level=0.1
        )
        
        print("✅ Signal generated successfully")
        
        if verbose:
            print(f"\nSignal Properties:")
            print(f"  Shape: {signal_data['signal'].shape}")
            print(f"  Mean: {np.mean(np.abs(signal_data['signal'])):.6f}")
            print(f"  Peak: {np.max(np.abs(signal_data['signal'])):.6f}")
            
            # Try to extract features
            features = get_all_features(signal_data['signal'][:1000])
            print(f"\n✅ Features extracted: {len(features)} features")
            if verbose:
                for name, val in list(features.items())[:5]:
                    print(f"    {name}: {val:.6f}")
        
        return 0
    except Exception as e:
        print(f"❌ Error generating signal: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_status(verbose):
    """Show application status."""
    print("📍 Application Status")
    print("=" * 60)
    
    import os
    import json
    from datetime import datetime
    
    checks = {
        'Config file': 'config.yaml',
        'Users file': 'users.json',
        'Model directory': 'results',
        'Tests directory': 'tests',
        'Source directory': 'src',
    }
    
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Working Directory: {os.getcwd()}")
    print("\nStatus Checks:")
    
    all_ok = True
    for name, path in checks.items():
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        print(f"  {status} {name}: {path}")
        if not exists:
            all_ok = False
    
    # Try to load config
    print("\nConfiguration:")
    try:
        import yaml  # type: ignore
        with open('config.yaml', 'r') as f:
            cfg = yaml.safe_load(f)
            sections = len(cfg) if cfg else 0
            print(f"  ✅ Config sections: {sections}")
    except Exception as e:
        print(f"  ❌ Config error: {e}")
        all_ok = False
    
    # Try to load users
    print("\nUsers:")
    try:
        with open('users.json', 'r') as f:
            users = json.load(f)
            print(f"  ✅ Total users: {len(users)}")
    except Exception as e:
        print(f"  ❌ Users error: {e}")
        all_ok = False
    
    print("\n" + "=" * 60)
    if all_ok:
        print("✅ Application ready to run")
        return 0
    else:
        print("⚠️  Some checks failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
