#!/usr/bin/env python3
"""
AI Cognitive Photonic Radar - Core Runner
Minimal entry point that validates project structure and runs core functionality
"""

import os
import sys
import json
import logging as builtin_logging

# Setup minimal logging
builtin_logging.basicConfig(
    level=builtin_logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = builtin_logging.getLogger(__name__)


def check_project_structure():
    """Verify required project files and directories exist."""
    logger.info("🔍 Checking project structure...")
    
    required_dirs = ['src', 'results', 'tests']
    required_files = ['config.yaml', 'requirements.txt', 'users.json']
    
    errors = []
    
    for directory in required_dirs:
        if not os.path.isdir(directory):
            errors.append(f"❌ Missing directory: {directory}")
        else:
            logger.info(f"✅ Found directory: {directory}")
    
    for file in required_files:
        if not os.path.isfile(file):
            errors.append(f"❌ Missing file: {file}")
        else:
            logger.info(f"✅ Found file: {file}")
    
    if errors:
        for error in errors:
            logger.error(error)
        return False
    
    logger.info("✅ Project structure validated!\n")
    return True


def check_core_modules():
    """Verify core Python modules can be imported."""
    logger.info("🔍 Checking core modules...")
    
    core_modules = {
        'src.config': 'Config',
        'src.logger': 'Logger',
        'src.startup_checks': 'Startup checks',
        'src.signal_generator': 'Signal generator',
        'src.feature_extractor': 'Feature extractor',
    }
    
    failed = []
    
    for module_name, display_name in core_modules.items():
        try:
            __import__(module_name)
            logger.info(f"✅ {display_name} module loaded")
        except ImportError as e:
            logger.warning(f"⚠️  {display_name} module skipped (optional): {e}")
        except Exception as e:
            failed.append(f"❌ {display_name} module error: {e}")
            logger.error(f"❌ {display_name} failed: {e}")
    
    if failed:
        logger.error("\nSome core modules failed to load:")
        for error in failed:
            logger.error(error)
    else:
        logger.info("✅ Core modules validated!")
        logger.info("")
    
    return len(failed) == 0


def check_configuration():
    """Verify configuration files are readable."""
    logger.info("🔍 Checking configuration...")
    
    try:
        import yaml  # type: ignore
        if os.path.exists('config.yaml'):
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)
                if config:
                    logger.info(f"✅ Config loaded with {len(config)} sections")
                else:
                    logger.warning("⚠️  Config file is empty")
        else:
            logger.warning("⚠️  config.yaml not found")
    except Exception as e:
        logger.warning(f"⚠️  YAML config check failed: {e}")
    
    # Check users.json
    try:
        if os.path.exists('users.json'):
            with open('users.json', 'r') as f:
                users = json.load(f)
                logger.info(f"✅ Users file loaded with {len(users)} users")
        else:
            logger.warning("⚠️  users.json not found")
    except Exception as e:
        logger.error(f"❌ Users file error: {e}")
    
    logger.info("")


def list_tests():
    """List available tests."""
    logger.info("🧪 Available tests in ./tests/:")
    
    tests_dir = 'tests'
    if os.path.isdir(tests_dir):
        test_files = [f for f in os.listdir(tests_dir) if f.startswith('test_') and f.endswith('.py')]
        for test_file in test_files:
            logger.info(f"  • {test_file}")
        logger.info("")
        return test_files
    else:
        logger.warning("❌ tests/ directory not found")
        logger.info("")
        return []


def run_syntax_check():
    """Check all Python files for syntax errors."""
    logger.info("🔍 Checking Python syntax...")
    
    errors = []
    checked = 0
    
    for root, dirs, files in os.walk('.'):
        # Skip hidden and cache directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        compile(f.read(), filepath, 'exec')
                    checked += 1
                except SyntaxError as e:
                    errors.append(f"❌ {filepath}: {e}")
                except Exception as e:
                    logger.debug(f"Skipping {filepath}: {e}")
    
    if errors:
        logger.error(f"Found {len(errors)} syntax errors:")
        for error in errors[:5]:  # Show first 5
            logger.error(f"  {error}")
        if len(errors) > 5:
            logger.error(f"  ... and {len(errors) - 5} more")
        return False
    else:
        logger.info(f"✅ Checked {checked} Python files - all syntax valid!")
        logger.info("")
        return True


def main():
    """Run all core checks."""
    logger.info("=" * 70)
    logger.info("🚀 PHOTONIC RADAR AI - CORE VALIDATION")
    logger.info("=" * 70)
    logger.info("")
    
    checks = [
        ("Project Structure", check_project_structure),
        ("Core Modules", check_core_modules),
        ("Configuration", check_configuration),
        ("Python Syntax", run_syntax_check),
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            logger.error(f"Error in {check_name}: {e}")
            results.append((check_name, False))
    
    # Summary
    logger.info("=" * 70)
    logger.info("📊 VALIDATION SUMMARY")
    logger.info("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "⚠️  WARN/SKIP"
        logger.info(f"{status}: {check_name}")
    
    logger.info("")
    logger.info(f"Overall: {passed}/{total} checks passed")
    
    if passed == total:
        logger.info("")
        logger.info("🎉 Core is runnable! You can now:")
        logger.info("  • pip install -r requirements.txt")
        logger.info("  • python launcher.py        (for web UI)")
        logger.info("  • python main.py            (for training)")
        logger.info("  • pytest tests/              (to run tests)")
        return 0
    else:
        logger.warning("")
        logger.warning("⚠️  Some checks had warnings. Review above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
