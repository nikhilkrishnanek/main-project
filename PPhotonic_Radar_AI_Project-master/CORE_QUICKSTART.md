# Quick Start Guide - Core Runner

## Overview

The core is now runnable! Use these commands to validate and interact with the Photonic Radar AI system.

## Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Quick Install

```bash
# Install core dependencies
pip install -r requirements.txt

# Note: PyTorch (torch, torchvision, torchaudio) may require additional setup
# See PyTorch documentation: https://pytorch.org/get-started/locally/
```

## Running the Core

### 1. Validate Core ✅

Run comprehensive core validation checks:

```bash
python run_core.py
```

This validates:
- ✅ Project structure (all required directories and files)
- ✅ Core modules (imports, dependencies)
- ✅ Configuration (YAML, users.json)
- ✅ Python syntax (all .py files)

### 2. CLI Commands 🎛️

Use the core CLI for system operations:

```bash
# Show application status
python core_cli.py status

# Show system information
python core_cli.py info

# Run core validation
python core_cli.py validate

# Generate test signal (requires numpy, scipy)
python core_cli.py signal

# Show help
python core_cli.py --help
```

### 3. Run Full Application 🚀

Once dependencies are installed:

```bash
# Start the web UI
python launcher.py

# Or run training
python main.py

# Or run console mode
python app_console.py
```

## Project Structure

```
PPhotonic_Radar_AI_Project-master/
├── run_core.py           ← Core validation script
├── core_cli.py           ← Core CLI interface
├── launcher.py           ← Web UI launcher
├── main.py               ← Training entry point
├── app.py                ← Streamlit web interface
├── app_console.py        ← Console interface
├── config.yaml           ← Configuration
├── requirements.txt      ← Dependencies
├── users.json            ← User database
├── src/                  ← Core library
│   ├── config.py         ← Configuration management
│   ├── logger.py         ← Logging system
│   ├── startup_checks.py ← Startup validation
│   ├── signal_generator.py    ← Signal generation
│   ├── feature_extractor.py   ← Feature extraction
│   ├── model_pytorch.py       ← PyTorch models
│   ├── detection.py          ← Target detection
│   ├── tracker.py            ← Multi-target tracking
│   └── [other modules...]
└── tests/                ← Unit tests
    ├── test_*.py
    └── ...
```

## Core Modules Status

| Module | Status | Purpose |
|--------|--------|---------|
| `config` | ✅ | Configuration management |
| `logger` | ✅ | Structured logging |
| `startup_checks` | ✅ | System validation |
| `signal_generator` | ✅ | Radar signal generation |
| `feature_extractor` | ✅ | Feature extraction |
| `detection` | ✅ | Target detection |
| `tracker` | ✅ | Multi-target tracking |
| `cognitive_controller` | ✅ | Adaptive control |
| `security_core` | ✅ | Security hardening |

## Configuration

Edit `config.yaml` to customize:
- Logging level and output
- Model paths and checksums
- Dataset locations
- Environment settings

## User Management

Users are stored in `users.json`. Default users:
- `admin` - Full system access
- `user` - Standard access

## Troubleshooting

### Missing Packages
```bash
pip install -r requirements.txt
```

### PyTorch Installation Issues
PyTorch is large and may require special handling. See:
https://pytorch.org/get-started/locally/

### Disk Space Issues
The project is ~1.8GB. Ensure sufficient disk space for:
- Dependencies (~2-3GB)
- Models and datasets (~varies)

### Configuration Issues
- Check `config.yaml` syntax
- Ensure `config.yaml` is in the root directory
- Check user permissions on config files

## Next Steps

1. **Validate**: Run `python run_core.py`
2. **Check Status**: Run `python core_cli.py status`
3. **Install**: Run `pip install -r requirements.txt`
4. **Run**: Execute `python launcher.py` or `python main.py`

## Support

For issues, check:
- Log files in `results/` directory
- Configuration in `config.yaml`
- Test files in `tests/` directory

---

**Status**: ✅ Core is runnable and validated
**Last Updated**: 2026-01-20
