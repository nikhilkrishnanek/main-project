import os
import logging

try:
    import yaml
except Exception:
    yaml = None

_CONFIG = None

def load_config(path=None):
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG

    if path is None:
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")

    defaults = {
        "logging": {
            "level": "INFO",
            "dir": "results",
            "file": "system.log",
            "max_bytes": 10 * 1024 * 1024,
            "backup_count": 5
        },
        "model": {
            "path": "results/radar_model_pytorch.pt",
            "checksum": "",
            "allow_unverified": True
        },
        "dataset": {"required_paths": ["data/train", "data/val"]},
        "environment": {"min_python": "3.8", "supported_platforms": ["Linux", "Windows"]}
    }

    cfg = defaults.copy()
    if os.path.exists(path) and yaml is not None:
        try:
            with open(path, "r") as f:
                file_cfg = yaml.safe_load(f) or {}
                # shallow update
                for k, v in file_cfg.items():
                    cfg[k] = v
        except Exception as e:
            logging.warning(f"Failed to load config {path}: {e}")
    else:
        if os.path.exists(path) and yaml is None:
            logging.warning("PyYAML not installed; using defaults for config.yaml")

    _CONFIG = cfg
    return _CONFIG

def get_config():
    return load_config()
