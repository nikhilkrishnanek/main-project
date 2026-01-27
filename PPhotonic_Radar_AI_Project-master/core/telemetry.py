import sys
import os
import hashlib
import platform
from importlib import util

from core.config import get_config
from core.logger import log_event, init_logging


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def run_startup_checks():
    cfg = get_config()
    init_logging(cfg)

    results = {
        "python_version": True,
        "gpu_available": False,
        "model_valid": True,
        "dataset_present": True,
        "platform_supported": True,
        "degraded": False,
        "messages": []
    }

    # Python version
    min_py = cfg.get("environment", {}).get("min_python", "3.8")
    cur = sys.version_info
    min_parts = tuple(int(x) for x in min_py.split("."))
    if cur < min_parts:
        results["python_version"] = False
        results["degraded"] = True
        msg = f"Python {cur.major}.{cur.minor} < required {min_py}"
        log_event(msg, level="error")
        results["messages"].append(msg)
    else:
        log_event(f"Python version OK: {cur.major}.{cur.minor}")

    # Platform
    plat = platform.system()
    supported = cfg.get("environment", {}).get("supported_platforms", ["Linux", "Windows"])
    if plat not in supported:
        results["platform_supported"] = False
        results["degraded"] = True
        msg = f"Platform {plat} not in supported list: {supported}"
        log_event(msg, level="warning")
        results["messages"].append(msg)
    else:
        log_event(f"Platform supported: {plat}")

    # GPU / CUDA
    try:
        import torch
        gpu = torch.cuda.is_available()
        results["gpu_available"] = bool(gpu)
        log_event(f"GPU available: {gpu}")
    except Exception:
        results["gpu_available"] = False
        log_event("PyTorch not available; GPU checks skipped", level="warning")

    # Model checksum
    model_cfg = cfg.get("model", {})
    model_path = model_cfg.get("path")
    checksum_expected = model_cfg.get("checksum", "")
    if model_path and os.path.exists(model_path):
        if checksum_expected:
            try:
                actual = _sha256(model_path)
                if actual.lower() != checksum_expected.lower():
                    results["model_valid"] = False
                    results["degraded"] = True
                    msg = f"Model checksum mismatch for {model_path}"
                    log_event(msg, level="error")
                    results["messages"].append(msg)
                else:
                    log_event("Model checksum validated")
            except Exception as e:
                results["model_valid"] = False
                results["degraded"] = True
                log_event(f"Error validating model checksum: {e}", level="warning")
        else:
            log_event("No model checksum provided; skipping model validation", level="warning")
    else:
        msg = f"Model file not found: {model_path}"
        log_event(msg, level="warning")
        results["model_valid"] = False
        # Don't force degradation if model missing; allow training from scratch
        results["messages"].append(msg)

    # Dataset presence
    required = cfg.get("dataset", {}).get("required_paths", [])
    missing = []
    for p in required:
        if not os.path.exists(p):
            missing.append(p)
    if missing:
        results["dataset_present"] = False
        results["degraded"] = True
        msg = f"Missing dataset paths: {missing}"
        log_event(msg, level="warning")
        results["messages"].append(msg)
    else:
        log_event("Dataset presence checks passed")

    # Final summary
    if results["degraded"]:
        log_event("Startup checks indicate degraded mode; continuing with graceful degradation", level="warning")
    else:
        log_event("All startup checks passed")

    return results
