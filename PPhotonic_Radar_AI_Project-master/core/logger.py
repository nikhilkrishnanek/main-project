import logging
import os
from logging.handlers import RotatingFileHandler
import json

try:
    from pythonjsonlogger import jsonlogger
except Exception:
    jsonlogger = None

_LOGGER = logging.getLogger("pphotonic")
_INITIALIZED = False
_LOG_FILE = None

def init_logging(cfg=None):
    global _INITIALIZED, _LOG_FILE
    if _INITIALIZED:
        return _LOGGER

    if cfg is None:
        # minimal defaults
        cfg = {"dir": "results", "file": "system.log", "level": "INFO", "max_bytes": 10*1024*1024, "backup_count": 5}
    else:
        cfg = cfg.get("logging", cfg)

    log_dir = cfg.get("dir", "results")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, cfg.get("file", "system.log"))
    _LOG_FILE = log_file

    level = getattr(logging, cfg.get("level", "INFO"))
    _LOGGER.setLevel(level)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)

    # File handler with rotation
    fh = RotatingFileHandler(log_file, maxBytes=cfg.get("max_bytes", 10*1024*1024), backupCount=cfg.get("backup_count", 5))
    fh.setLevel(level)

    if jsonlogger is not None:
        fmt = jsonlogger.JsonFormatter('%(asctime)s %(name)s %(levelname)s %(message)s')
    else:
        fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    ch.setFormatter(fmt)
    fh.setFormatter(fmt)

    _LOGGER.addHandler(ch)
    _LOGGER.addHandler(fh)

    _INITIALIZED = True
    _LOGGER.debug("Structured logging initialized")
    return _LOGGER


def log_event(message, level="info", **kwargs):
    if not _INITIALIZED:
        init_logging()
    data = {"message": message}
    data.update(kwargs)
    if level == "info":
        _LOGGER.info(json.dumps(data) if jsonlogger is None else data)
    elif level == "warning":
        _LOGGER.warning(json.dumps(data) if jsonlogger is None else data)
    elif level == "error":
        _LOGGER.error(json.dumps(data) if jsonlogger is None else data)
    else:
        _LOGGER.debug(json.dumps(data) if jsonlogger is None else data)


def read_logs(lines=20):
    try:
        with open(_LOG_FILE, "r") as f:
            return f.readlines()[-lines:]
    except Exception:
        return ["No logs available"]

