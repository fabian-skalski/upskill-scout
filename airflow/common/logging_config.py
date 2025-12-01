"""
Logging configuration for Airflow tasks.
"""
import logging
import json
import sys
from datetime import datetime, timezone

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "module": record.module,
            "funcName": record.funcName,
            "lineno": record.lineno,
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record)


def setup_logging(name=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    
    # Remove existing handlers to avoid duplication
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.addHandler(handler)
    return logger
