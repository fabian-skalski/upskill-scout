"""
Centralized logging configuration.
Provides JSON-formatted structured logging for the application.
"""
import logging
import json
import sys
from datetime import datetime, timezone
from typing import Optional


class JSONFormatter(logging.Formatter):
    """Custom formatter that outputs logs in JSON format."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_record = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
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


def setup_logging(name: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Set up structured JSON logging for a module.
    
    Args:
        name: Logger name (usually module name)
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    
    # Remove existing handlers to avoid duplication
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.addHandler(handler)
    return logger