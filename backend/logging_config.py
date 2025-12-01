"""
Shared logging configuration.
This file can be imported by both backend and airflow to eliminate duplication.
"""
# Re-export logging setup from the backend for backwards compatibility
from app.utils.logger import setup_logging

__all__ = ['setup_logging']