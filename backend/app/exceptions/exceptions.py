"""
Custom exceptions for the application.
Provides specific error types for better error handling and user experience.
"""
from typing import Any, Dict, Optional


class UpskillScoutException(Exception):
    """Base exception for all application-specific errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(UpskillScoutException):
    """Raised when input validation fails."""


class ExternalServiceError(UpskillScoutException):
    """Raised when external service calls fail."""


class AirflowError(ExternalServiceError):
    """Raised when Airflow operations fail."""


class LLMServiceError(ExternalServiceError):
    """Raised when LLM service operations fail."""


class DatabaseError(UpskillScoutException):
    """Raised when database operations fail."""


class MilvusError(DatabaseError):
    """Raised when Milvus operations fail."""


class InsufficientDataError(ValidationError):
    """Raised when user doesn't have enough data for processing."""


class ConcurrentOperationError(ValidationError):
    """Raised when user tries to run concurrent operations that aren't allowed."""


class JobNotFoundError(UpskillScoutException):
    """Raised when a job cannot be found."""


class OverviewNotFoundError(UpskillScoutException):
    """Raised when user overview data cannot be found."""