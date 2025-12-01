"""
Repository base classes and interfaces.
Provides abstraction layer for data access operations.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class Repository(ABC):
    """Base repository interface for data access operations."""
    pass


class VectorRepository(Repository):
    """Base repository interface for vector database operations."""
        
    @abstractmethod
    async def query(
        self, 
        filters: Dict[str, Any], 
        output_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Query by filters."""