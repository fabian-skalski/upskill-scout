"""
Milvus repository implementation for vector operations.
Handles all Milvus database interactions.
"""
from typing import Any, Dict, List, Optional
from pymilvus import MilvusClient
from app.repositories.base import VectorRepository
from app.exceptions.exceptions import MilvusError
from app.utils.logger import setup_logging

logger = setup_logging(__name__)


class MilvusRepository(VectorRepository):
    """Repository for Milvus vector database operations."""
    
    def __init__(self, uri: str, db_name: str, collection_name: str):
        """
        Initialize Milvus repository.
        
        Args:
            uri: Milvus server URI
            db_name: Database name
            collection_name: Collection name
        """
        self.uri = uri
        self.db_name = db_name
        self.collection_name = collection_name
        self._client: Optional[MilvusClient] = None
    
    @property
    def client(self) -> MilvusClient:
        """Get Milvus client, creating if necessary."""
        if self._client is None:
            try:
                self._client = MilvusClient(uri=self.uri, db_name=self.db_name)
                logger.info(f"Connected to Milvus at {self.uri}")
            except Exception as e:
                logger.error(f"Failed to connect to Milvus: {e}")
                raise MilvusError(f"Failed to connect to Milvus: {e}")
        return self._client
    
    async def query(
        self, 
        filters: Dict[str, Any], 
        output_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Query documents by filters."""
        try:
            # Convert filters to Milvus filter expression
            filter_parts = []
            for key, value in filters.items():
                if isinstance(value, str):
                    filter_parts.append(f'{key} == "{value}"')
                else:
                    filter_parts.append(f'{key} == {value}')
            filter_expr = " and ".join(filter_parts)
            
            results = self.client.query(
                collection_name=self.collection_name,
                filter=filter_expr,
                output_fields=output_fields or ["*"]
            )
            
            return results
        except Exception as e:
            logger.error(f"Failed to query documents: {e}")
            raise MilvusError(f"Failed to query documents: {e}")
    
    async def has_collection(self, collection_name: Optional[str] = None) -> bool:
        """Check if collection exists."""
        try:
            return self.client.has_collection(collection_name or self.collection_name)
        except Exception as e:
            logger.error(f"Failed to check collection existence: {e}")
            return False
    
    async def create_collection(
        self, 
        collection_name: str, 
        dimension: int, 
        auto_id: bool = True, 
        enable_dynamic_field: bool = True
    ) -> None:
        """Create a new collection."""
        try:
            self.client.create_collection(
                collection_name=collection_name,
                dimension=dimension,
                auto_id=auto_id,
                enable_dynamic_field=enable_dynamic_field
            )
            logger.info(f"Created collection {collection_name}")
        except Exception as e:
            logger.error(f"Failed to create collection {collection_name}: {e}")
            raise MilvusError(f"Failed to create collection: {e}")