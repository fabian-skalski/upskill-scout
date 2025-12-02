"""
Backend service client for LLM and embedding operations.
Provides clean interfaces for interacting with the backend API.
"""
from typing import List, Tuple
import requests
from common.logging_config import setup_logging
from common.config import get_and_set_settings

logger = setup_logging("backend_client")


class BackendServiceClient:
    """Client for interacting with the backend service."""
    
    def __init__(self, base_url: str = None):
        """
        Initialize the backend service client.
        
        Args:
            base_url: Base URL of the backend service
        """
        settings = get_and_set_settings()
        self.base_url = (base_url or settings.backend_url).rstrip('/')
        self.request_timeout = settings.request_timeout
    
    def extract_job_info(self, text: str) -> Tuple[str, List[str]]:
        """
        Extract occupation title and skills from posting text.
        
        Args:
            text: Posting text to analyze
            
        Returns:
            Tuple of (title, skills_list)
            
        Raises:
            Exception: If the request fails
        """
        try:
            response = requests.post(
                f"{self.base_url}/llm/extract",
                json={"text": text},
                timeout=self.request_timeout
            )
            response.raise_for_status()
            data = response.json()
            return data["title"], data["skills"]
        except Exception as e:
            logger.error("Failed to extract job info: %s", str(e))
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            Exception: If the request fails
        """
        try:
            response = requests.post(
                f"{self.base_url}/llm/embed",
                json={"texts": texts},
                timeout=self.request_timeout
            )
            response.raise_for_status()
            data = response.json()
            return data["embeddings"]
        except Exception as e:
            logger.error("Failed to generate embeddings: %s", str(e))
            raise
    
    def describe_cluster(self, skills: List[str]) -> str:
        """
        Generate a description for a skill cluster.
        
        Args:
            skills: List of skills to describe
            
        Returns:
            Cluster description string
            
        Raises:
            Exception: If the request fails
        """
        try:
            response = requests.post(
                f"{self.base_url}/llm/describe_cluster",
                json={"skills": skills},
                timeout=self.request_timeout
            )
            response.raise_for_status()
            data = response.json()
            return data.get("description", "").strip()
        except Exception as e:
            logger.error("Failed to generate cluster description: %s", str(e))
            raise
