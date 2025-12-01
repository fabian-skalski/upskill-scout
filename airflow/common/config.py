"""
Centralized configuration for Airflow tasks.
All environment variables and constants are defined here.
Validates all required environment variables at startup using Pydantic.
"""
from pydantic_settings import BaseSettings
from pydantic import Field


class AirflowSettings(BaseSettings):
    """
    Airflow pipeline settings loaded from environment variables.
    Will raise ValidationError if required environment variables are missing.
    """
    
    # ============================================================================
    # SERVICE URLS
    # ============================================================================
    
    milvus_uri: str = Field(env="MILVUS_URI")
    backend_url: str = Field(env="BACKEND_URL")
    mlflow_tracking_uri: str = Field(env="MLFLOW_TRACKING_URI")
    
    # ============================================================================
    # DATABASE CONFIGURATION
    # ============================================================================
    
    milvus_db_name: str = Field(env="MILVUS_DB_NAME")
    milvus_collection_name: str = Field(env="MILVUS_COLLECTION_NAME")
    
    # ============================================================================
    # MODEL CONFIGURATION
    # ============================================================================
    
    model_gen: str = Field(env="MODEL_GEN")
    embedding_dim: int = Field(env="EMBEDDING_DIM")
    
    # ============================================================================
    # PROCESSING CONFIGURATION
    # ============================================================================
    
    # Minimum number of posts required for user overview generation
    min_posts_for_overview: int = Field(env="MIN_POSTS_FOR_OVERVIEW")
    
    # UMAP parameters
    umap_n_neighbors: int = Field(env="UMAP_N_NEIGHBORS")
    umap_min_dist: float = Field(env="UMAP_MIN_DIST")
    umap_n_components: int = Field(env="UMAP_N_COMPONENTS")
    umap_metric: str = Field(env="UMAP_METRIC")
    
    # HDBSCAN parameters
    hdbscan_min_cluster_size: int = Field(env="HDBSCAN_MIN_CLUSTER_SIZE")
    hdbscan_min_samples: int = Field(env="HDBSCAN_MIN_SAMPLES")
    hdbscan_metric: str = Field(env="HDBSCAN_METRIC")
    
    # Request timeout (seconds)
    request_timeout: int = Field(env="REQUEST_TIMEOUT")


# Function to get settings - lazy instantiation
_settings_instance = None

def get_settings() -> AirflowSettings:
    """Get settings instance with lazy loading"""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = AirflowSettings()
    return _settings_instance
