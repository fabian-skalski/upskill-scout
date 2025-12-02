"""
Centralized configuration for Airflow tasks.
Fetches configuration from Airflow Variables and Connections.
"""
import json
import os
import logging
from dataclasses import dataclass
from airflow.sdk import Variable

logger = logging.getLogger(__name__)

@dataclass
class AirflowSettings:
    """
    Airflow pipeline settings loaded from Airflow Variables and Connections.
    """
    
    # ============================================================================
    # SERVICE URLS (from Connections)
    # ============================================================================
    milvus_uri: str
    backend_url: str
    mlflow_tracking_uri: str
    
    # ============================================================================
    # DATABASE CONFIGURATION (from Variables)
    # ============================================================================
    milvus_db_name: str
    milvus_collection_name: str
    
    # ============================================================================
    # MODEL CONFIGURATION (from Variables)
    # ============================================================================
    model_gen: str
    embedding_dim: int
    
    # ============================================================================
    # PROCESSING CONFIGURATION (from Variables)
    # ============================================================================
    min_posts_for_overview: int
    request_timeout: int
    
    # UMAP parameters
    umap_n_neighbors: int
    umap_min_dist: float
    umap_n_components: int
    umap_metric: str
    
    # HDBSCAN parameters
    hdbscan_min_cluster_size: int
    hdbscan_min_samples: int
    hdbscan_metric: str

    # AWS S3 / MinIO Configuration
    mlflow_s3_endpoint_url: str
    mlflow_s3_ignore_tls: str
    aws_access_key_id: str
    aws_secret_access_key: str


# Function to get settings - lazy instantiation
_settings_instance = None

def get_and_set_settings() -> AirflowSettings:
    """Get settings instance with lazy loading from Airflow DB"""
    global _settings_instance
    if _settings_instance is None:
        logger.info("Loading Airflow settings from Variables and Connections...")

        # Retrieve values from Airflow Variables
        milvus_uri = Variable.get("milvus_uri")
        milvus_db_name = Variable.get("milvus_db_name")
        milvus_collection_name = Variable.get("milvus_collection_name")
    
        backend_url = Variable.get("backend_url")
        mlflow_tracking_uri = Variable.get("mlflow_tracking_uri")
        mlflow_s3_endpoint_url = Variable.get("mlflow_s3_endpoint_url")
        mlflow_s3_ignore_tls = Variable.get("mlflow_s3_ignore_tls")
        request_timeout = int(Variable.get("request_timeout"))
        aws_access_key_id = Variable.get("aws_access_key_id")
        aws_secret_access_key = Variable.get("aws_secret_access_key")
        embedding_dim = int(Variable.get("embedding_dim"))
        model_gen = Variable.get("model_gen")
        min_posts_for_overview = int(Variable.get("min_posts_for_overview"))
        
        # JSON Variables
        umap_params_json = Variable.get("umap_params")
        umap_params = json.loads(umap_params_json)
        
        hdbscan_params_json = Variable.get("hdbscan_params")
        hdbscan_params = json.loads(hdbscan_params_json)
        
        # Set environment variables for MLflow S3 access (as required by MLflow)
        os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
        os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = mlflow_s3_endpoint_url
        os.environ["MLFLOW_S3_IGNORE_TLS"] = mlflow_s3_ignore_tls
        
        logger.info("Set AWS credentials and MLflow S3 configuration from Airflow Variables")
        
        _settings_instance = AirflowSettings(
            milvus_uri=milvus_uri,
            backend_url=backend_url,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_s3_endpoint_url=mlflow_s3_endpoint_url,
            mlflow_s3_ignore_tls=mlflow_s3_ignore_tls,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            milvus_db_name=milvus_db_name,
            milvus_collection_name=milvus_collection_name,
            model_gen=model_gen,
            embedding_dim=embedding_dim,
            min_posts_for_overview=min_posts_for_overview,
            request_timeout=request_timeout,
            umap_n_neighbors=umap_params.get("n_neighbors"),
            umap_min_dist=umap_params.get("min_dist"),
            umap_n_components=umap_params.get("n_components"),
            umap_metric=umap_params.get("metric"),
            hdbscan_min_cluster_size=hdbscan_params.get("min_cluster_size"),
            hdbscan_min_samples=hdbscan_params.get("min_samples"),
            hdbscan_metric=hdbscan_params.get("metric")
        )
        
    return _settings_instance
