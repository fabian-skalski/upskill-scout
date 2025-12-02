import os
import json
import logging
from airflow.sdk import Variable
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_or_update_variable(key, value, description=None):
    try:
        existing_value = Variable.get(key)
        if existing_value != str(value):
            Variable.set(key, value, description=description)
            logger.info(f"Updated variable {key}.")
        else:
            logger.info(f"Variable {key} already up to date.")
    except KeyError:
        # Variable doesn't exist, create it
        Variable.set(key, value, description=description)
        logger.info(f"Created variable {key}.")

def main():
    logger.info("Initializing Airflow Connections and Variables...")

    # --- Variables ---
    # Database Configuration
    create_or_update_variable("milvus_uri", os.environ.get("MILVUS_URI"), "Milvus URI")
    create_or_update_variable("milvus_db_name", os.environ.get("MILVUS_DB_NAME"), "Milvus Database Name")
    create_or_update_variable("milvus_collection_name", os.environ.get("MILVUS_COLLECTION_NAME"), "Milvus Collection Name")

    # HTTP Configuration
    create_or_update_variable("backend_url", os.environ.get("BACKEND_URL"), "Backend API URL")
    create_or_update_variable("mlflow_tracking_uri", os.environ.get("MLFLOW_TRACKING_URI"), "MLflow Tracking URI")
    create_or_update_variable("mlflow_s3_endpoint_url", os.environ.get("MLFLOW_S3_ENDPOINT_URL"), "MLflow S3 Endpoint URL")
    create_or_update_variable("mlflow_s3_ignore_tls", os.environ.get("MLFLOW_S3_IGNORE_TLS"), "MLflow S3 Ignore TLS")
    create_or_update_variable("request_timeout", os.environ.get("REQUEST_TIMEOUT"), "Request timeout in seconds")
    create_or_update_variable("aws_access_key_id", os.environ.get("AWS_ACCESS_KEY_ID"), "AWS Storage Access Key")
    create_or_update_variable("aws_secret_access_key", os.environ.get("AWS_SECRET_ACCESS_KEY"), "AWS Storage Secret Key")

    # LLM Model Configuration
    create_or_update_variable("embedding_dim", os.environ.get("EMBEDDING_DIM"), "Embedding Dimension")
    create_or_update_variable("model_gen", os.environ.get("MODEL_GEN"), "Generation Model Name")

    # Processing Configuration
    create_or_update_variable("min_posts_for_overview", os.environ.get("MIN_POSTS_FOR_OVERVIEW"), "Min posts for overview")

    # UMAP Parameters
    umap_params = {
        "n_neighbors": int(os.environ.get("UMAP_N_NEIGHBORS")),
        "min_dist": float(os.environ.get("UMAP_MIN_DIST")),
        "n_components": int(os.environ.get("UMAP_N_COMPONENTS")),
        "metric": os.environ.get("UMAP_METRIC")
    }
    create_or_update_variable("umap_params", json.dumps(umap_params))

    # HDBSCAN Parameters
    hdbscan_params = {
        "min_cluster_size": int(os.environ.get("HDBSCAN_MIN_CLUSTER_SIZE")),
        "min_samples": int(os.environ.get("HDBSCAN_MIN_SAMPLES")),
        "metric": os.environ.get("HDBSCAN_METRIC")
    }
    create_or_update_variable("hdbscan_params", json.dumps(hdbscan_params))

    logger.info("Initialization complete.")

if __name__ == "__main__":
    main()
