"""
Task implementations for user overview pipeline.
Performs skill clustering and description generation.
"""
import os
import numpy as np
import mlflow
from typing import Dict, Any, List
from umap import UMAP
from hdbscan import HDBSCAN
from pymilvus import MilvusClient

from common.logging_config import setup_logging
from common.config import get_settings
from common.clients.backend_client import BackendServiceClient
from common.clients.milvus_client import (
    query_user_skills,
    insert_user_overview,
    get_processed_text_hashes
)

logger = setup_logging("user_overview_tasks")


def fetch_user_skills(user_id: str) -> Dict[str, Any]:
    """
    Fetch all skills for a user from Milvus.
    Preferably processes only new skills since the last overview run.
    
    Args:
        user_id: User identifier
        
    Returns:
        Dictionary containing user_id, skills, and embeddings
        
    Raises:
        ValueError: If no skills found for user
    """
    logger.info("Fetching skills for user: %s", user_id)
    
    # Query user skills from Milvus
    skills_data = query_user_skills(user_id)
    
    if not skills_data:
        raise ValueError(f"No skills found for user {user_id}")
    
    # Get previously processed text_hashes
    processed_hashes = get_processed_text_hashes(user_id)
    
    
    # Extract skill names and embeddings
    skill_names = [s["skill_name"] for s in skills_data]
    # Convert numpy arrays to native Python lists for XCom serialization
    embeddings = [
        np.asarray(s["vector"]).tolist() for s in skills_data
    ]
    
    logger.info("Fetched %d skills for user %s", len(skill_names), user_id)
    
    return {
        "user_id": user_id,
        "skill_names": skill_names,
        "embeddings": embeddings,
        "text_hashes": [s["text_hash"] for s in skills_data]
    }


def reduce_dimensions(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reduce skill embedding dimensions using UMAP to 3D.
    
    Args:
        data: Data from fetch task
        
    Returns:
        Data with UMAP coordinates added
    """
    logger.info("Reducing dimensions for user: %s", data["user_id"])
    settings = get_settings()

    embeddings = np.array(data["embeddings"])
    
    # Check minimum samples
    if len(embeddings) < settings.min_posts_for_overview:
        raise ValueError(
            f"Insufficient data: {len(embeddings)} skills, "
            f"need at least {settings.min_posts_for_overview}"
        )
    
    # Set MLflow tracking
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    experiment_name = f"{data['user_id']}#user_overview"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name="umap_reduction") as run:
        # Configure UMAP
        n_neighbors = min(settings.umap_n_neighbors, len(embeddings) - 1)
        
        umap_model = UMAP(
            n_neighbors=n_neighbors,
            min_dist=settings.umap_min_dist,
            n_components=settings.umap_n_components,
            metric=settings.umap_metric,
            random_state=42
        )
        
        # Fit and transform
        umap_coords = umap_model.fit_transform(embeddings)
        
        # Log parameters and model to MLflow
        mlflow.log_params({
            "n_neighbors": n_neighbors,
            "min_dist": settings.umap_min_dist,
            "n_components": settings.umap_n_components,
            "metric": settings.umap_metric,
            "n_samples": len(embeddings)
        })
        
        # Register model in MLflow
        model_name = f"{data['user_id']}#umap_model"
        mlflow.sklearn.log_model(
            umap_model,
            "umap_model",
            registered_model_name=model_name
        )
        
        logger.info("UMAP model logged to MLflow for user %s", data['user_id'])
    
    data["umap_coords"] = umap_coords.tolist()
    
    logger.info("Reduced %d embeddings for user %s", len(embeddings), data["user_id"])
    
    return data


def cluster_skills(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cluster skills using HDBSCAN.
    
    Args:
        data: Data with UMAP coordinates
        
    Returns:
        Data with cluster labels added
    """
    logger.info("Clustering skills for user: %s", data["user_id"])
    settings = get_settings()

    umap_coords = np.array(data["umap_coords"])
    
    # Set MLflow tracking
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    experiment_name = f"{data['user_id']}#user_overview"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name="hdbscan_clustering") as run:
        # Configure HDBSCAN
        hdbscan_model = HDBSCAN(
            min_cluster_size=settings.hdbscan_min_cluster_size,
            min_samples=settings.hdbscan_min_samples,
            metric=settings.hdbscan_metric
        )
        
        # Fit
        cluster_labels = hdbscan_model.fit_predict(umap_coords)
        
        # Calculate metrics
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        noise_ratio = (cluster_labels == -1).sum() / len(cluster_labels)
        
        # Log parameters and metrics to MLflow
        mlflow.log_params({
            "min_cluster_size": settings.hdbscan_min_cluster_size,
            "min_samples": settings.hdbscan_min_samples,
            "metric": settings.hdbscan_metric
        })
        
        mlflow.log_metrics({
            "n_clusters": n_clusters,
            "noise_ratio": noise_ratio,
            "n_samples": len(cluster_labels)
        })
        
        # Register model in MLflow
        model_name = f"{data['user_id']}#hdbscan_model"
        mlflow.sklearn.log_model(
            hdbscan_model,
            "hdbscan_model",
            registered_model_name=model_name
        )
        
        logger.info("HDBSCAN model logged to MLflow for user %s (clusters: %d, noise: %.2f%%)", 
                    data['user_id'], n_clusters, noise_ratio * 100)
    
    data["cluster_labels"] = cluster_labels.tolist()
    
    logger.info("Found %d clusters (%.1f%% noise) for user %s", 
                n_clusters, noise_ratio * 100, data["user_id"])
    
    return data


def describe_clusters(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate descriptions for each cluster using LLM.
    
    Args:
        data: Data with cluster labels
        
    Returns:
        Data with cluster descriptions added
    """
    logger.info("Describing clusters for user: %s", data["user_id"])
    settings = get_settings()
    
    cluster_labels = data["cluster_labels"]
    skill_names = data["skill_names"]
    
    # Group skills by cluster
    clusters = {}
    for skill, label in zip(skill_names, cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(skill)
    
    # Set MLflow tracking for LLM tracing
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    experiment_name = f"{data['user_id']}#user_overview"
    mlflow.set_experiment(experiment_name)
    
    # Generate descriptions for each cluster
    client = BackendServiceClient()
    cluster_descriptions = {}
    
    with mlflow.start_run(run_name=f"{data['user_id']}#cluster_description") as run:
        for cluster_id, skills in clusters.items():
            if cluster_id == -1:
                cluster_descriptions[cluster_id] = "Miscellaneous Skills"
                continue
            
            try:
                # Use MLflow tracing for LLM calls
                description = client.describe_cluster(skills[:10])  # Limit to 10 skills
                cluster_descriptions[cluster_id] = description
                
                logger.info("Cluster %d (%d skills): Generated description='%s'", 
                           cluster_id, len(skills), description)
                
                # Log cluster info to MLflow
                mlflow.log_metric(f"cluster_{cluster_id}_size", len(skills))
                mlflow.log_text(description, f"cluster_{cluster_id}_description.txt")
                
            except Exception as e:
                logger.error("Failed to describe cluster %d: %s", cluster_id, str(e))
                fallback = f"Skill Cluster {cluster_id + 1}"
                cluster_descriptions[cluster_id] = fallback
                logger.warning("Using fallback description for cluster %d: %s", cluster_id, fallback)
    
    data["cluster_descriptions"] = cluster_descriptions
    
    logger.info("Generated descriptions for %d clusters for user %s", 
                len(cluster_descriptions), data["user_id"])
    
    return data


def persist_overview(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Persist user overview to Milvus overview collection.
    
    Args:
        data: Complete overview data
        
    Returns:
        Summary data
    """
    logger.info("Persisting overview for user: %s", data["user_id"])
    
    user_id = data["user_id"]
    skill_names = data["skill_names"]
    umap_coords = data["umap_coords"]
    cluster_labels = data["cluster_labels"]
    cluster_descriptions = data["cluster_descriptions"]
    
    # Debug logging
    logger.info("Cluster descriptions being passed to Milvus: %s", cluster_descriptions)
    logger.info("Sample cluster labels: %s", cluster_labels[:10] if len(cluster_labels) > 10 else cluster_labels)
    
    # Use the clean Milvus client interface
    insert_user_overview(
        user_id=user_id,
        skill_names=skill_names,
        umap_coords=umap_coords.tolist() if hasattr(umap_coords, 'tolist') else umap_coords,
        cluster_labels=cluster_labels.tolist() if hasattr(cluster_labels, 'tolist') else cluster_labels,
        cluster_descriptions=cluster_descriptions,
        text_hashes=data.get("text_hashes", [])
    )
    
    logger.info("Successfully persisted overview for user %s", user_id)
    
    # Return summary
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    
    return {
        "user_id": user_id,
        "n_skills": len(skill_names),
        "n_clusters": n_clusters,
        "status": "completed"
    }

