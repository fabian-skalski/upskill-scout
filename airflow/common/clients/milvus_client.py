"""
Milvus database client for vector storage operations.
Provides clean interfaces for collection management and data persistence.
"""
from typing import List, Dict, Any
from pymilvus import MilvusClient
from common.logging_config import setup_logging
from common.config import get_and_set_settings

logger = setup_logging("milvus_client")
settings = get_and_set_settings()

def get_milvus_client() -> MilvusClient:
    """
    Get a configured Milvus client instance.
    
    Returns:
        Configured MilvusClient
    """
    return MilvusClient(uri=settings.milvus_uri, db_name=settings.milvus_db_name)


def ensure_collection_exists() -> None:
    """
    Ensure the Milvus collection exists, creating it if necessary.
    """
    logger.info("Setting up Milvus collection: %s", settings.milvus_collection_name)
    client = get_milvus_client()
    
    if client.has_collection(settings.milvus_collection_name):
        logger.info("Collection %s already exists", settings.milvus_collection_name)
        return
    
    try:
        client.create_collection(
            collection_name=settings.milvus_collection_name,
            dimension=settings.embedding_dim,
            auto_id=True,
            enable_dynamic_field=True
        )
        logger.info("Collection %s created successfully", settings.milvus_collection_name)
    except Exception as e:
        # Collection might have been created by another worker
        logger.warning("Collection creation failed (may already exist): %s", str(e))
        # Try to load the collection
        try:
            result = client.load_collection(settings.milvus_collection_name)
            logger.info("Collection loaded with state: %s", result)
        except Exception as load_error:
            logger.error("Failed to load collection: %s", str(load_error))
            raise


def insert_posting(
    text_hash: str,
    user_id: str,
    title: str,
    clean_text: str,
    text_embedding: List[float],
    skills: List[Dict[str, Any]],
    source_url: str,
    timestamp: str
) -> None:
    """
    Insert a posting and its skills into Milvus.
    
    Args:
        text_hash: Unique hash of the posting
        user_id: User ID who submitted the posting
        title: Title
        clean_text: Cleaned posting text
        text_embedding: Embedding vector for the posting text
        skills: List of skill dictionaries with name and embedding
        source_url: Source URL of the posting
        timestamp: Timestamp when posted
    """
    logger.info("Persisting posting %s to Milvus", text_hash)
    client = get_milvus_client()
    ensure_collection_exists()
    
    # Prepare data for insertion
    data = []
    
    # Add posting record
    data.append({
        "text_hash": text_hash,
        "user_id": user_id,
        "entity_type": "posting",
        "title": title,
        "text": clean_text,
        "vector": text_embedding,
        "source_url": source_url,
        "timestamp": timestamp
    })
    
    # Add skill records
    for skill in skills:
        data.append({
            "text_hash": text_hash,
            "user_id": user_id,
            "entity_type": "skill",
            "skill_name": skill["name"],
            "vector": skill["embedding"],
            "source_url": source_url,
            "timestamp": timestamp
        })
    
    # Insert into Milvus
    result = client.insert(
        collection_name=settings.milvus_collection_name,
        data=data
    )
    
    logger.info("Milvus insert result: %s", result)
    logger.info("Job %s persisted successfully (%d records)", text_hash, len(data))


def query_user_skills(user_id: str) -> List[Dict[str, Any]]:
    """
    Query all skills for a specific user.
    
    Args:
        user_id: User ID to query
        
    Returns:
        List of skill records with embeddings
    """
    logger.info("Querying skills for user: %s", user_id)

    client = get_milvus_client()
    results = client.query(
        collection_name=settings.milvus_collection_name,
        filter=f'user_id == "{user_id}" && entity_type == "skill"',
        output_fields=["skill_name", "vector", "text_hash"]
    )
    
    logger.info("Found %d skills for user %s", len(results), user_id)
    return results


def get_overview_collection_name() -> str:
    """
    Get the name of the overview collection.
    
    Returns:
        Overview collection name
    """
    return f"{settings.milvus_collection_name}_overview"


def ensure_overview_collection_exists() -> None:
    """
    Ensure the Milvus overview collection exists, creating it if necessary.
    The overview collection stores UMAP coordinates and cluster information.
    """
    collection_name = get_overview_collection_name()
    logger.info("Setting up Milvus overview collection: %s", collection_name)
    client = get_milvus_client()
    
    if client.has_collection(collection_name):
        logger.info("Overview collection %s already exists", collection_name)
        return
    
    try:
        client.create_collection(
            collection_name=collection_name,
            dimension=settings.umap_n_components,
            auto_id=True,
            enable_dynamic_field=True
        )
        logger.info("Overview collection %s created successfully", collection_name)
    except Exception as e:
        # Collection might have been created by another worker
        logger.warning("Overview collection creation failed (may already exist): %s", str(e))
        # Try to load the collection
        try:
            result = client.load_collection(collection_name)
            logger.info("Overview collection loaded with state: %s", result)
        except Exception as load_error:
            logger.error("Failed to load overview collection: %s", str(load_error))
            raise


def insert_user_overview(
    user_id: str,
    skill_names: List[str],
    umap_coords: List[List[float]],
    cluster_labels: List[int],
    cluster_descriptions: Dict[int, str],
    text_hashes: List[str] = None
) -> None:
    """
    Insert or update user overview data in Milvus.
    Replaces existing overview data for the user.
    
    Args:
        user_id: User identifier
        skill_names: List of skill names
        umap_coords: List of UMAP coordinates for each skill
        cluster_labels: List of cluster labels for each skill
        cluster_descriptions: Mapping of cluster ID to description
        text_hashes: List of text hashes that were processed (for change tracking)
    """
    logger.info("Persisting overview for user: %s", user_id)
    logger.info("Cluster descriptions to persist: %s", cluster_descriptions)
    
    client = get_milvus_client()
    collection_name = get_overview_collection_name()
    ensure_overview_collection_exists()
    
    # Prepare records for insertion
    records = []
    for idx, (skill, coords, label) in enumerate(zip(skill_names, umap_coords, cluster_labels)):
        label_int = int(label)
        # Try both int and str keys (Airflow XCom may convert dict keys to strings)
        cluster_desc = cluster_descriptions.get(label_int, cluster_descriptions.get(str(label_int), "Unknown"))
        logger.debug("Skill '%s' -> Cluster %d -> Description: '%s'", skill, label_int, cluster_desc)
        
        record = {
            "user_id": user_id,
            "skill_name": skill,
            "cluster_id": label_int,
            "cluster_description": cluster_desc,
            "vector": coords  # UMAP coordinates (dimensionality from config)
        }
        
        # Add text_hash if available for change tracking
        if text_hashes and idx < len(text_hashes):
            record["text_hash"] = text_hashes[idx]
        
        records.append(record)
    
    # Delete existing records for this user
    try:
        client.delete(
            collection_name=collection_name,
            filter=f'user_id == "{user_id}"'
        )
        logger.info("Deleted existing overview data for user %s", user_id)
    except Exception as e:
        logger.warning("Could not delete existing data (may not exist): %s", str(e))
    
    # Insert new records
    result = client.insert(
        collection_name=collection_name,
        data=records
    )
    
    logger.info("Milvus insert result: %s", result)
    logger.info("Persisted %d skill records to overview collection for user %s", 
                len(records), user_id)


def query_user_overview(user_id: str) -> List[Dict[str, Any]]:
    """
    Query overview data for a specific user.
    
    Args:
        user_id: User ID to query
        
    Returns:
        List of overview records with skill names, clusters, and UMAP coordinates
    """
    logger.info("Querying overview for user: %s", user_id)
    
    client = get_milvus_client()
    collection_name = get_overview_collection_name()
    
    # Check if collection exists
    if not client.has_collection(collection_name):
        logger.warning("Overview collection does not exist")
        return []
    
    results = client.query(
        collection_name=collection_name,
        filter=f'user_id == "{user_id}"',
        output_fields=["skill_name", "cluster_id", "cluster_description", "vector", "text_hash"]
    )
    
    logger.info("Found %d overview records for user %s", len(results), user_id)
    return results


def get_processed_text_hashes(user_id: str) -> set:
    """
    Get set of text_hashes that were processed in the last overview run.
    
    Args:
        user_id: User ID to query
        
    Returns:
        Set of text_hash values from the last overview
    """
    logger.info("Getting processed text_hashes for user: %s", user_id)
    
    results = query_user_overview(user_id)
    text_hashes = {r.get("text_hash") for r in results if r.get("text_hash")}
    
    logger.info("Found %d processed text_hashes for user %s", len(text_hashes), user_id)
    return text_hashes
