"""
Worker functions for background processing.
This module contains all the heavy processing logic for the job pipeline.
Functions are designed to be executed by RQ workers, not the API server.
"""
import os
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'  # Needed for macOS multiprocessing with PyTorch

import re
import requests
from redis import Redis
from rq import Queue, Retry
from pymilvus import MilvusClient
from models import ProcessedJob, JobSkill, PipelineStep
from logging_config import setup_logging

# Setup logging
logger = setup_logging("workers")

# Configuration from environment (set in docker-compose.yml)
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT"))
REDIS_DB = int(os.getenv("REDIS_DB"))
MILVUS_URI = os.getenv("MILVUS_URI")
MILVUS_DB_NAME = os.getenv("MILVUS_DB_NAME")
MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME")
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM"))

redis_conn = Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
q = Queue(connection=redis_conn)
q_retries = int(os.getenv("REDIS_RETRIES"))

def setup_milvus():
    """Initialize Milvus collection if it doesn't exist."""
    logger.info(f"Setting up Milvus at {MILVUS_URI} with database {MILVUS_DB_NAME}...")
    client = MilvusClient(uri=MILVUS_URI, db_name=MILVUS_DB_NAME)
    
    if client.has_collection(MILVUS_COLLECTION_NAME):
        logger.info(f"Collection {MILVUS_COLLECTION_NAME} exists.")
        return

    try:
        client.create_collection(
            collection_name=MILVUS_COLLECTION_NAME,
            dimension=EMBEDDING_DIM,
            auto_id=True,
            enable_dynamic_field=True
        )
        logger.info(f"Collection {MILVUS_COLLECTION_NAME} created.")
    except Exception as e:
        res = client.get_load_state(
            collection_name=MILVUS_COLLECTION_NAME
        )
        logger.warning(f"Error creating collection, likely because it was just created by another worker. Load state: {res}. Exception stack: {e}.")
        return


# ============================================================================
# PIPELINE STEP 1: CLEANSE
# ============================================================================
def process_step_1_cleanse(job_data):
    """
    Cleanse the input text by converting to lowercase and removing non-ASCII characters.
    This is the first step in the processing pipeline.
    """
    job = ProcessedJob(**job_data)
    logger.info(f"Step 1 Cleanse: Processing job {job.text_hash}. Input length: {len(job.original_text)}")
    text = job.original_text.lower()
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    job.clean_text = text
    job.step = PipelineStep.CLEANSED
    
    logger.info(f"Step 1 Cleanse: Finished. Clean text length: {len(job.clean_text)}")
    
    # Enqueue next step with timeout
    q.enqueue('workers.process_step_2_extract', job.dict(), retry=Retry(max=q_retries), job_timeout=600)
    logger.info(f"Step 1 Cleanse: Enqueued step 2 for job {job.text_hash}")


# ============================================================================
# PIPELINE STEP 2: EXTRACT
# ============================================================================
def process_step_2_extract(job_data):
    """
    Extract occupation title and skills from the cleansed text using the LLM service.
    This is a heavy processing step that calls an external service.
    """
    job = ProcessedJob(**job_data)
    logger.info(f"Step 2 Extract: Processing job {job.text_hash}. Clean text length: {len(job.clean_text) if job.clean_text else 0}")
    
    try:
        response = requests.post(f"{LLM_SERVICE_URL}/extract", json={"text": job.clean_text})
        response.raise_for_status()
        data = response.json()
        job.title = data["title"]
        job.skills = [JobSkill(name=s) for s in data["skills"]]
        logger.info(f"Step 2 Extract: Extracted title '{job.title}' and {len(job.skills)} skills")
    except Exception as e:
        logger.error(f"Error calling LLM service: {e}")
        job.title = "Error Title"
        job.skills = []

    job.step = PipelineStep.EXTRACTED
    
    # Enqueue next step with timeout
    q.enqueue('workers.process_step_3_embed', job.dict(), retry=Retry(max=q_retries), job_timeout=600)
    logger.info(f"Step 2 Extract: Enqueued step 3 for job {job.text_hash}")


# ============================================================================
# PIPELINE STEP 3: EMBED
# ============================================================================
def process_step_3_embed(job_data):
    """
    Generate embeddings for the clean text and all extracted skills.
    This is a heavy processing step that calls the embedding service.
    """
    job = ProcessedJob(**job_data)
    logger.info(f"Step 3 Embed: Processing job {job.text_hash}. Skills count: {len(job.skills)}")
    
    texts_to_embed = [job.clean_text] + [skill.name for skill in job.skills]
    
    try:
        response = requests.post(f"{LLM_SERVICE_URL}/embed", json={"texts": texts_to_embed})
        response.raise_for_status()
        data = response.json()
        embeddings = data["embeddings"]
        
        job.text_embedding = embeddings[0]
        for i, skill in enumerate(job.skills):
            skill.embedding = embeddings[i + 1]
        logger.info(f"Step 3 Embed: Generated {len(embeddings)} embeddings")
    except Exception as e:
        logger.error(f"Error calling embedding service: {e}")
        job.text_embedding = [0.0] * EMBEDDING_DIM
        for skill in job.skills:
            skill.embedding = [0.0] * EMBEDDING_DIM
    
    job.step = PipelineStep.EMBEDDED
    
    # Enqueue next step with timeout
    q.enqueue('workers.process_step_4_persist', job.dict(), retry=Retry(max=q_retries), job_timeout=600)
    logger.info(f"Step 3 Embed: Enqueued step 4 for job {job.text_hash}")


# ============================================================================
# PIPELINE STEP 4: PERSIST
# ============================================================================
def process_step_4_persist(job_data):
    """
    Persist the processed job data to Milvus vector database.
    This is the final step in the processing pipeline.
    """
    job = ProcessedJob(**job_data)
    logger.info(f"Step 4 Persist: Processing job {job.text_hash}")
    
    try:
        logger.info(f"Step 4 Persist: Creating Milvus client...")
        client = MilvusClient(uri=MILVUS_URI, db_name=MILVUS_DB_NAME)
        logger.info("Step 4 Persist: Milvus client created.")

        if not client.has_collection(MILVUS_COLLECTION_NAME):
            logger.warning(f"Step 4 Persist: Collection {MILVUS_COLLECTION_NAME} not found, creating...")
            setup_milvus()
            # Recreate client after setup
            client = MilvusClient(uri=MILVUS_URI, db_name=MILVUS_DB_NAME)
        
        logger.info(f"Step 4 Persist: Preparing data for insertion...")
        data = [{
            "text_hash": job.text_hash,
            "llm_inferred_title": job.title,
            "original_full_text": job.original_text,
            "source_url": job.source_url,
            "timestamp": job.timestamp,
            "vector": job.text_embedding,
            "step": job.step.value,
            "llm_inferred_skills": [skill.dict() for skill in job.skills]
        }]
        
        logger.info(f"Step 4 Persist: Inserting data into Milvus collection {MILVUS_COLLECTION_NAME}...")
        result = client.insert(collection_name=MILVUS_COLLECTION_NAME, data=data)
        logger.info(f"Step 4 Persist: Milvus insert result: {result}")
        
        job.step = PipelineStep.PERSISTED
        
        # Save status to Redis for the API to query
        logger.info(f"Step 4 Persist: Saving status to Redis...")
        redis_conn.set(f"job:{job.text_hash}", job.json())
        
        logger.info(f"Job {job.text_hash} persisted to Milvus successfully.")
        
    except Exception as e:
        logger.error(f"Step 4 Persist: Failed to persist job {job.text_hash}: {e}")
        # Update status to failed
        redis_conn.set(f"job:{job.text_hash}", job.json())
        raise
