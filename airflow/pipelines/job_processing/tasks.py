"""
Task implementations for job processing pipeline.
Each task represents a step in the job processing workflow.
"""
import re
from typing import Dict, Any
from common.logging_config import setup_logging
from common.models import ProcessedJob, PipelineStep, JobSkill
from common.clients.backend_client import BackendServiceClient
from common.clients.milvus_client import insert_posting

logger = setup_logging("job_processing_tasks")


def cleanse_job_text(conf: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cleanse posting text by removing special characters and normalizing.
    
    Args:
        conf: DAG configuration containing job data
        
    Returns:
        Job data with cleansed text
    """
    logger.info("Starting cleanse task")
    
    # Parse the job data from conf
    job = ProcessedJob(**conf)
    
    # Cleanse the text
    clean_text = job.original_text.lower()
    clean_text = re.sub(r'[^\x00-\x7F]+', ' ', clean_text)  # Remove non-ASCII
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()  # Normalize whitespace
    
    job.clean_text = clean_text
    job.step = PipelineStep.CLEANSED
    
    logger.info("Cleansed text for job %s (length: %d -> %d)", 
                job.text_hash, len(job.original_text), len(clean_text))
    
    return job.model_dump()


def extract_job_info(job_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract title and skills from cleansed text using LLM.
    
    Args:
        job_data: Job data with cleansed text
        
    Returns:
        Job data with extracted title and skills
    """
    logger.info("Starting extract task")
    
    job = ProcessedJob(**job_data)
    
    # Use backend client to call LLM extraction
    client = BackendServiceClient()
    title, skill_names = client.extract_job_info(job.clean_text or job.original_text)
    
    job.title = title
    job.skills = [JobSkill(name=skill) for skill in skill_names]
    job.step = PipelineStep.EXTRACTED
    
    logger.info("Extracted from job %s - Title: %s, Skills: %d", 
                job.text_hash, title, len(skill_names))
    
    return job.model_dump()


def embed_job_data(job_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate embeddings for job text and skills.
    
    Args:
        job_data: Job data with extracted skills
        
    Returns:
        Job data with embeddings
    """
    logger.info("Starting embed task")
    
    job = ProcessedJob(**job_data)
    
    # Use backend client to generate embeddings
    client = BackendServiceClient()
    
    # Prepare texts to embed: full text + individual skills
    texts_to_embed = [job.clean_text or job.original_text]
    texts_to_embed.extend([skill.name for skill in job.skills])
    
    logger.info("Generating embeddings for %d texts", len(texts_to_embed))
    embeddings = client.generate_embeddings(texts_to_embed)
    
    # Assign embeddings
    job.text_embedding = embeddings[0]
    for i, skill in enumerate(job.skills):
        skill.embedding = embeddings[i + 1]
    
    job.step = PipelineStep.EMBEDDED
    
    logger.info("Generated embeddings for job %s (text + %d skills)", 
                job.text_hash, len(job.skills))
    
    return job.model_dump()


def persist_job_data(job_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Persist job data and embeddings to Milvus.
    
    Args:
        job_data: Job data with embeddings
        
    Returns:
        Job data (for XCom)
    """
    logger.info("Starting persist task")
    
    job = ProcessedJob(**job_data)
    
    # Prepare skills for Milvus
    skills_for_milvus = [
        {
            "name": skill.name,
            "embedding": skill.embedding
        }
        for skill in job.skills if skill.embedding
    ]
    
    # Insert into Milvus
    insert_posting(
        text_hash=job.text_hash,
        user_id=job.user_id,
        title=job.title or "Unknown",
        clean_text=job.clean_text or job.original_text,
        text_embedding=job.text_embedding or [],
        skills=skills_for_milvus,
        source_url=job.source_url,
        timestamp=job.timestamp
    )
    
    job.step = PipelineStep.PERSISTED
    
    logger.info("Persisted job %s to Milvus", job.text_hash)
    
    return job.model_dump()
