"""
Queue Management for background job processing.
Handles Redis/RQ queue operations and job status tracking.
"""
import os
from redis import Redis
from rq import Queue, Retry
from rq.job import Job
from logging_config import setup_logging
from models import ProcessedJob, PipelineStep

# Setup logging
logger = setup_logging("queue_manager")

# Configuration from environment (set in docker-compose.yml)
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT"))
REDIS_DB = int(os.getenv("REDIS_DB"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
REDIS_RETRIES = int(os.getenv("REDIS_RETRIES"))

# Initialize Redis connection
redis_conn = Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    db=REDIS_DB
)

# Initialize RQ queue
job_queue = Queue(connection=redis_conn)

def enqueue_job(job_payload: ProcessedJob):
    """
    Enqueue a job for background processing.
    
    Args:
        job_payload: ProcessedJob object containing job data
    """
    try:
        # Set job timeout to 10 minutes (600 seconds) to handle slow operations
        job = job_queue.enqueue(
            'workers.process_step_1_cleanse',
            job_payload.dict(),
            retry=Retry(max=REDIS_RETRIES),
            job_timeout=600
        )
        logger.info(f"Job enqueued: {job.id} for text hash {job_payload.text_hash}")
        return job.id
    except Exception as e:
        logger.error(f"Error enqueuing job: {e}")
        return None

def get_job_status(job_id: str):
    """Check the status of a job in the queue."""
    try:
        job = Job.fetch(job_id, connection=redis_conn)
        return job.get_status()
    except Exception as e:
        logger.error(f"Error fetching job status: {e}")
        return "unknown"

def get_job_result(text_hash: str):
    """Retrieve the final result from Redis (if persisted)."""
    try:
        result = redis_conn.get(f"job:{text_hash}")
        if result:
            return result.decode('utf-8')
        return None
    except Exception as e:
        logger.error(f"Error fetching job result: {e}")
        return None

def get_job_status_from_cache(text_hash: str) -> dict:
    """
    Retrieve job status from Redis cache.
    
    Args:
        text_hash: Hash of the submitted text
        
    Returns:
        Dictionary with job status information
    """
    try:
        # Check Redis for job status
        job_data_raw = redis_conn.get(f"job:{text_hash}")
        
        if job_data_raw:
            job_data = ProcessedJob.parse_raw(job_data_raw)
            if job_data.step == PipelineStep.PERSISTED:
                return {"status": "completed", "data": job_data.dict()}
            else:
                return {"status": "processing", "detail": f"Current step: {job_data.step}"}
        else:
            return {"status": "processing", "detail": "Not found in Redis yet"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

def get_redis_connection():
    """Get the Redis connection (for use by other modules)"""
    return redis_conn
