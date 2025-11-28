"""
API Routes for the backend service.
Handles HTTP endpoints and request/response logic.
"""
from fastapi import APIRouter, HTTPException
import hashlib
from models import TextSubmission, SubmissionResponse, ProcessedJob, PipelineStep
from queue_manager import enqueue_job, get_job_status_from_cache

router = APIRouter()

@router.post("/text", response_model=SubmissionResponse)
async def submit_text(submission: TextSubmission):
    """
    API endpoint to submit text for processing.
    Enqueues the job to workers and returns immediately.
    """
    text_hash = hashlib.sha256(submission.description.encode()).hexdigest()
    
    job_payload = ProcessedJob(
        original_text=submission.description,
        source_url=submission.sourceUrl,
        timestamp=submission.timestamp,
        text_hash=text_hash,
        step=PipelineStep.RECEIVED
    )

    # Enqueue job to background workers
    enqueue_job(job_payload)

    return SubmissionResponse(
        message="Text received and processing started.",
        text_hash=text_hash
    )

@router.get("/job/{text_hash}")
async def get_job_status(text_hash: str):
    """
    API endpoint to check job processing status.
    Retrieves status from Redis cache.
    """
    return get_job_status_from_cache(text_hash)

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "backend"}
