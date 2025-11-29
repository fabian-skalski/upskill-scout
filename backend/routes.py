"""
API Routes for the backend service.
Handles HTTP endpoints and request/response logic.
"""
from fastapi import APIRouter, HTTPException
import hashlib
import requests
from models import TextSubmission, SubmissionResponse, ProcessedJob, PipelineStep
from queue_manager import enqueue_job, get_job_status_from_cache

router = APIRouter()

@router.post("/text", response_model=SubmissionResponse)
async def submit_text(submission: TextSubmission):
    """
    API endpoint to submit text for processing.
    Triggers the Airflow DAG and returns immediately.
    """
    text_hash = hashlib.sha256(submission.description.encode()).hexdigest()
    
    job_payload = ProcessedJob(
        original_text=submission.description,
        source_url=submission.sourceUrl,
        timestamp=submission.timestamp,
        user_id=submission.user_id,
        text_hash=text_hash,
        step=PipelineStep.RECEIVED
    )

    # Trigger Airflow DAG
    airflow_url = "http://airflow-webserver:8080/api/v1/dags/data_processing_pipeline/dagRuns"
    try:
        # Basic Auth for Airflow (default is airflow:airflow)
        response = requests.post(
            airflow_url,
            json={"conf": job_payload.dict()},
            auth=("airflow", "airflow")
        )
        response.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to trigger Airflow DAG: {e}")

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
