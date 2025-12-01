"""
API routes for job-related operations.
Handles HTTP endpoints for job submission and status checking.
"""
from fastapi import APIRouter, Depends, HTTPException
from app.schemas.api import (
    TextSubmissionRequest, 
    SubmissionResponse, 
    JobStatusResponse,
    HealthResponse
)
from app.services.job import JobService
from app.core.dependencies import get_dependency_container, DependencyContainer
from app.exceptions.exceptions import UpskillScoutException
from app.utils.logger import setup_logging

logger = setup_logging(__name__)

router = APIRouter(tags=["jobs"])


def get_job_service(
    container: DependencyContainer = Depends(get_dependency_container)
) -> JobService:
    """Dependency injection for job service."""
    return container.job_service


@router.post("/text", response_model=SubmissionResponse)
async def submit_text(
    request: TextSubmissionRequest,
    job_service: JobService = Depends(get_job_service)
) -> SubmissionResponse:
    """
    Submit text for processing.
    
    Triggers the Airflow DAG and returns immediately with a job hash.
    """
    try:
        return await job_service.submit_job(request)
    except UpskillScoutException as e:
        logger.error("Failed to submit job: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/job/{text_hash}", response_model=JobStatusResponse)
async def get_job_status(
    text_hash: str,
    job_service: JobService = Depends(get_job_service)
) -> JobStatusResponse:
    """
    Checks job processing status.
    """
    try:
        return await job_service.get_job_status(text_hash)
    except UpskillScoutException as e:
        logger.error("Failed to get job status for %s: %s", text_hash, str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint for job service."""
    return HealthResponse(status="ok", service="jobs")