"""
Business logic service for job processing operations.
Handles the core job submission and status checking functionality.
"""
import hashlib
from app.core.models import ProcessedJob, PipelineStep
from app.schemas.api import TextSubmissionRequest, JobStatusResponse, SubmissionResponse
from app.services.airflow import AirflowService
from app.repositories.milvus import MilvusRepository
from app.utils.logger import setup_logging

logger = setup_logging(__name__)


class JobService:
    """Service for managing job processing operations."""
    
    def __init__(
        self,
        airflow_service: AirflowService,
        milvus_repository: MilvusRepository
    ):
        """
        Initialize job service.
        
        Args:
            airflow_service: Service for Airflow operations
            milvus_repository: Repository for Milvus operations
        """
        self.airflow_service = airflow_service
        self.milvus_repository = milvus_repository
    
    async def submit_job(self, request: TextSubmissionRequest) -> SubmissionResponse:
        """
        Submit a new job for processing.
        
        Args:
            request: Text submission request
            
        Returns:
            Submission response with job hash
            
        Raises:
            AirflowError: If job submission to Airflow fails
        """
        # Generate hash for the job
        text_hash = hashlib.sha256(request.description.encode()).hexdigest()
        
        logger.info("Submitting job with hash: %s for user: %s", text_hash, request.user_id)
        
        # Check if this exact job (text_hash + user_id) already exists in Milvus
        existing_job = await self._check_existing_job(text_hash, request.user_id)
        
        if existing_job:
            logger.info("Job %s already exists for user %s, skipping DAG trigger", 
                       text_hash, request.user_id)
            return SubmissionResponse(
                message="Text already processed. Job exists in database.",
                text_hash=text_hash
            )
        
        # Create job payload
        job_payload = ProcessedJob(
            original_text=request.description,
            source_url=request.sourceUrl,
            timestamp=request.timestamp,
            user_id=request.user_id,
            text_hash=text_hash,
            step=PipelineStep.RECEIVED
        )
        
        # Submit to Airflow
        await self.airflow_service.trigger_post_processing_dag(job_payload)
        
        logger.info("Job submitted successfully: %s", text_hash)
        
        return SubmissionResponse(
            message="Text received and processing started.",
            text_hash=text_hash
        )
    
    async def _check_existing_job(self, text_hash: str, user_id: str) -> bool:
        """
        Check if a job with the given text_hash and user_id already exists in Milvus.
        
        Args:
            text_hash: Hash of the job text
            user_id: User identifier
            
        Returns:
            True if job exists, False otherwise
        """
        try:
            # Query for posting records with matching text_hash and user_id
            results = await self.milvus_repository.query(
                filters={
                    "text_hash": text_hash,
                    "user_id": user_id,
                    "entity_type": "posting"
                },
                output_fields=["text_hash"]
            )
            
            return len(results) > 0
        except Exception as e:
            # If query fails, log warning and allow submission to proceed
            # (better to process duplicate than to block legitimate submission)
            logger.warning("Failed to check existing job: %s. Proceeding with submission.", str(e))
            return False
    
    async def get_job_status(self, text_hash: str) -> JobStatusResponse:
        """
        Get job processing status from Airflow.
        
        Args:
            text_hash: Hash of the submitted job
            
        Returns:
            Job status response
        """
        logger.info("Getting status for job: %s", text_hash)
        
        try:
            # Query Airflow for DAG runs with matching text_hash
            dag_runs = await self.airflow_service.get_dag_runs_by_conf(
                dag_id="post_processing_pipeline",
                conf_key="text_hash",
                conf_value=text_hash
            )
            
            if not dag_runs:
                return JobStatusResponse(
                    status="not_found",
                    detail="No pipeline run found for this job"
                )
            
            # Get the most recent run
            latest_run = sorted(dag_runs, key=lambda x: x.get("execution_date", ""), reverse=True)[0]
            state = latest_run.get("state")
            dag_run_id = latest_run.get("dag_run_id")
            
            # Map Airflow states to our status
            if state == "success":
                # Try to get the task result from XCom
                try:
                    job_data = await self.airflow_service.get_task_instance_xcom(
                        dag_id="post_processing_pipeline",
                        dag_run_id=dag_run_id,
                        task_id="step_4_persist"
                    )
                    
                    if job_data:
                        return JobStatusResponse(
                            status="completed",
                            data=job_data
                        )
                except Exception as e:
                    logger.warning("Could not fetch task XCom data: %s", str(e))
                
                return JobStatusResponse(
                    status="completed",
                    detail="Job processing completed successfully"
                )
            
            elif state == "running":
                return JobStatusResponse(
                    status="processing",
                    detail="Job is currently being processed"
                )
            
            elif state == "failed":
                return JobStatusResponse(
                    status="error",
                    detail="Job processing failed"
                )
            
            elif state in ["queued", "scheduled"]:
                return JobStatusResponse(
                    status="processing",
                    detail="Job is queued for processing"
                )
            
            else:
                return JobStatusResponse(
                    status="processing",
                    detail=f"Current state: {state}"
                )
                
        except Exception as e:
            logger.error("Error getting job status for %s: %s", text_hash, str(e))
            return JobStatusResponse(
                status="error",
                detail=str(e)
            )