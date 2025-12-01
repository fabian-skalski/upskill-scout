"""
Service for Airflow operations.
Handles communication with the Airflow API for DAG triggering and status checking.
"""
import requests
import uuid
import time
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta, timezone
from app.core.models import ProcessedJob
from app.exceptions.exceptions import AirflowError
from app.utils.logger import setup_logging

logger = setup_logging(__name__)


class AirflowService:
    """Service for interacting with Airflow API using JWT authentication (Airflow 3.0+)."""
    
    def __init__(self, base_url: str, username: str, password: str):
        """
        Initialize Airflow service.
        
        Args:
            base_url: Airflow API base URL (should include /api/v2)
            username: Airflow username  
            password: Airflow password
        """
        self.base_url = base_url
        self.username = username
        self.password = password
        self._token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None
    
    def _refresh_token(self, attempt: int = 1, max_retries: int = 3) -> str:
        """
        Refresh JWT token with retry logic.
        
        Args:
            attempt: Current attempt number (1-indexed)
            max_retries: Maximum number of retry attempts
            
        Returns:
            JWT token
            
        Raises:
            AirflowError: If token generation fails after all retries
        """
        # Remove /api/v2 from base_url to get auth endpoint
        auth_url = self.base_url.replace("/api/v2", "") + "/auth/token"
        
        last_error = None
        for current_attempt in range(attempt, max_retries + 1):
            try:
                logger.info("Attempting to obtain JWT token (attempt %d/%d)", current_attempt, max_retries)
                response = requests.post(
                    auth_url,
                    json={"username": self.username, "password": self.password},
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )
                response.raise_for_status()
                
                data = response.json()
                token = data["access_token"]
                
                # Set expiry to 23 hours from now (tokens typically last 24h)
                self._token = token
                self._token_expiry = datetime.now() + timedelta(hours=23)
                
                logger.info("Successfully obtained JWT token for Airflow API")
                return token
                
            except requests.exceptions.RequestException as e:
                last_error = e
                logger.warning("Failed to obtain JWT token (attempt %d/%d): %s", current_attempt, max_retries, str(e))
                
                # Wait before retry (exponential backoff)
                if current_attempt < max_retries:
                    wait_time = 2 ** (current_attempt - 1)  # 1s, 2s, 4s, etc.
                    logger.info("Waiting %ds before retry...", wait_time)
                    time.sleep(wait_time)
        
        # All retries failed
        logger.error("Failed to obtain JWT token after %d attempts: %s", max_retries, str(last_error))
        raise AirflowError(f"Failed to authenticate with Airflow after {max_retries} attempts: {last_error}") from last_error
    
    def _get_token(self) -> str:
        """
        Get JWT token for authentication. Caches token until near expiry.
        
        Returns:
            JWT token
            
        Raises:
            AirflowError: If token generation fails
        """
        # Check if we have a valid cached token
        if self._token and self._token_expiry and datetime.now() < self._token_expiry:
            return self._token
        
        # Token expired or not cached, refresh it
        return self._refresh_token()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers with JWT authorization."""
        return {
            "Authorization": f"Bearer {self._get_token()}",
            "Content-Type": "application/json"
        }
    
    def _execute_with_auth_retry(
        self, 
        request_func: Callable[[], requests.Response],
        operation_name: str,
        max_retries: int = 3
    ) -> requests.Response:
        """
        Execute an HTTP request with automatic token refresh on 403 errors.
        
        Args:
            request_func: Function that performs the HTTP request
            operation_name: Name of the operation for logging
            max_retries: Maximum number of retry attempts
            
        Returns:
            Response object
            
        Raises:
            AirflowError: If request fails after all retries
        """
        last_error = None
        
        for attempt in range(1, max_retries + 1):
            try:
                response = request_func()
                response.raise_for_status()
                return response
                
            except requests.exceptions.HTTPError as e:
                last_error = e
                
                # On 403 Forbidden, refresh token and retry
                if e.response.status_code == 403 and attempt < max_retries:
                    logger.warning(
                        "%s failed with 403 Forbidden (attempt %d/%d), refreshing token...",
                        operation_name, attempt, max_retries
                    )
                    # Clear token cache and force refresh
                    self._token = None
                    self._token_expiry = None
                    # Refresh token with remaining retries
                    self._refresh_token(attempt=1, max_retries=max_retries - attempt + 1)
                    continue
                    
                # For other HTTP errors or last attempt, raise immediately
                error_detail = ""
                try:
                    error_detail = f" - {e.response.text}"
                except:
                    pass
                logger.error("%s failed: %s%s", operation_name, str(e), error_detail)
                raise AirflowError(f"{operation_name} failed: {e}{error_detail}") from e
                
            except requests.exceptions.RequestException as e:
                last_error = e
                logger.error("%s failed: %s", operation_name, str(e))
                raise AirflowError(f"{operation_name} failed: {e}") from e
        
        # Should not reach here, but just in case
        raise AirflowError(f"{operation_name} failed after {max_retries} attempts") from last_error
    
    async def trigger_post_processing_dag(self, job_payload: ProcessedJob) -> Dict[str, Any]:
        """
        Trigger the data processing DAG.
        
        Args:
            job_payload: Job data to process
            
        Returns:
            DAG run response
            
        Raises:
            AirflowError: If DAG triggering fails
        """
        url = f"{self.base_url}/dags/post_processing_pipeline/dagRuns"
        
        # Airflow 3.x requires specific payload structure
        # Generate unique DAG run ID to avoid conflicts
        dag_run_id = f"manual__{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        
        payload = {
            "dag_run_id": dag_run_id,
            "logical_date": now,
            "conf": job_payload.model_dump(),
            "note": "Triggered by backend API",
        }
        
        # Execute with automatic authentication retry
        response = self._execute_with_auth_retry(
            request_func=lambda: requests.post(
                url,
                json=payload,
                headers=self._get_headers(),
                timeout=30
            ),
            operation_name="Trigger post processing DAG"
        )
        
        result = response.json()
        logger.info("Triggered data processing DAG: %s", result.get("dag_run_id"))
        return result
    
    async def trigger_user_overview_dag(self, user_id: str) -> Dict[str, Any]:
        """
        Trigger the user overview DAG.
        
        Args:
            user_id: User identifier
            
        Returns:
            DAG run response
            
        Raises:
            AirflowError: If DAG triggering fails
        """
        url = f"{self.base_url}/dags/user_skill_overview_pipeline/dagRuns"
        
        # Airflow 3.x requires specific payload structure
        # Generate unique DAG run ID to avoid conflicts
        dag_run_id = f"manual__{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        
        payload = {
            "dag_run_id": dag_run_id,
            "logical_date": now,
            "conf": {"user_id": user_id},
            "note": f"Triggered by backend API for user {user_id}",
        }
        
        # Execute with automatic authentication retry
        response = self._execute_with_auth_retry(
            request_func=lambda: requests.post(
                url,
                json=payload,
                headers=self._get_headers(),
                timeout=30
            ),
            operation_name="Trigger user overview DAG"
        )
        
        result = response.json()
        logger.info("Triggered user overview DAG: %s", result.get("dag_run_id"))
        return result
    
    async def get_running_dag_runs(self, dag_id: str) -> List[Dict[str, Any]]:
        """
        Get currently running DAG runs for a specific DAG.
        
        Args:
            dag_id: DAG identifier
            
        Returns:
            List of running DAG runs
            
        Raises:
            AirflowError: If request fails
        """
        url = f"{self.base_url}/dags/{dag_id}/dagRuns"
        
        # Execute with automatic authentication retry
        response = self._execute_with_auth_retry(
            request_func=lambda: requests.get(
                url,
                params={"state": "running"},
                headers=self._get_headers(),
                timeout=30
            ),
            operation_name=f"Get running DAG runs for {dag_id}"
        )
        
        result = response.json()
        return result.get("dag_runs", [])
    
    async def check_user_skill_overview_running(self, user_id: str) -> bool:
        """
        Check if user skill overview pipeline is already running for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if pipeline is running for this user
        """
        try:
            running_dags = await self.get_running_dag_runs("user_skill_overview_pipeline")
            
            # Check if any running DAG has the same user_id in conf
            for run in running_dags:
                conf = run.get("conf", {})
                if conf.get("user_id") == user_id:
                    return True
            
            return False
            
        except AirflowError:
            # If we can't check, assume it's not running to avoid blocking users
            logger.warning("Could not check for existing DAG runs for user %s", user_id)
            return False
    
    async def get_dag_runs_by_conf(self, dag_id: str, conf_key: str, conf_value: str) -> List[Dict[str, Any]]:
        """
        Get DAG runs filtered by configuration parameter.
        
        Args:
            dag_id: DAG identifier
            conf_key: Configuration key to filter by
            conf_value: Configuration value to match
            
        Returns:
            List of matching DAG runs
            
        Raises:
            AirflowError: If request fails
        """
        url = f"{self.base_url}/dags/{dag_id}/dagRuns"
        
        # Execute with automatic authentication retry
        response = self._execute_with_auth_retry(
            request_func=lambda: requests.get(
                url,
                headers=self._get_headers(),
                timeout=30
            ),
            operation_name=f"Get DAG runs for {dag_id}"
        )
        
        result = response.json()
        all_runs = result.get("dag_runs", [])
        
        # Filter by conf parameter
        matching_runs = [
            run for run in all_runs
            if run.get("conf", {}).get(conf_key) == conf_value
        ]
        
        return matching_runs
    
    async def get_task_instance_xcom(self, dag_id: str, dag_run_id: str, task_id: str) -> Any:
        """
        Get XCom return value from a task instance.
        
        Args:
            dag_id: DAG identifier
            dag_run_id: DAG run identifier
            task_id: Task identifier
            
        Returns:
            XCom value (typically the task return value)
            
        Raises:
            AirflowError: If request fails
        """
        url = f"{self.base_url}/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}/xcomEntries/return_value"
        
        # Execute with automatic authentication retry
        response = self._execute_with_auth_retry(
            request_func=lambda: requests.get(
                url,
                headers=self._get_headers(),
                timeout=30
            ),
            operation_name=f"Get XCom for task {task_id}"
        )
        
        result = response.json()
        return result.get("value")