"""
Business logic service for user overview operations.
Handles user skill clustering and overview generation.
"""
import numpy as np
from typing import List
from app.schemas.api import UserOverviewRequest, ClusterInfo, ClusterPoint
from app.services.airflow import AirflowService
from app.repositories.milvus import MilvusRepository
from app.exceptions.exceptions import (
    InsufficientDataError, 
    ConcurrentOperationError, 
    OverviewNotFoundError
)
from app.utils.logger import setup_logging
from app.core.config import settings

logger = setup_logging(__name__)


class OverviewService:
    """Service for managing user overview operations."""
    
    def __init__(
        self,
        airflow_service: AirflowService,
        milvus_repository: MilvusRepository,
        milvus_overview_repository: MilvusRepository
    ):
        """
        Initialize overview service.
        
        Args:
            airflow_service: Service for Airflow operations
            milvus_repository: Repository for main Milvus collection
            milvus_overview_repository: Repository for overview collection
        """
        self.airflow_service = airflow_service
        self.milvus_repository = milvus_repository
        self.milvus_overview_repository = milvus_overview_repository
    
    async def trigger_user_overview(self, request: UserOverviewRequest) -> dict:
        """
        Trigger user overview pipeline generation.
        
        Args:
            request: User overview request
            
        Returns:
            Response message
            
        Raises:
            InsufficientDataError: If user doesn't have enough data
            ConcurrentOperationError: If pipeline already running for user
            AirflowError: If pipeline triggering fails
        """
        user_id = request.user_id
        
        logger.info("Triggering overview for user: %s", user_id)
        
        # Check if user has sufficient data
        await self._check_user_skill_data_sufficiency(user_id)
        
        # Check if there are new skills to process
        has_new_skills = await self._check_for_new_skills(user_id)
        if not has_new_skills:
            logger.info("No new skills for user %s, skipping DAG trigger", user_id)
            return {"message": "No new skills to process. Use GET /overview to retrieve existing results."}
        
        # Check for concurrent operations
        is_running = await self.airflow_service.check_user_skill_overview_running(user_id)
        if is_running:
            raise ConcurrentOperationError(
                f"Overview pipeline already running for user {user_id}. Please wait."
            )
        
        # Trigger the pipeline
        await self.airflow_service.trigger_user_overview_dag(user_id)
        
        logger.info("Overview pipeline started for user: %s", user_id)
        
        return {"message": "Overview pipeline started."}
    
    async def get_user_overview(self, user_id: str) -> List[ClusterInfo]:
        """
        Get user overview results.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of cluster information
            
        Raises:
            OverviewNotFoundError: If overview data not found
        """
        logger.info("Getting overview for user: %s", user_id)
        
        
        # Query overview data using the dedicated repository
        results = await self.milvus_overview_repository.query(
            filters={"user_id": user_id},
            output_fields=["skill_name", "cluster_id", "cluster_description", "vector"]
        )
        
        if not results:
            raise OverviewNotFoundError("No overview data found for this user.")
        
        # Aggregate by cluster
        clusters = {}
        
        for skill in results:
            c_id = skill["cluster_id"]
            c_desc = skill.get("cluster_description", "Unknown")
            
            if c_id not in clusters:
                clusters[c_id] = {
                    "description": c_desc,
                    "count": 0,
                    "skills": [],
                    "umap_points": []
                }
            clusters[c_id]["count"] += 1
            clusters[c_id]["skills"].append(skill["skill_name"])
            
            # Get vector and round to 2 decimal places using numpy (fast)
            vector = skill.get("vector", [])
            if vector:
                # Convert to numpy array, round to 2 decimals, convert back to list
                rounded_coords = np.round(np.array(vector), decimals=2).tolist()
                clusters[c_id]["umap_points"].append(ClusterPoint(
                    coordinates=rounded_coords,
                    name=skill["skill_name"]
                ))
            else:
                logger.warning(
                    "Skill '%s' has no vector data",
                    skill["skill_name"]
                )
        
        # Log cluster descriptions for debugging
        for c_id, data in clusters.items():
            logger.info("Cluster %d description from DB: '%s'", c_id, data["description"])
        
        # Calculate relevancy scores
        total_skills = sum(c["count"] for c in clusters.values())
        cluster_infos = []
        
        for c_id, data in clusters.items():
            if c_id == -1:  # Skip noise cluster
                continue
                
            relevancy = (data["count"] / total_skills) * 100 if total_skills > 0 else 0
            
            cluster_infos.append(ClusterInfo(
                cluster_id=c_id,
                description=data["description"],
                relevancy_score=relevancy,
                skill_count=data["count"],
                umap_points=data["umap_points"]
            ))
        
        logger.info("Retrieved %d clusters for user %s", len(cluster_infos), user_id)
        
        return cluster_infos
    
    async def _check_user_skill_data_sufficiency(self, user_id: str) -> None:
        """
        Check if user has sufficient data for skill overview generation.
        
        Args:
            user_id: User identifier
            
        Raises:
            InsufficientDataError: If user doesn't have enough data
        """
        try:
            # Query user skills from main collection
            # Skills are stored as separate entities with entity_type == "skill"
            results = await self.milvus_repository.query(
                filters={"user_id": user_id, "entity_type": "skill"},
                output_fields=["skill_name", "text_hash"]
            )
            
            # Count unique postings (by text_hash)
            unique_postings = len(set(skill.get("text_hash") for skill in results))
            
            if unique_postings < settings.min_posts_for_overview:
                raise InsufficientDataError(
                    f"Insufficient processed data. User has {unique_postings} "
                    f"processed postings, need at least {settings.min_posts_for_overview}. "
                    f"Please wait for data processing to complete."
                )
                
            logger.info("User %s has %d skills from %d postings", 
                       user_id, len(results), unique_postings)
            
        except Exception as e:
            if isinstance(e, InsufficientDataError):
                raise
            logger.error("Error checking user data sufficiency: %s", str(e))
            raise InsufficientDataError("Too few samples (or collection not found).") from e
    
    async def _check_for_new_skills(self, user_id: str) -> bool:
        """
        Check if there are new skills to process since the last overview.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if there are new skills, False otherwise
        """
        try:
            # Get all current skills
            current_skills = await self.milvus_repository.query(
                filters={"user_id": user_id, "entity_type": "skill"},
                output_fields=["text_hash"]
            )
            
            if not current_skills:
                return False
            
            current_hashes = set(skill.get("text_hash") for skill in current_skills)
            
            # Get previously processed skills from overview collection
            overview_results = await self.milvus_overview_repository.query(
                filters={"user_id": user_id},
                output_fields=["text_hash"]
            )
            
            if not overview_results:
                # No previous overview exists, so all skills are new
                logger.info("No previous overview found for user %s, all skills are new", user_id)
                return True
            
            processed_hashes = set(skill.get("text_hash") for skill in overview_results)
            
            # Check if there are any new hashes
            new_hashes = current_hashes - processed_hashes
            
            logger.info("User %s: %d current skills, %d processed, %d new",
                       user_id, len(current_hashes), len(processed_hashes), len(new_hashes))
            
            return len(new_hashes) > 0
            
        except Exception as e:
            logger.error("Error checking for new skills: %s", str(e))
            # If we can't determine, assume there are new skills to be safe
            return True