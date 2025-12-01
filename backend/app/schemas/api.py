"""
API request and response schemas.
Defines the structure of HTTP requests and responses separate from domain models.
"""
from pydantic import BaseModel, Field
from typing import List, Optional


# Request Schemas
class TextSubmissionRequest(BaseModel):
    """Request model for text submission."""
    description: str = Field(..., description="Posting text", min_length=1)
    sourceUrl: str = Field(..., description="Source URL of the posting")
    timestamp: str = Field(..., description="ISO timestamp")
    user_id: str = Field(..., description="User identifier")


class UserOverviewRequest(BaseModel):
    """Request model for user overview generation."""
    user_id: str = Field(..., description="User identifier")


# Response Schemas
class SubmissionResponse(BaseModel):
    """Response model for text submission."""
    message: str = Field(..., description="Response message")
    text_hash: str = Field(..., description="Hash of submitted text")


class JobStatusResponse(BaseModel):
    """Response model for job status."""
    status: str = Field(..., description="Job status")
    detail: Optional[str] = Field(None, description="Additional status details")
    data: Optional[dict] = Field(None, description="Job data if completed")


class HealthResponse(BaseModel):
    """Response model for health checks."""
    status: str = Field(..., description="Service status")
    service: Optional[str] = Field(None, description="Service name")


class ClusterPoint(BaseModel):
    """Model for cluster visualization points with arbitrary dimensions."""
    coordinates: List[float] = Field(..., description="UMAP coordinates (truncated to 2 decimals)")
    name: str = Field(..., description="Skill name")


class ClusterInfo(BaseModel):
    """Model for cluster information."""
    cluster_id: int = Field(..., description="Cluster identifier")
    description: str = Field(..., description="Cluster description")
    relevancy_score: float = Field(..., description="Relevancy score as percentage")
    skill_count: int = Field(..., description="Number of skills in cluster")
    umap_points: List[ClusterPoint] = Field(..., description="Visualization points")