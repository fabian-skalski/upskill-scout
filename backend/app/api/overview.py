"""
API routes for user overview operations.
Handles HTTP endpoints for user skill clustering and overview generation.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List
from app.schemas.api import (
    UserOverviewRequest, 
    ClusterInfo,
    HealthResponse
)
from app.services.overview import OverviewService
from app.core.dependencies import get_dependency_container, DependencyContainer
from app.exceptions.exceptions import (
    UpskillScoutException,
    InsufficientDataError,
    ConcurrentOperationError,
    OverviewNotFoundError
)
from app.utils.logger import setup_logging

logger = setup_logging(__name__)

router = APIRouter(tags=["overview"])


def get_overview_service(
    container: DependencyContainer = Depends(get_dependency_container)
) -> OverviewService:
    """Dependency injection for overview service."""
    return container.overview_service


@router.post("/overview")
async def trigger_user_overview(
    request: UserOverviewRequest,
    overview_service: OverviewService = Depends(get_overview_service)
) -> dict:
    """
    Trigger the user overview pipeline.
    
    Checks if user has at least 10 posts and prevents concurrent runs.
    """
    try:
        return await overview_service.trigger_user_overview(request)
    except InsufficientDataError as e:
        logger.warning("Insufficient data for user %s: %s", request.user_id, str(e))
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ConcurrentOperationError as e:
        logger.warning("Concurrent operation for user %s: %s", request.user_id, str(e))
        raise HTTPException(status_code=409, detail=str(e)) from e
    except UpskillScoutException as e:
        logger.error("Failed to trigger overview for user %s: %s", request.user_id, str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/overview", response_model=List[ClusterInfo])
async def get_user_overview(
    user_id: str = Query(..., description="User identifier"),
    overview_service: OverviewService = Depends(get_overview_service)
) -> List[ClusterInfo]:
    """
    Get the user overview results from the overview collection.
    
    Returns skill clusters with visualization data.
    """
    try:
        return await overview_service.get_user_overview(user_id)
    except OverviewNotFoundError as e:
        logger.warning("Overview not found for user %s: %s", user_id, str(e))
        raise HTTPException(status_code=404, detail=str(e)) from e
    except UpskillScoutException as e:
        logger.error("Failed to get overview for user %s: %s", user_id, str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint for overview service."""
    return HealthResponse(status="ok", service="overview")