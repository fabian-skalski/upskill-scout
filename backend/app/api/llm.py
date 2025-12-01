"""
API endpoints for LLM operations.
Provides extraction, embedding, and cluster description functionality.
"""
from app.core.models import ExtractionRequest, ExtractionResponse, EmbeddingRequest, EmbeddingResponse, ClusterDescriptionRequest, ClusterDescriptionResponse
from fastapi import APIRouter, HTTPException
from app.services.llm import LLMService
from app.schemas.api import HealthResponse
from app.utils.logger import setup_logging

logger = setup_logging(__name__)

router = APIRouter(
    prefix="/llm",
    tags=["llm"]
)

# Initialize service
llm_service = LLMService()


@router.post("/extract", response_model=ExtractionResponse)
async def extract_info(request: ExtractionRequest):
    """
    Extract occupation title and skills from posting text.
    
    Args:
        request: Text to extract information from
        
    Returns:
        Extracted title and skills
    """
    try:
        logger.info("Extracting info from text (length: %d)", len(request.text))
        title, skills = await llm_service.extract_info(request.text)
        return ExtractionResponse(title=title, skills=skills)
    except Exception as e:
        logger.error("Extraction failed: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@router.post("/embed", response_model=EmbeddingResponse)
async def get_embeddings(request: EmbeddingRequest):
    """
    Generate embeddings for a list of texts.
    
    Args:
        request: List of texts to embed
        
    Returns:
        List of embedding vectors
    """
    try:
        logger.info("Generating embeddings for %d texts", len(request.texts))
        embeddings = await llm_service.get_embeddings(request.texts)
        return EmbeddingResponse(embeddings=embeddings)
    except Exception as e:
        logger.error("Embedding generation failed: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")


@router.post("/describe_cluster", response_model=ClusterDescriptionResponse)
async def describe_cluster(request: ClusterDescriptionRequest):
    """
    Generate a concise description for a skill cluster.
    
    Args:
        request: List of skills to describe
        
    Returns:
        Cluster description
    """
    try:
        logger.info("Generating cluster description for %d skills", len(request.skills))
        description = await llm_service.describe_cluster(request.skills)
        return ClusterDescriptionResponse(description=description)
    except Exception as e:
        logger.error("Cluster description failed: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Cluster description failed: {str(e)}")


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check LLM service health."""
    return HealthResponse(status="ok", service="llm")
