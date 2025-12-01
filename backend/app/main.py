"""
FastAPI application factory and configuration.
Creates the main application instance with all routes and middleware.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import jobs, overview, llm
from app.schemas.api import HealthResponse
from app.utils.logger import setup_logging
from app.core.config import settings
from app.repositories.milvus import MilvusRepository

logger = setup_logging(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Application lifespan context manager.
    Handles startup and shutdown events.
    
    Args:
        application: FastAPI application instance
    """
    # Startup
    logger.info("Upskill Scout Backend starting up...")
    logger.info("API documentation available at /docs")
    
    # Initialize Milvus collections
    try:
        milvus_repo = MilvusRepository(
            uri=settings.milvus_uri,
            db_name=settings.milvus_db_name,
            collection_name=settings.milvus_collection_name
        )
        
        # Create main collection
        if not await milvus_repo.has_collection(settings.milvus_collection_name):
            logger.info(f"Creating collection: {settings.milvus_collection_name}")
            await milvus_repo.create_collection(
                collection_name=settings.milvus_collection_name,
                dimension=settings.embedding_dim
            )
            
        # Create user skills overview collection
        overview_collection = f"{settings.milvus_collection_name}_overview"
        if not await milvus_repo.has_collection(overview_collection):
            logger.info(f"Creating collection: {overview_collection}")
            await milvus_repo.create_collection(
                collection_name=overview_collection,
                dimension=settings.umap_n_components
            )
            
    except Exception as e:
        logger.error(f"Failed to initialize Milvus collections: {e}")
        # We don't raise here to allow the app to start even if Milvus is down, 
        # but functionality will be degraded.
    
    yield
    
    # Shutdown
    logger.info("Upskill Scout Backend shutting down...")


def create_application() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application instance
    """
    application = FastAPI(
        title="Upskill Scout Backend",
        description="API for skill analysis and extraction",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # TODO Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )
    
    # Include API routes
    application.include_router(jobs.router)
    application.include_router(overview.router)
    application.include_router(llm.router)
    
    # Add global health endpoint
    @application.get("/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        """Global health check endpoint."""
        return HealthResponse(status="ok", service="backend")
    
    return application


# Create the application instance
app = create_application()