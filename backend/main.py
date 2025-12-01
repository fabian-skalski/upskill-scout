"""
Application entry point.
Runs the FastAPI application using uvicorn.
"""
import uvicorn
from app.main import app
from app.core.config import settings
from app.utils.logger import setup_logging

logger = setup_logging(__name__)

if __name__ == "__main__":
    port = settings.backend_internal_port
    logger.info(f"Starting Upskill Scout Backend on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
