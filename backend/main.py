"""
Backend API for Upskill Scout.
Provides HTTP endpoints for job submission and status checking.
Delegates heavy processing to RQ workers via queue_manager.
"""
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import router

from logging_config import setup_logging

# Setup logging
logger = setup_logging("backend")

app = FastAPI(
    title="Upskill Scout Backend",
    description="API for analysis and skill extraction",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    port = int(os.getenv("BACKEND_INTERNAL_PORT", "8000"))
    logger.info(f"Starting backend on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
