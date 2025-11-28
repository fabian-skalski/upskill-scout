#!/usr/bin/env python3
"""
Start RQ workers to process background jobs.
This script is used by the backend-workers service in docker-compose.
Uses SimpleWorker to avoid forking issues with gRPC connections.
"""
import os
import sys
from redis import Redis
from rq import SimpleWorker, Queue
from workers import setup_milvus
from logging_config import setup_logging

# Setup logging
logger = setup_logging("run_workers")

# Configuration from environment
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
REDIS_DB = int(os.getenv("REDIS_DB"))

listen = ['default']

if __name__ == '__main__':
    # Initialize Milvus
    try:
        setup_milvus()
    except Exception as e:
        logger.error(f"Failed to initialize Milvus: {e}")
        # Continue anyway, worker might be able to recover or fail individual jobs

    redis_conn = Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, db=REDIS_DB)
    
    queues = [Queue(name, connection=redis_conn) for name in listen]
    # Use SimpleWorker to avoid fork() issues with gRPC/Milvus
    worker = SimpleWorker(queues, connection=redis_conn)
    logger.info(f"SimpleWorker started, listening on {listen}")
    worker.work(with_scheduler=False, max_jobs=1000)

