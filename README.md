# Upskill Scout

## Purpose
To analyze and organize **project or job postings that the user is interested in evolving into**, utilizing a sophisticated hybrid AI architecture to identify strategic, personalized growth opportunities.

## Hybrid AI Architecture
This project demonstrates a sophisticated synthesis of AI paradigms, assigning tasks to the models best suited for them:

- **Generative AI (LLMs)**: Utilized for **extraction and nuance**. Local LLMs parse unstructured job data to identify skills with human-level understanding.
- **Classical ML (UMAP + HDBSCAN)**: Utilized for **structure and pattern discovery**. Mathematical density-based clustering provides stable, deterministic insights that LLMs often hallucinate.

By combining these via **Vector Search**, the system achieves a level of privacy, scalability, and analytical depth that neither approach could achieve alone.

## Quick Start
```bash
cp .env.example .env
docker-compose up
```

## Technologies
Built with state-of-the-art technologies for scalable, local-first ML pipelines:
- **Orchestration**: Apache Airflow (Celery Executor)
- **API**: FastAPI (High performance async framework)
- **AI/LLM**: Ollama (Local LLMs for privacy & cost)
- **Vector Search**: Milvus (High-scale vector database)
- **ML Ops**: MLflow (Experiment tracking & model registry)
- **Clustering**: UMAP + HDBSCAN (State-of-the-art unsupervised learning)
- **Infrastructure**: Docker Compose, MinIO (S3), Redis, Fluentd, PostgreSQL

## Author
Fabian Skalski