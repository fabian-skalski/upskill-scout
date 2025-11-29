# upskill-scout

Skill analysis and upskilling recommendations powered by LLMs and vector search.

## Quick Start

# macOS: Ensure to run e.g. colima Docker with enough (custom) memory, e.g. colima start --memory 16

```bash
cp .env.example .env
docker-compose up
```

**First run:** Downloads Ollama models (~500MB-1GB). Takes 2-5 minutes.

## Architecture

**Services:**
- **Backend** (FastAPI) - API endpoints for job submission
- **Airflow** - Orchestrates 4-step processing pipeline
- **LLM Service** - Ollama wrapper for text generation & embeddings
- **Milvus** - Vector database for semantic skill search
- **Redis** - Job status cache
- **MinIO** - Object storage
- **Fluentd** - Centralized logging

**How It Works:**
1. **Submit** - POST job description to `/text` endpoint
2. **Cleanse** - Lowercase & remove non-ASCII characters
3. **Extract** - LLM extracts job title and skills from text
4. **Embed** - Generate 768-dim vectors for each skill
5. **Persist** - Store skills & embeddings in Milvus for semantic search

Each step runs as an Airflow task, passing data via XCom. Status tracked in Redis.

## Configuration

Edit `.env` to customize:
- `MODEL_GEN` - Text generation model (default: gemma3:270m)
- `MODEL_EMB` - Embedding model (default: embeddinggemma:300m-qat-q8_0)
- `EMBEDDING_DIM` - Vector dimensions (default: 768)
- `GEN_TEMPERATURE` - Generation randomness (default: 0.7)

## Manage Models

```bash
docker exec ollama ollama list              # List models
docker exec ollama ollama pull <model>      # Download model
docker exec ollama ollama rm <model>        # Remove model
```
