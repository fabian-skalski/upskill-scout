# upskill-scout

Skill analysis and upskilling recommendations powered by LLMs and vector search.

## Quick Start

```bash
cp .env.example .env
docker-compose up
```

**First run:** Downloads Ollama models (~500MB-1GB). Takes 2-5 minutes.

## Architecture

- **Backend** - FastAPI server with analysis endpoints
- **Workers** - Background processors for LLM tasks (3 replicas)
- **LLM Service** - Ollama API wrapper for text generation & embeddings
- **Milvus** - Vector database for semantic search
- **Redis** - Job queue

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
