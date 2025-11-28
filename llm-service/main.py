from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
import uvicorn
import requests
import re

from logging_config import setup_logging
from constants import (
    USER_PROMPT_EXTRACT_PRIMARY,
    SYSTEM_PROMPT_EXTRACT,
    TITLE_PATTERNS,
    SKILLS_PATTERNS,
    NON_RESPONSE_PHRASES,
    INVALID_TITLES,
    MIN_RESPONSE_LENGTH,
    MIN_SKILL_LENGTH
)

# Setup logging
logger = setup_logging("llm-service")

app = FastAPI()

# Configuration from environment (set in docker-compose.yml)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
MODEL_GEN_NAME = os.getenv("MODEL_GEN")
MODEL_EMB_NAME = os.getenv("MODEL_EMB")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM"))
PROMPT_EXTRACT = os.getenv("PROMPT_EXTRACT")

# LLM Generation Parameters
GEN_TEMPERATURE = float(os.getenv("GEN_TEMPERATURE"))
GEN_MAX_TOKENS = int(os.getenv("GEN_MAX_TOKENS"))
GEN_TIMEOUT = int(os.getenv("GEN_TIMEOUT"))

# Embedding Parameters
EMB_TIMEOUT = int(os.getenv("EMB_TIMEOUT"))

logger.info(f"Ollama URL: {OLLAMA_BASE_URL}")
logger.info(f"Generation model: {MODEL_GEN_NAME}")
logger.info(f"Embedding model: {MODEL_EMB_NAME}")

@app.on_event("startup")
async def startup_event():
    logger.info("LLM Service started successfully!")
    logger.info(f"Using Ollama at: {OLLAMA_BASE_URL}")
    logger.info(f"Generation model: {MODEL_GEN_NAME}")
    logger.info(f"  - Temperature: {GEN_TEMPERATURE}")
    logger.info(f"  - Max tokens: {GEN_MAX_TOKENS}")
    logger.info(f"  - Timeout: {GEN_TIMEOUT}s")
    logger.info(f"Embedding model: {MODEL_EMB_NAME}")
    logger.info(f"  - Dimension: {EMBEDDING_DIM}")
    logger.info(f"  - Timeout: {EMB_TIMEOUT}s")

class ExtractionRequest(BaseModel):
    text: str

class ExtractionResponse(BaseModel):
    title: str
    skills: List[str]

class EmbeddingRequest(BaseModel):
    texts: List[str]

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]

@app.post("/extract", response_model=ExtractionResponse)
async def extract_info(request: ExtractionRequest):
    """Extract occupation title and skills using Ollama generation model with retry logic"""
    max_retries = 3
    last_error = None
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Extraction attempt {attempt + 1}/{max_retries}")
            
            # Format the user prompt with the actual text
            user_prompt = USER_PROMPT_EXTRACT_PRIMARY.format(text=request.text)
            
            # Log what we're sending for debugging
            logger.info(f"Request text length: {len(request.text)}")
            logger.info(f"Request text preview: {request.text[:200]}...")
            logger.info(f"Formatted user prompt preview: {user_prompt[:300]}...")
            
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": MODEL_GEN_NAME,
                    "system": SYSTEM_PROMPT_EXTRACT,
                    "prompt": user_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3 if attempt > 0 else GEN_TEMPERATURE,  # Lower temp on retry
                        "num_predict": GEN_MAX_TOKENS
                    }
                },
                timeout=GEN_TIMEOUT
            )
            response.raise_for_status()
            response_data = response.json()
            output_text = response_data["response"].strip()
            output_text = output_text.lower() if output_text else ""
            
            # Log raw content returned by Ollama
            logger.info(f"Raw Ollama /api/generate response (attempt {attempt + 1}): {output_text}")
            
            # Validate response is not empty or just acknowledgment
            if not output_text or len(output_text) < MIN_RESPONSE_LENGTH:
                logger.warning(f"Empty or too short response on attempt {attempt + 1}")
                last_error = "Empty response from LLM"
                continue
                
            # Check for common non-responses
            if any(phrase in output_text.lower()[:100] for phrase in NON_RESPONSE_PHRASES):
                logger.warning(f"LLM returned acknowledgment instead of extraction on attempt {attempt + 1}")
                last_error = "LLM returned acknowledgment instead of data"
                continue
            
            # Parse the output with multiple patterns
            title = None
            skills = []
            
            # Try different title patterns
            for pattern in TITLE_PATTERNS:
                title_match = re.search(pattern, output_text, re.IGNORECASE)
                if title_match:
                    title = title_match.group(1).strip().strip('*').strip()
                    # Clean up markdown formatting
                    title = re.sub(r'^\*+\s*|\s*\*+$', '', title)
                    break
            
            # Try different skills patterns
            for pattern in SKILLS_PATTERNS:
                skills_match = re.search(pattern, output_text, re.IGNORECASE)
                if skills_match:
                    skills_str = skills_match.group(1).strip()
                    # Parse comma-separated skills
                    skills = [s.strip().lower() for s in re.split(r'[,;]', skills_str) if s.strip()]
                    # Clean up markdown and special chars
                    skills = [re.sub(r'^\*+\s*|\s*\*+$', '', s) for s in skills]
                    skills = [s for s in skills if len(s) > MIN_SKILL_LENGTH]
                    break
            
            # Validate extraction results
            if not title or title.lower() in INVALID_TITLES:
                logger.warning(f"No valid title extracted on attempt {attempt + 1}")
                last_error = "No valid title found in response"
                if attempt < max_retries - 1:
                    continue
                else:
                    title = "Unknown Position"
            
            if not skills:
                logger.warning(f"No skills extracted on attempt {attempt + 1}")
                # Don't retry just for missing skills, but log it
                
            logger.info(f"Successfully extracted - Title: '{title}', Skills: {skills}")
            return ExtractionResponse(title=title, skills=skills)
            
        except Exception as e:
            logger.error(f"Error during extraction attempt {attempt + 1}: {e}")
            last_error = str(e)
            if attempt == max_retries - 1:
                raise HTTPException(status_code=500, detail=f"Extraction failed after {max_retries} attempts: {last_error}")
            continue
    
    # Fallback if all retries failed
    raise HTTPException(status_code=500, detail=f"Extraction failed after {max_retries} attempts: {last_error}")

@app.post("/embed", response_model=EmbeddingResponse)
async def get_embeddings(request: EmbeddingRequest):
    """Generate embeddings using Ollama embedding model"""
    try:
        embeddings = []
        for text in request.texts:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/embed",
                json={
                    "model": MODEL_EMB_NAME,
                    "input": text
                },
                timeout=EMB_TIMEOUT
            )
            response.raise_for_status()
            embedding = response.json()["embeddings"][0]  # Ollama returns embeddings array
            
            # Check if vector is null/zero
            is_null_vector = all(val == 0.0 for val in embedding)
            if is_null_vector:
                logger.warning(f"Null/zero vector returned for embedding of text: {text[:100]}...")
            else:
                logger.info(f"Valid non-zero embedding vector returned (dimension: {len(embedding)})")
            
            # Ensure the embedding has the correct dimension
            if len(embedding) != EMBEDDING_DIM:
                logger.warning(f"Expected {EMBEDDING_DIM} dimensions, got {len(embedding)}")
                # Pad or truncate to match expected dimension
                if len(embedding) < EMBEDDING_DIM:
                    embedding = embedding + [0.0] * (EMBEDDING_DIM - len(embedding))
                else:
                    embedding = embedding[:EMBEDDING_DIM]
            
            embeddings.append(embedding)
        
        return EmbeddingResponse(embeddings=embeddings)
    except Exception as e:
        logger.error(f"Error during embedding: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Check if Ollama is accessible"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return {"status": "ok", "ollama_accessible": response.status_code == 200}
    except Exception as e:
        return {"status": "degraded", "ollama_accessible": False, "error": str(e)}


if __name__ == "__main__":
    # Port from environment for local development (docker-compose sets this)
    port = int(os.getenv("LLM_SERVICE_INTERNAL_PORT"))
    uvicorn.run(app, host="0.0.0.0", port=port)
