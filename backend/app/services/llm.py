"""
LLM service for text extraction, embeddings, and cluster descriptions.
Handles all interactions with the Ollama LLM API.
"""
import re
import requests
import mlflow
from typing import List, Tuple
from app.core.config import settings
from app.utils.logger import setup_logging

logger = setup_logging(__name__)

# Import constants
from app.core.llm_constants import (
    USER_PROMPT_EXTRACT_PRIMARY,
    SYSTEM_PROMPT_EXTRACT,
    TITLE_PATTERNS,
    SKILLS_PATTERNS,
    NON_RESPONSE_PHRASES,
    INVALID_TITLES,
    MIN_RESPONSE_LENGTH,
    MIN_SKILL_LENGTH,
    USER_PROMPT_EXTRACT_CLUSTER_DESCRIPTION
)


class LLMService:
    """Service for LLM-based text analysis and embeddings."""
    
    def __init__(self):
        """Initialize the LLM service with configuration from settings."""
        self.ollama_base_url = settings.ollama_base_url
        self.model_gen = settings.model_gen
        self.model_emb = settings.model_emb
        self.embedding_dim = settings.embedding_dim
        self.gen_temperature = settings.gen_temperature
        self.gen_max_tokens = settings.gen_max_tokens
        self.gen_max_retries = settings.gen_max_retries
        self.request_timeout = settings.request_timeout

        # Initialize MLflow
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        
        logger.info(f"LLM Service initialized with Ollama at: {self.ollama_base_url}")
        logger.info(f"Generation model: {self.model_gen}, Embedding model: {self.model_emb}")
        logger.info(f"MLflow tracking URI: {settings.mlflow_tracking_uri}")
    
    def get_temperature(self, curr_attempt: int) -> float:
        """
        Calculate temperature for current attempt, scaling linearly from gen_temperature to 0.0.
        
        Args:
            curr_attempt: Current attempt number (0-indexed)
            
        Returns:
            Temperature value for this attempt
        """
        if self.gen_max_retries <= 1:
            return self.gen_temperature
        
        # Scale linearly: gen_temperature -> 0.0 over gen_max_retries steps
        temperature = self.gen_temperature * (1 - curr_attempt / (self.gen_max_retries - 1))
        return temperature
    
    async def extract_info(self, text: str) -> Tuple[str, List[str]]:
        """
        Extract occupation title and skills from posting text.
        
        Args:
            text: Posting text to analyze
            
        Returns:
            Tuple of (title, skills_list)
            
        Raises:
            Exception: If extraction fails after max retries
        """
        last_error = None
        
        for attempt in range(self.gen_max_retries):
            try:
                logger.info(f"Extraction attempt {attempt + 1}/{self.gen_max_retries}")
                
                # Format the user prompt with the actual text
                user_prompt = USER_PROMPT_EXTRACT_PRIMARY.format(text=text)
                
                logger.info(f"Request text length: {len(text)}")
                logger.info(f"Request text preview: {text[:200]}...")
                
                response = requests.post(
                    f"{self.ollama_base_url}/api/generate",
                    json={
                        "model": self.model_gen,
                        "system": SYSTEM_PROMPT_EXTRACT,
                        "prompt": user_prompt,
                        "stream": False,
                        "options": {
                            "temperature": self.get_temperature(attempt),
                            "num_predict": self.gen_max_tokens
                        }
                    },
                    timeout=self.request_timeout
                )
                response.raise_for_status()
                response_data = response.json()
                output_text = response_data["response"].strip()
                output_text = output_text.lower() if output_text else ""
                
                logger.info(f"Raw Ollama response (attempt {attempt + 1}): {output_text}")
                
                # Validate response
                if not output_text or len(output_text) < MIN_RESPONSE_LENGTH:
                    logger.warning(f"Empty or too short response on attempt {attempt + 1}")
                    last_error = "Empty response from LLM"
                    continue
                    
                # Check for non-responses
                if any(phrase in output_text.lower()[:100] for phrase in NON_RESPONSE_PHRASES):
                    logger.warning(f"LLM returned acknowledgment instead of extraction")
                    last_error = "LLM returned acknowledgment instead of data"
                    continue
                
                # Parse the output
                title = None
                skills = []
                
                # Try different title patterns
                for pattern in TITLE_PATTERNS:
                    title_match = re.search(pattern, output_text, re.IGNORECASE)
                    if title_match:
                        title = title_match.group(1).strip().strip('*').strip()
                        title = re.sub(r'^\*+\s*|\s*\*+$', '', title)
                        break
                
                # Try different skills patterns
                for pattern in SKILLS_PATTERNS:
                    skills_match = re.search(pattern, output_text, re.IGNORECASE)
                    if skills_match:
                        skills_str = skills_match.group(1).strip()
                        skills = [s.strip().lower() for s in re.split(r'[,;]', skills_str) if s.strip()]
                        skills = [re.sub(r'^\*+\s*|\s*\*+$', '', s) for s in skills]
                        skills = [s for s in skills if len(s) > MIN_SKILL_LENGTH]
                        break
                
                # Validate extraction results
                if not title or title.lower() in INVALID_TITLES:
                    logger.warning(f"No valid title extracted on attempt {attempt + 1}")
                    last_error = "No valid title found in response"
                    if attempt < self.gen_max_retries - 1:
                        continue
                    else:
                        title = "Unknown Position"
                
                if not skills:
                    logger.warning(f"No skills extracted on attempt {attempt + 1}")
                    
                logger.info(f"Successfully extracted - Title: '{title}', Skills: {skills}")
                return title, skills
                
            except Exception as e:
                logger.error(f"Error during extraction attempt {attempt + 1}: {e}")
                last_error = str(e)
                if attempt == self.gen_max_retries - 1:
                    raise Exception(f"Extraction failed after {self.gen_max_retries} attempts: {last_error}")
                continue
        
        raise Exception(f"Extraction failed after {self.gen_max_retries} attempts: {last_error}")
    
    async def describe_cluster(self, skills: List[str]) -> str:
        """
        Generate a concise description for a skill cluster.
        
        Args:
            skills: List of skills to describe
            
        Returns:
            Cluster description string
            
        Raises:
            Exception: If description generation fails after max retries
        """
        last_error = None
        
        # Set up MLflow experiment
        mlflow.set_experiment("cluster_descriptions")
        
        with mlflow.start_run(run_name=f"describe_cluster_{len(skills)}_skills"):
            mlflow.log_param("model", self.model_gen)
            mlflow.log_param("num_skills", len(skills))
            mlflow.log_param("max_retries", self.gen_max_retries)
            
            for attempt in range(self.gen_max_retries):
                try:
                    skills_list = ', '.join(skills)
                    user_prompt = USER_PROMPT_EXTRACT_CLUSTER_DESCRIPTION.format(text=skills_list)
                    
                    logger.info(f"Generating cluster description (attempt {attempt + 1}/{self.gen_max_retries})")
                    logger.info(f"Skills: {skills_list[:100]}...")
                    
                    temperature = self.get_temperature(attempt)
                    mlflow.log_param(f"attempt_{attempt + 1}_temperature", temperature)
                    mlflow.log_text(user_prompt, f"attempt_{attempt + 1}_prompt.txt")
                    
                    response = requests.post(
                        f"{self.ollama_base_url}/api/generate",
                        json={
                            "model": self.model_gen,
                            "prompt": user_prompt,
                            "stream": False,
                            "options": {
                                "temperature": temperature,
                                "num_predict": 10000
                            }
                        },
                        timeout=self.request_timeout
                    )
                    response.raise_for_status()
                    response_data = response.json()
                    raw_response = response_data.get("response", "").strip()
                    
                    logger.info(f"Raw LLM response (attempt {attempt + 1}): '{raw_response}'")
                    mlflow.log_text(raw_response, f"attempt_{attempt + 1}_raw_response.txt")
                    
                    # Extract the description
                    description = re.sub(r'theme:?', '', raw_response, flags=re.IGNORECASE)
                    description = re.sub(r'category:?', '', description, flags=re.IGNORECASE)
                    description = description.rstrip('.').strip()
                    description = description.split('.')[0].split('\n')[0].strip()
                    description = re.sub(r'^\*+\s*|\s*\*+$', '', description)
                    description = re.sub(r'\*\*([^*]+)\*\*', r'\1', description)
                    
                    # Capitalize first letter
                    if description:
                        description = description[0].upper() + description[1:]
                    
                    # Validate
                    if description and len(description) > 0:
                        # Check if response is just listing all skills (too long or contains many commas)
                        if len(description) > 60 or description.count(',') > 2 or not any(char.isalpha() for char in description):
                            logger.warning(f"Response too long or lists skills or no letters, retrying...")
                            mlflow.log_metric(f"attempt_{attempt + 1}_validation_failed", 1)
                            continue
                        
                        # Valid description found
                        logger.info(f"Generated cluster description: '{description}'")
                        mlflow.log_text(description, "final_description.txt")
                        mlflow.log_metric("successful_attempt", attempt + 1)
                        mlflow.log_metric("success", 1)
                        return description
                    
                    # Empty description - retry
                    logger.warning(f"Empty description on attempt {attempt + 1}, retrying...")
                    mlflow.log_metric(f"attempt_{attempt + 1}_empty", 1)
                    continue
                    
                except Exception as e:
                    logger.error(f"Error during cluster description generation: {e}")
                    mlflow.log_param(f"attempt_{attempt + 1}_error", str(e))
                    if attempt < self.gen_max_retries - 1:
                        continue
            
            # All attempts failed - use fallback
            fallback_description = f"{', '.join(skills[:3])}"
            logger.warning(f"All {self.gen_max_retries} attempts failed. Using fallback description: {fallback_description}")
            mlflow.log_text(fallback_description, "fallback_description.txt")
            mlflow.log_metric("success", 0)
            mlflow.log_metric("used_fallback", 1)
            return fallback_description
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            Exception: If embedding generation fails
        """
        try:
            embeddings = []
            for text in texts:
                response = requests.post(
                    f"{self.ollama_base_url}/api/embed",
                    json={
                        "model": self.model_emb,
                        "input": text
                    },
                    timeout=self.request_timeout
                )
                response.raise_for_status()
                embedding = response.json()["embeddings"][0]
                
                # Check if vector is null/zero
                is_null_vector = all(val == 0.0 for val in embedding)
                if is_null_vector:
                    logger.warning(f"Null/zero vector returned for: {text[:100]}...")
                else:
                    logger.info(f"Valid embedding vector returned (dim: {len(embedding)})")
                
                # Ensure correct dimension
                if len(embedding) != self.embedding_dim:
                    logger.warning(f"Expected {self.embedding_dim} dimensions, got {len(embedding)}")
                    if len(embedding) < self.embedding_dim:
                        embedding = embedding + [0.0] * (self.embedding_dim - len(embedding))
                    else:
                        embedding = embedding[:self.embedding_dim]
                
                embeddings.append(embedding)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error during embedding: {e}")
            raise Exception(f"Embedding generation failed: {str(e)}")
