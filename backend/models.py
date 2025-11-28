from pydantic import BaseModel
from typing import List, Optional
from enum import Enum

class PipelineStep(str, Enum):
    RECEIVED = "received"
    CLEANSED = "cleansed"
    EXTRACTED = "extracted"
    EMBEDDED = "embedded"
    PERSISTED = "persisted"

class TextSubmission(BaseModel):
    description: str
    sourceUrl: str
    timestamp: str

class SubmissionResponse(BaseModel):
    message: str
    text_hash: str

class JobSkill(BaseModel):
    name: str
    embedding: Optional[List[float]] = None

class ProcessedJob(BaseModel):
    original_text: str  # This will be the description
    source_url: str
    timestamp: str
    text_hash: str
    clean_text: Optional[str] = None
    title: Optional[str] = None
    skills: List[JobSkill] = []
    text_embedding: Optional[List[float]] = None
    step: PipelineStep = PipelineStep.RECEIVED

