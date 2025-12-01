"""
Shared models module.
This file can be imported by both backend and airflow to eliminate duplication.
"""
from app.core.models import PipelineStep, JobSkill, ProcessedJob

__all__ = ['PipelineStep', 'JobSkill', 'ProcessedJob']