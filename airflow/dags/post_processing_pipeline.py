"""
Data processing pipeline DAG.
Processes postings through: cleanse → extract → embed → persist.
"""
from airflow.decorators import dag, task
from airflow.sdk import get_current_context
from datetime import datetime, timedelta
import sys
import os

# Add pipelines to path
sys.path.append("/opt/airflow")

default_args = {
    'owner': os.environ.get('AIRFLOW__DAGS__OWNER'),
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1), # start date in the past to allow manual triggering
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

@dag(
    dag_id='post_processing_pipeline',
    default_args=default_args,
    description='Posting processing pipeline',
    schedule=None,  # Triggered manually via API
    catchup=False,
    tags=['job-processing'],
)
def post_processing_pipeline():
    """Posting processing pipeline."""
    
    @task
    def cleanse_task():
        """Cleanse posting text."""
        from pipelines.job_processing.tasks import cleanse_job_text
        context = get_current_context()
        conf = context.get('dag_run').conf if context.get('dag_run') else None
        if not conf:
            raise ValueError("No configuration provided to DAG run")
        return cleanse_job_text(conf)
    
    @task
    def extract_task(job_data):
        """Extract job information from cleansed text."""
        from pipelines.job_processing.tasks import extract_job_info
        return extract_job_info(job_data)
    
    @task
    def embed_task(job_data):
        """Generate embeddings for job data."""
        from pipelines.job_processing.tasks import embed_job_data
        return embed_job_data(job_data)
    
    @task
    def persist_task(job_data):
        """Persist job data to database."""
        from pipelines.job_processing.tasks import persist_job_data
        return persist_job_data(job_data)
    
    # Define task dependencies using TaskFlow API
    cleansed_data = cleanse_task()
    extracted_data = extract_task(cleansed_data)
    embedded_data = embed_task(extracted_data)
    persist_task(embedded_data)

# Instantiate the DAG
dag_instance = post_processing_pipeline()