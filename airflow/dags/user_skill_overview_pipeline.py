"""
User overview pipeline DAG.
Generates user skill analysis: fetch → UMAP → cluster → describe → persist.
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
    'retry_delay': timedelta(minutes=5),
}

@dag(
    dag_id='user_skill_overview_pipeline',
    default_args=default_args,
    description='User skill overview pipeline',
    schedule=None,  # Triggered manually via API
    catchup=False,
    tags=['user-skill-overview'],
)
def user_skill_overview_pipeline():
    """User skill overview analysis pipeline."""
    
    @task
    def fetch_task():
        """Fetch user skills and prepare data."""
        from pipelines.user_overview.tasks import fetch_user_skills
        context = get_current_context()
        conf = context.get('dag_run').conf if context.get('dag_run') else None
        if not conf or 'user_id' not in conf:
            raise ValueError("user_id not provided in DAG run configuration")
        user_id = conf['user_id']
        return fetch_user_skills(user_id)
    
    @task
    def umap_task(step_1_data):
        """Reduce dimensions using UMAP."""
        from pipelines.user_overview.tasks import reduce_dimensions
        return reduce_dimensions(step_1_data)
    
    @task
    def cluster_task(step_2_data):
        """Cluster skills using HDBSCAN."""
        from pipelines.user_overview.tasks import cluster_skills
        return cluster_skills(step_2_data)
    
    @task
    def describe_task(step_3_data):
        """Generate cluster descriptions using LLM."""
        from pipelines.user_overview.tasks import describe_clusters
        return describe_clusters(step_3_data)
    
    @task
    def persist_task(step_4_data):
        """Persist user overview to database."""
        from pipelines.user_overview.tasks import persist_overview
        return persist_overview(step_4_data)
    
    # Define task dependencies using TaskFlow API
    fetched_data = fetch_task()
    reduced_data = umap_task(fetched_data)
    clustered_data = cluster_task(reduced_data)
    described_data = describe_task(clustered_data)
    persist_task(described_data)

# Instantiate the DAG
dag_instance = user_skill_overview_pipeline()