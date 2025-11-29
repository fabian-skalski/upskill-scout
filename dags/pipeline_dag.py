from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

# Add airflow_tasks to path so we can import workers
sys.path.append(os.path.join(os.environ.get("AIRFLOW_HOME", "/opt/airflow"), "airflow_tasks"))

# DO NOT import workers here - it's too slow and causes DAG import timeout
# Import inside functions instead

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG(
    'data_processing_pipeline',
    default_args=default_args,
    description='A data processing pipeline for Upskill Scout',
    schedule_interval=None,  # Triggered manually via API
    catchup=False,
)

def cleanse_wrapper(**kwargs):
    from workers import process_step_1_cleanse
    conf = kwargs.get('dag_run').conf
    if not conf:
        raise ValueError("No configuration provided to DAG run")
    return process_step_1_cleanse(conf)

def extract_wrapper(**kwargs):
    from workers import process_step_2_extract
    ti = kwargs['ti']
    job_data = ti.xcom_pull(task_ids='cleanse_task')
    return process_step_2_extract(job_data)

def embed_wrapper(**kwargs):
    from workers import process_step_3_embed
    ti = kwargs['ti']
    job_data = ti.xcom_pull(task_ids='extract_task')
    return process_step_3_embed(job_data)

def persist_wrapper(**kwargs):
    from workers import process_step_4_persist
    ti = kwargs['ti']
    job_data = ti.xcom_pull(task_ids='embed_task')
    return process_step_4_persist(job_data)

cleanse_task = PythonOperator(
    task_id='cleanse_task',
    python_callable=cleanse_wrapper,
    provide_context=True,
    dag=dag,
)

extract_task = PythonOperator(
    task_id='extract_task',
    python_callable=extract_wrapper,
    provide_context=True,
    dag=dag,
)

embed_task = PythonOperator(
    task_id='embed_task',
    python_callable=embed_wrapper,
    provide_context=True,
    dag=dag,
)

persist_task = PythonOperator(
    task_id='persist_task',
    python_callable=persist_wrapper,
    provide_context=True,
    dag=dag,
)

cleanse_task >> extract_task >> embed_task >> persist_task
