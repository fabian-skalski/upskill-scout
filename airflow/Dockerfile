FROM apache/airflow:3.1.3-python3.12

ARG AIRFLOW_USER=${AIRFLOW_USER}


USER root
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Copy requirements file
COPY airflow/requirements-airflow.txt /requirements-airflow.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /requirements-airflow.txt
