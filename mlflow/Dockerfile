FROM ghcr.io/mlflow/mlflow:v3.7.0rc0

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python package
COPY requirements-mlflow.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-mlflow.txt