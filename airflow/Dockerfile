FROM apache/airflow:2.7.1

# Copy requirements file
COPY requirements-airflow.txt /requirements-airflow.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /requirements-airflow.txt
