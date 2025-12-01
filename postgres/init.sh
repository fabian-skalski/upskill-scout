#!/bin/bash

DATABASES=("$POSTGRES_DB_AIRFLOW" "$POSTGRES_DB_MLFLOW")

for DB_NAME in "${DATABASES[@]}"; do
  echo "Checking if database '$DB_NAME' exists..."
  # Check if database exists (trim whitespace using `xargs` to prevent mismatches)
  DB_EXISTS=$(psql -U "$POSTGRES_USER" -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname = '$DB_NAME'" | xargs)
  
  if [ "$DB_EXISTS" == "1" ]; then
    echo "Database '$DB_NAME' already exists, skipping creation."
  else
    echo "Database '$DB_NAME' not found. Creating..."
    psql -U "$POSTGRES_USER" -d postgres -c "CREATE DATABASE \"$DB_NAME\""
    echo "Database '$DB_NAME' created successfully."
  fi
done