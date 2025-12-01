#!/bin/sh
set -e

# Start MinIO server in the background
minio server /minio_data --console-address ":${MINIO_CONSOLE_PORT}" &

# Wait for MinIO to be ready
echo "Waiting for MinIO to start..."
until curl -sf http://localhost:${MINIO_API_PORT}/minio/health/live > /dev/null 2>&1; do
    sleep 1
done

echo "MinIO is ready. Initializing buckets..."

# Install mc (MinIO Client) if not present
if ! command -v mc > /dev/null 2>&1; then
    echo "Installing MinIO client..."
    # Detect architecture
    ARCH=$(uname -m)
    if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
        MC_URL="https://dl.min.io/client/mc/release/linux-arm64/mc"
    else
        MC_URL="https://dl.min.io/client/mc/release/linux-amd64/mc"
    fi
    curl -sSL "$MC_URL" -o /usr/local/bin/mc
    chmod +x /usr/local/bin/mc
fi

# Configure MinIO client
mc alias set myminio http://localhost:${MINIO_API_PORT} ${MINIO_ACCESS_KEY} ${MINIO_SECRET_KEY}

# Create MLflow bucket if it doesn't exist
if ! mc ls myminio/${MLFLOW_BUCKET_NAME} > /dev/null 2>&1; then
    echo "Creating bucket: ${MLFLOW_BUCKET_NAME}"
    mc mb myminio/${MLFLOW_BUCKET_NAME}
    echo "Bucket ${MLFLOW_BUCKET_NAME} created successfully"
else
    echo "Bucket ${MLFLOW_BUCKET_NAME} already exists"
fi

# Set bucket policy to allow read/write (adjust as needed for production)
mc anonymous set download myminio/${MLFLOW_BUCKET_NAME}

echo "MinIO initialization complete"

# Keep the script running and wait for the MinIO server process
wait
