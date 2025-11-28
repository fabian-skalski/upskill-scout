#!/bin/bash
# init_ollama_models.sh
# This script pulls the required models for the Ollama service

set -e

echo "=========================================="
echo "Initializing Ollama Models"
echo "=========================================="

# Default port if not set
OLLAMA_PORT=${OLLAMA_PORT}

# Check if Ollama port is already in use
echo "Checking if Ollama is already running on port ${OLLAMA_PORT}..."
if nc -z localhost ${OLLAMA_PORT} 2>/dev/null; then
    echo "✓ Ollama is already running on port ${OLLAMA_PORT}"
else
    echo "Port ${OLLAMA_PORT} is free. Starting Ollama server..."
    ollama serve &
    echo "Ollama server started in the background."
    sleep 2
fi

# Pull generation model
echo ""
echo "Pulling generation model: ${MODEL_GEN}..."
ollama pull "${MODEL_GEN}"

# Pull embedding model
echo ""
echo "Pulling embedding model: ${MODEL_EMB}..."
ollama pull "${MODEL_EMB}"

echo ""
echo "=========================================="
echo "✓ Models initialized successfully"
echo "=========================================="
echo ""
echo "Available models:"
ollama list
