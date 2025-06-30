#!/bin/bash

echo "Starting Video Memory Framework (Local Development Mode)"
echo "======================================================="

# Check if data directory exists
if [ ! -d "./data" ]; then
    echo "Creating data directory..."
    mkdir -p ./data
fi

# Initialize database if it doesn't exist
if [ ! -f "./data/video_memory.db" ]; then
    echo "Initializing SQLite database..."
    ./database/init_db.sh ./data/video_memory.db
    if [ $? -ne 0 ]; then
        echo "Failed to initialize database!"
        exit 1
    fi
else
    echo "Database already exists at ./data/video_memory.db"
fi

# Copy local env file
if [ ! -f ".env" ]; then
    echo "Creating .env from .env.local..."
    cp .env.local .env
fi

# Build and start services
echo -e "\nStarting services with docker-compose..."
docker-compose -f docker-compose.local.yml up --build

# Optional: Add monitoring
# docker-compose -f docker-compose.local.yml --profile monitoring up --build