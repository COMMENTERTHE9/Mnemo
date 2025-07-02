#!/bin/bash
# Deploy script for Mnemo on DigitalOcean server

echo "Deploying Mnemo services..."

# Stop existing containers
echo "Stopping old containers..."
docker-compose -f docker-compose.server.yml down

# Pull latest code
echo "Pulling latest code..."
git pull

# Build images
echo "Building orchestrator..."
docker build -t mnemo_orchestrator:latest -f orchestrator/Dockerfile .

echo "Building video worker..."
docker build -t mnemo_video_worker:latest -f worker/Dockerfile .

echo "Building motion extractor..."
docker build -t mnemo_motion_extractor:latest -f motion-extractor/Dockerfile .

# Start services using the server config
echo "Starting services..."
docker-compose -f docker-compose.server.yml up -d

# Show status
echo "Services status:"
docker ps

echo "Deployment complete!"
echo "Check logs with: docker logs -f mnemo_orchestrator"