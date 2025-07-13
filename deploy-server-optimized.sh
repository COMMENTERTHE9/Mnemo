#!/bin/bash

# Deploy optimized Mnemo services on 2GB DigitalOcean server
# Total memory allocation: ~1.7GB (400M + 600M + 700M)

set -e

echo "🚀 Deploying Mnemo Video Memory System (2GB Optimized)"
echo "=================================================="

# Check if we're on the server
if [ ! -d "~/Mnemo" ]; then
    echo "❌ Mnemo directory not found. Make sure you're on the DigitalOcean server."
    echo "   Expected path: ~/Mnemo"
    exit 1
fi

# Navigate to project directory
cd ~/Mnemo || exit 1

echo "📋 Current system memory:"
free -h

echo ""
echo "🔧 Stopping any existing services..."
docker-compose -f docker-compose.server.yml down

echo ""
echo "🏗️  Building images (this may take a few minutes)..."
docker build -t mnemo_orchestrator:latest -f orchestrator/Dockerfile .
docker build -t mnemo_video_worker:latest -f worker/Dockerfile .
docker build -t mnemo_motion_extractor:latest -f motion-extractor/Dockerfile .

echo ""
echo "🧹 Cleaning up unused images to save space..."
docker image prune -f

echo ""
echo "📊 Docker disk usage:"
docker system df

echo ""
echo "🚀 Starting optimized services..."
docker-compose -f docker-compose.server.yml up -d

echo ""
echo "⏳ Waiting for services to start..."
sleep 10

echo ""
echo "📊 Service status:"
docker-compose -f docker-compose.server.yml ps

echo ""
echo "💾 Memory usage after startup:"
free -h

echo ""
echo "🔍 Container resource usage:"
docker stats --no-stream

echo ""
echo "✅ Deployment complete!"
echo ""
echo "🌐 Orchestrator API: http://$(curl -s ifconfig.me):80"
echo "📊 Check logs: docker-compose -f docker-compose.server.yml logs -f"
echo "📈 Monitor: docker stats"
echo ""
echo "🎯 Ready to process queued video: video_1751340188716362897"