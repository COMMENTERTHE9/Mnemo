#!/bin/bash

# Test local setup with optimized memory configuration
set -e

echo "🧪 Testing Mnemo Local Setup"
echo "============================"

echo "📋 System memory:"
free -h

echo ""
echo "🔧 Stopping any existing services..."
docker-compose down

echo ""
echo "🗃️  Database status:"
ls -lh data/video_memory.db
echo "Tables in database:"
sqlite3 data/video_memory.db ".tables"

echo ""
echo "📦 Building essential services only (orchestrator + video-worker)..."
docker-compose up -d orchestrator video-worker

echo ""
echo "⏳ Waiting for services to start..."
sleep 15

echo ""
echo "📊 Service status:"
docker-compose ps

echo ""
echo "🌐 Testing orchestrator API..."
curl -f http://localhost:8080/health || echo "❌ Health check failed"

echo ""
echo "💾 Memory usage:"
docker stats --no-stream

echo ""
echo "📝 Recent logs:"
docker-compose logs --tail=10

echo ""
echo "✅ Local test complete!"
echo "🔧 To stop: docker-compose down"
echo "🚀 To start all: docker-compose up -d"