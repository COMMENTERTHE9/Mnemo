#!/bin/bash

# Deployment script for DigitalOcean 2GB server using Podman
# Optimized for 2GB RAM constraint with improved resource management

set -e

echo "🚀 Starting Mnemo deployment on DigitalOcean server with Podman..."

# Check if we're on the server
if [ ! -f "/root/Mnemo/.git/config" ]; then
    echo "❌ This script should be run on the DigitalOcean server in /root/Mnemo/"
    exit 1
fi

# Check if podman is installed
if ! command -v podman &> /dev/null; then
    echo "❌ Podman not found. Installing..."
    apt update && apt install -y podman podman-compose
fi

echo "📥 Pulling latest changes from GitHub..."
git pull origin master

echo "🔧 Setting up environment configuration..."
cp .env.server .env

echo "🛑 Stopping existing containers gracefully..."
podman-compose -f docker-compose.server.yml down --remove-orphans || true

echo "🏗️ Building updated images..."
podman-compose -f docker-compose.server.yml build --no-cache

echo "🚀 Starting optimized services..."
podman-compose -f docker-compose.server.yml --env-file .env.server up -d

echo "⏳ Waiting for services to be ready..."
sleep 10

echo "📊 Checking service status..."
podman-compose -f docker-compose.server.yml ps

echo "💾 Checking memory usage..."
echo "Total system memory usage:"
free -h

echo "Container resource usage:"
podman stats --no-stream

echo "✅ Deployment complete!"
echo ""
echo "🔍 To monitor:"
echo "  podman-compose -f docker-compose.server.yml logs -f"
echo ""
echo "🔧 To manage:"
echo "  podman-compose -f docker-compose.server.yml ps    # Status"
echo "  podman-compose -f docker-compose.server.yml down  # Stop"
echo "  podman-compose -f docker-compose.server.yml up -d # Start"
echo ""
echo "📊 Resource monitoring:"
echo "  podman stats    # Real-time container stats"
echo "  free -h         # System memory"
echo "  df -h           # Disk usage"