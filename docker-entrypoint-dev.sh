#!/bin/bash

# BIT Tutor - Development Docker Entrypoint Script
# This script is optimized for development with hot reloading

set -e

echo "=== BIT Tutor - Development Mode ==="
echo "Timestamp: $(date)"

# Create necessary directories
mkdir -p data logs static/uploads

# Install development dependencies
pip install --no-cache-dir flask-cors python-dotenv

# Set development environment variables
export FLASK_ENV=development
export FLASK_DEBUG=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

# Initialize sample data for development
if [ ! -f "data/dev_initialized" ]; then
    echo "Setting up development data..."
    python -c "
try:
    from services.knowledge_graph import build_cognitive_foundation
    build_cognitive_foundation()
    print('✓ Development data initialized')
except Exception as e:
    print(f'Warning: {e}')
    
with open('data/dev_initialized', 'w') as f:
    f.write('dev_initialized')
    "
fi

echo "Starting development server with hot reload..."
echo "Access the application at: http://localhost:5000"
echo "======================================"

# Start Flask development server with hot reloading
exec python nexus_app.py --debug
