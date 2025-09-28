#!/bin/bash

# BIT Tutor - Docker Entrypoint Script
# This script initializes the application environment and starts the Flask server

set -e

echo "=== BIT Tutor - Starting Application ==="
echo "Timestamp: $(date)"
echo "Python Version: $(python --version)"
echo "Working Directory: $(pwd)"

# Create necessary directories if they don't exist
mkdir -p data logs static/uploads

# Set proper permissions
chmod -R 755 static/
chmod -R 755 data/
chmod -R 755 logs/

# Initialize data directory with sample data if empty
if [ ! -f "data/initialized" ]; then
    echo "Initializing application data..."
    python -c "
from services.knowledge_graph import build_cognitive_foundation
try:
    build_cognitive_foundation()
    print('✓ Cognitive foundation initialized')
except Exception as e:
    print(f'Warning: Could not initialize cognitive foundation: {e}')

with open('data/initialized', 'w') as f:
    f.write('initialized')
print('✓ Data initialization complete')
    "
fi

# Check database connectivity (if using external database)
if [ "$DATABASE_URL" ]; then
    echo "Checking database connectivity..."
    # Add database connectivity check here if needed
fi

# Run database migrations or initialization scripts
if [ -f "init_db.py" ]; then
    echo "Running database initialization..."
    python init_db.py
fi

# Start the Flask application
echo "Starting BIT Tutor Flask application..."
echo "Listening on port: ${PORT:-5000}"
echo "Environment: ${FLASK_ENV:-production}"
echo "========================================"

# Use gunicorn for production or flask dev server for development
if [ "$FLASK_ENV" = "development" ]; then
    exec python nexus_app.py
else
    # Install gunicorn if not already installed
    pip install gunicorn 2>/dev/null || true
    
    # Use gunicorn with optimal settings for production
    exec gunicorn \
        --bind 0.0.0.0:${PORT:-5000} \
        --workers ${WORKERS:-4} \
        --worker-class sync \
        --worker-connections 1000 \
        --max-requests 1000 \
        --max-requests-jitter 100 \
        --timeout 30 \
        --keep-alive 5 \
        --log-level info \
        --access-logfile - \
        --error-logfile - \
        nexus_app:app
fi
