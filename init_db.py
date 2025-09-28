#!/usr/bin/env python3
"""
BIT Tutor - Database Initialization Script
This script initializes the database schema and sample data for the application.
"""

import os
import sys
import logging
from datetime import datetime

# Add the app directory to Python path
sys.path.insert(0, '/app')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def init_database():
    """Initialize the database with required tables and sample data."""
    logger.info("Starting database initialization...")
    
    try:
        # Import after adding to path
        from services.knowledge_graph import build_cognitive_foundation
        from services.student_data import StudentDataService
        
        logger.info("Building cognitive foundation...")
        foundational_kg, qm, kcs = build_cognitive_foundation()
        logger.info(f"✓ Initialized {len(kcs)} knowledge components")
        
        logger.info("Initializing student data service...")
        student_service = StudentDataService()
        student_service.initialize_students(force_recreate=True)
        logger.info(f"✓ Initialized {len(student_service.student_ids)} students")
        
        # Create marker file to indicate initialization is complete
        with open('/app/data/db_initialized', 'w') as f:
            f.write(f"Database initialized at {datetime.now().isoformat()}\n")
        
        logger.info("✓ Database initialization completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        return False

def check_database():
    """Check if database is already initialized."""
    return os.path.exists('/app/data/db_initialized')

if __name__ == "__main__":
    if check_database():
        logger.info("Database already initialized, skipping...")
        sys.exit(0)
    
    success = init_database()
    sys.exit(0 if success else 1)