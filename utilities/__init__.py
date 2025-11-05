# utilities/__init__.py

"""
KTCD_Aug Utilities Package

This package contains utility scripts and tools for:
- Knowledge graph management (cleanup, visualization)
- Benchmarking and testing AI models
- Data migration and setup
- Configuration management
- Deployment tools

Available Scripts:
- benchmark_algorithms.py - Compare AI models with state-of-the-art
- cleanup_knowledge_graph.py - Clean duplicate/unused nodes
- visualize_knowledge_graph.py - Visualize graph structure
- create_demo_students.py - Create demo student data
- setup_demo_system.py - Initialize demo environment
- verify_and_test_pipelines.py - Test complete pipelines
- migrate_passwords_to_bcrypt.py - Migrate to secure passwords
- generate_all_quizzes.py - Generate quiz questions
- load_concepts_to_neo4j.py - Load concepts to database
- aggregate_lab_tutor_data.py - Aggregate lab tutor data
- add_sample_grades_roma.py - Add sample grades
- visualize_students_in_graph.py - Visualize student data

Subpackages:
- configuration: Configuration management utilities
- data_processing: Student data processing and validation utilities
- visualization: Data visualization and reporting utilities
- testing: Testing utilities and helpers
- deployment: Deployment and infrastructure utilities
"""

from .data_processing import StudentDataProcessor
from .configuration import ConfigManager

# Re-export main classes
__all__ = [
    'StudentDataProcessor',
    'ConfigManager'
]

# Version information
__version__ = '4.0.0'
__author__ = 'KTCD_Aug Development Team'