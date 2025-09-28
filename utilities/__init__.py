# utilities/__init__.py

"""
BIT Tutor Utilities

This package provides utility modules for data processing, configuration management,
visualization, testing, and deployment support across the BIT Tutor system.

Main Modules:
- data_processing: Student data processing and validation utilities
- configuration: Configuration management utilities
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
__version__ = '1.0.0'
__author__ = 'BIT Tutor Development Team'