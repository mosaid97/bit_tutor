# utilities/data_processing/__init__.py

"""
Data Processing Utilities

This module provides utilities for processing, validating, and transforming data
across the BIT Tutor system.

Main Components:
- StudentDataProcessor: Processes and validates student data and interactions
"""

from .student_data_processor import StudentDataProcessor

# Re-export main classes
__all__ = [
    'StudentDataProcessor'
]

# Version information
__version__ = '1.0.0'
__author__ = 'BIT Tutor Development Team'