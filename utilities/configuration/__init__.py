# utilities/configuration/__init__.py

"""
Configuration Utilities

This module provides utilities for managing configuration settings
across the BIT Tutor system.

Main Components:
- ConfigManager: Manages application configuration settings
"""

from .config_manager import ConfigManager

# Re-export main classes
__all__ = [
    'ConfigManager'
]

# Version information
__version__ = '1.0.0'
__author__ = 'BIT Tutor Development Team'