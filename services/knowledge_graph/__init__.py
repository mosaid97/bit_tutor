# services/knowledge_graph/__init__.py

"""
Knowledge Graph Service

This service manages personalized student knowledge graphs and cognitive foundations.
It provides the core memory system for the Educational Agent, serving as the single
source of truth for all student data and learning activities.

Main Components:
- StudentKnowledgeGraph: Manages individual student knowledge graphs
- build_cognitive_foundation: Creates the foundational knowledge structure
- run_educational_agent: Main educational agent orchestration function
"""

from .models.student_knowledge_graph import StudentKnowledgeGraph
from .models.cognitive_foundation import build_cognitive_foundation, run_educational_agent

# Re-export main classes and functions for backward compatibility
__all__ = [
    'StudentKnowledgeGraph',
    'build_cognitive_foundation',
    'run_educational_agent'
]

# Version information
__version__ = '1.0.0'
__author__ = 'BIT Tutor Development Team'