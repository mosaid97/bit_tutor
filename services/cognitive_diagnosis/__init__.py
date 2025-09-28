# services/cognitive_diagnosis/__init__.py

"""
Cognitive Diagnosis Service

This service implements advanced cognitive diagnosis algorithms including GNN-based
Cognitive Diagnosis Models (GNN-CDM), cold-start assessment, and explainable AI components.

Main Components:
- LLM_Cold_Start_Assessor: Provides initial diagnosis for new students
- GNN_CDM: Graph Neural Network-based Cognitive Diagnosis Model
- ExplainableAIEngine: Generates human-readable explanations for AI decisions
- convert_nx_to_pyg: Utility for converting NetworkX graphs to PyTorch Geometric format
"""

from .models.cold_start_assessor import LLM_Cold_Start_Assessor
from .models.gnn_cdm import GNN_CDM, convert_nx_to_pyg
from .models.explainable_ai_engine import ExplainableAIEngine

# Re-export main classes for backward compatibility
__all__ = [
    'LLM_Cold_Start_Assessor',
    'GNN_CDM',
    'ExplainableAIEngine',
    'convert_nx_to_pyg'
]

# Version information
__version__ = '1.0.0'
__author__ = 'BIT Tutor Development Team'