# services/cognitive_diagnosis/__init__.py

"""
Cognitive Diagnosis Service

This service implements advanced cognitive diagnosis algorithms including GNN-based
Cognitive Diagnosis Models (GNN-CDM), cold-start assessment, explainable AI components,
and AD4CD (Anomaly Detection for Cognitive Diagnosis).

Main Components:
- LLM_Cold_Start_Assessor: Provides initial diagnosis for new students
- GNN_CDM: Graph Neural Network-based Cognitive Diagnosis Model
- ExplainableAIEngine: Generates human-readable explanations for AI decisions
- AD4CD_Model: Anomaly detection for cognitive diagnosis
- AD4CD_CognitiveDiagnosis: AD4CD service for enhanced diagnosis
- AD4CDIntegrationService: Integration with Neo4j and SQKT
- convert_nx_to_pyg: Utility for converting NetworkX graphs to PyTorch Geometric format
"""

from .models.cold_start_assessor import LLM_Cold_Start_Assessor
from .models.explainable_ai_engine import ExplainableAIEngine

# Try to import torch-dependent models
try:
    from .models.gnn_cdm import GNN_CDM, convert_nx_to_pyg
    from .models.ad4cd_model import AD4CD_Model, AD4CD_CognitiveDiagnosis
    from .services.ad4cd_service import AD4CDIntegrationService
    _TORCH_AVAILABLE = True
    print("✅ Cognitive Diagnosis module loaded (with AD4CD)")
except ImportError as e:
    print(f"Warning: torch-dependent models not available: {e}")
    print("Creating mock implementations...")
    _TORCH_AVAILABLE = False

    # Create mock classes
    class GNN_CDM:
        def __init__(self, *args, **kwargs):
            raise ImportError("GNN_CDM requires torch and torch_geometric. Please install: pip install torch torch-geometric")

    def convert_nx_to_pyg(*args, **kwargs):
        raise ImportError("convert_nx_to_pyg requires torch_geometric. Please install: pip install torch torch-geometric")

    class AD4CD_Model:
        def __init__(self, *args, **kwargs):
            raise ImportError("AD4CD_Model requires torch. Please install: pip install torch")

    class AD4CD_CognitiveDiagnosis:
        def __init__(self, *args, **kwargs):
            raise ImportError("AD4CD_CognitiveDiagnosis requires torch. Please install: pip install torch")

    class AD4CDIntegrationService:
        def __init__(self, *args, **kwargs):
            raise ImportError("AD4CDIntegrationService requires torch. Please install: pip install torch")

# Re-export main classes for backward compatibility
__all__ = [
    'LLM_Cold_Start_Assessor',
    'GNN_CDM',
    'ExplainableAIEngine',
    'convert_nx_to_pyg',
    'AD4CD_Model',
    'AD4CD_CognitiveDiagnosis',
    'AD4CDIntegrationService'
]

# Version information
__version__ = '2.0.0'
__author__ = 'BIT Tutor Development Team'