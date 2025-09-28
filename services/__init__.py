# services/__init__.py

"""
BIT Tutor Services

This package contains all the core services for the BIT Tutor educational AI platform.
Each service is designed as a modular component that can be used independently or
as part of the integrated system.

Main Services:
- knowledge_graph: Student knowledge graph management and cognitive foundations
- knowledge_tracing: SQKT and MLFBK-based knowledge tracing algorithms
- cognitive_diagnosis: GNN-CDM and cold-start assessment services
- recommendation: RL-based recommendation engine and content generation
- educational_agent: Main AI orchestrator and decision-making engine
- student_data: Student profile and data management services
- content_generation: Dynamic learning content generation services
- analytics: Learning analytics and progress tracking services
- ai_chat: Conversational AI and chatbot services
"""

# Import main classes from each service
from .knowledge_graph import StudentKnowledgeGraph, build_cognitive_foundation, run_educational_agent
from .knowledge_tracing import LLM_Skill_Extractor, ASTNN, TextEmbedder, MLFBK_Model, MLFBK_KnowledgeTracer
from .cognitive_diagnosis import LLM_Cold_Start_Assessor, GNN_CDM, ExplainableAIEngine, convert_nx_to_pyg
from .recommendation import LLM_Content_Generator, RL_Recommender_Agent, RecommendationService

# Re-export main classes for easy access
__all__ = [
    # Knowledge Graph Service
    'StudentKnowledgeGraph',
    'build_cognitive_foundation',
    'run_educational_agent',

    # Knowledge Tracing Service
    'LLM_Skill_Extractor',
    'ASTNN',
    'TextEmbedder',
    'MLFBK_Model',
    'MLFBK_KnowledgeTracer',

    # Cognitive Diagnosis Service
    'LLM_Cold_Start_Assessor',
    'GNN_CDM',
    'ExplainableAIEngine',
    'convert_nx_to_pyg',

    # Recommendation Service
    'LLM_Content_Generator',
    'RL_Recommender_Agent',
    'RecommendationService'
]

# Version information
__version__ = '1.0.0'
__author__ = 'BIT Tutor Development Team'