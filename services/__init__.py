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
# These imports are lazy to avoid loading heavy dependencies when only accessing submodules
def __getattr__(name):
    """Lazy import to avoid loading torch and other heavy dependencies unnecessarily."""
    if name == 'StudentKnowledgeGraph' or name == 'build_cognitive_foundation' or name == 'run_educational_agent':
        from .knowledge_graph import StudentKnowledgeGraph, build_cognitive_foundation, run_educational_agent
        return locals()[name]
    elif name in ['LLM_Skill_Extractor', 'ASTNN', 'TextEmbedder', 'SQKT_Model', 'SQKT_KnowledgeTracer', 'MLFBK_Model', 'MLFBK_KnowledgeTracer']:
        from .knowledge_tracing import LLM_Skill_Extractor, ASTNN, TextEmbedder, SQKT_Model, SQKT_KnowledgeTracer, MLFBK_Model, MLFBK_KnowledgeTracer
        return locals()[name]
    elif name in ['LLM_Cold_Start_Assessor', 'GNN_CDM', 'ExplainableAIEngine', 'convert_nx_to_pyg']:
        from .cognitive_diagnosis import LLM_Cold_Start_Assessor, GNN_CDM, ExplainableAIEngine, convert_nx_to_pyg
        return locals()[name]
    elif name in ['LLM_Content_Generator', 'RL_Recommender_Agent', 'RecommendationService']:
        from .recommendation import LLM_Content_Generator, RL_Recommender_Agent, RecommendationService
        return locals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Re-export main classes for easy access
__all__ = [
    # Knowledge Graph Service
    'StudentKnowledgeGraph',
    'build_cognitive_foundation',
    'run_educational_agent',

    # Knowledge Tracing Service (SQKT)
    'LLM_Skill_Extractor',
    'ASTNN',
    'TextEmbedder',
    'SQKT_Model',
    'SQKT_KnowledgeTracer',
    'MLFBK_Model',  # Backward compatibility
    'MLFBK_KnowledgeTracer',  # Backward compatibility

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