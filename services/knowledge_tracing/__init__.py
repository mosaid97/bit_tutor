# services/knowledge_tracing/__init__.py

"""
Knowledge Tracing Service

This service implements advanced knowledge tracing algorithms including SQKT (Students' Question-based
Knowledge Tracing) with MLFBK (Multi-Features with Latent Relations BERT Knowledge Tracing) models.

Main Components:
- LLM_Skill_Extractor: Extracts skills from student questions using LLM techniques
- ASTNN: Converts Abstract Syntax Trees to vector embeddings
- TextEmbedder: Embeds natural language text (student questions)
- MLFBK_Model: Advanced transformer-based knowledge tracing model
- MLFBK_KnowledgeTracer: Main service orchestrator
"""

from .models.llm_skill_extractor import LLM_Skill_Extractor
from .models.astnn_model import ASTNN
from .models.text_embedder import TextEmbedder
from .models.mlfbk_model import MLFBK_Model, MLFBK_KnowledgeTracer

# Re-export main classes for backward compatibility
__all__ = [
    'LLM_Skill_Extractor',
    'ASTNN',
    'TextEmbedder',
    'MLFBK_Model',
    'MLFBK_KnowledgeTracer'
]

# Version information
__version__ = '1.0.0'
__author__ = 'BIT Tutor Development Team'