# services/knowledge_tracing/__init__.py

"""
Knowledge Tracing Service

Integrated knowledge tracing with two state-of-the-art models:

1. SQKT (Sequential Question-based Knowledge Tracing)
   - Temporal sequence modeling with transformers
   - 81% accuracy (6% improvement over OKT)
   - Tracks submissions, questions, and educator responses
   - Reference: https://github.com/holi-lab/SQKT

2. MLFBK (Multi-Features with Latent Relations BERT Knowledge Tracing)
   - Multi-feature extraction (student_id, item_id, skill_id, etc.)
   - BERT-based latent representations
   - Captures complex feature interactions

Main Components:
- LLM_Skill_Extractor: Extracts skills from student questions using LLM
- ASTNN: Converts Abstract Syntax Trees to vector embeddings
- TextEmbedder: Embeds natural language text
- SQKT_Model: Sequential Question-based Knowledge Tracing model
- SQKT_KnowledgeTracer: SQKT service orchestrator
- MLFBK_Model: Multi-feature BERT-based knowledge tracing model
- MLFBK_KnowledgeTracer: MLFBK service orchestrator

Both models can be used independently or together for enhanced accuracy.
"""

from .models.llm_skill_extractor import LLM_Skill_Extractor

# Try to import torch-dependent models
try:
    from .models.astnn_model import ASTNN
    from .models.text_embedder import TextEmbedder
    from .models.mlfbk_model import (
        SQKT_Model,
        SQKT_KnowledgeTracer,
        MLFBK_Model,  # Backward compatibility
        MLFBK_KnowledgeTracer  # Backward compatibility
    )
    _TORCH_AVAILABLE = True
    print("✅ Knowledge Tracing models loaded: SQKT + MLFBK")
except ImportError as e:
    print(f"Warning: torch-dependent models not available: {e}")
    print("Creating mock implementations...")
    _TORCH_AVAILABLE = False

    # Create mock classes
    class ASTNN:
        def __init__(self, *args, **kwargs):
            raise ImportError("ASTNN requires torch. Please install: pip install torch")

    class TextEmbedder:
        def __init__(self, *args, **kwargs):
            raise ImportError("TextEmbedder requires torch. Please install: pip install torch")

    class SQKT_Model:
        def __init__(self, *args, **kwargs):
            raise ImportError("SQKT_Model requires torch. Please install: pip install torch")

    class SQKT_KnowledgeTracer:
        def __init__(self, *args, **kwargs):
            raise ImportError("SQKT_KnowledgeTracer requires torch. Please install: pip install torch")

    class MLFBK_Model:
        def __init__(self, *args, **kwargs):
            raise ImportError("MLFBK_Model requires torch. Please install: pip install torch")

    class MLFBK_KnowledgeTracer:
        def __init__(self, *args, **kwargs):
            raise ImportError("MLFBK_KnowledgeTracer requires torch. Please install: pip install torch")

# Re-export main classes
__all__ = [
    'LLM_Skill_Extractor',
    'ASTNN',
    'TextEmbedder',
    'SQKT_Model',
    'SQKT_KnowledgeTracer',
    'MLFBK_Model',
    'MLFBK_KnowledgeTracer'
]

# Version information
__version__ = '3.0.0'  # Updated to 3.0.0 for SQKT + MLFBK integration
__author__ = 'BIT Tutor Development Team'