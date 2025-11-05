# services/recommendation/__init__.py

"""
Recommendation Service

This service implements advanced recommendation algorithms including RL-based recommendation
agents and LLM-powered content generation for personalized learning experiences.

Main Components:
- LLM_Content_Generator: Generates personalized educational content (exercises, explanations, hints)
- RL_Recommender_Agent: Reinforcement learning agent for optimal recommendation policies
- RecommendationService: Main service orchestrator that integrates all components
"""

from .models.llm_content_generator import LLM_Content_Generator

# Try to import torch-dependent models
try:
    from .models.rl_recommender_agent import RL_Recommender_Agent
    from .services.recommendation_service import RecommendationService
    _TORCH_AVAILABLE = True
except ImportError as e:
    print(f"Warning: torch-dependent recommendation models not available: {e}")
    print("Creating mock implementations...")
    _TORCH_AVAILABLE = False

    # Create mock classes
    class RL_Recommender_Agent:
        def __init__(self, *args, **kwargs):
            raise ImportError("RL_Recommender_Agent requires torch. Please install: pip install torch")

    class RecommendationService:
        def __init__(self, *args, **kwargs):
            raise ImportError("RecommendationService requires torch. Please install: pip install torch")

# Re-export main classes for backward compatibility
__all__ = [
    'LLM_Content_Generator',
    'RL_Recommender_Agent',
    'RecommendationService'
]

# Version information
__version__ = '1.0.0'
__author__ = 'BIT Tutor Development Team'