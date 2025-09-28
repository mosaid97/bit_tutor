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
from .models.rl_recommender_agent import RL_Recommender_Agent
from .services.recommendation_service import RecommendationService

# Re-export main classes for backward compatibility
__all__ = [
    'LLM_Content_Generator',
    'RL_Recommender_Agent',
    'RecommendationService'
]

# Version information
__version__ = '1.0.0'
__author__ = 'BIT Tutor Development Team'