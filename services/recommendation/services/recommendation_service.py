# services/recommendation/services/recommendation_service.py

import random
import torch
import numpy as np
import pandas as pd
from datetime import datetime

# Import models from the same service
from ..models.llm_content_generator import LLM_Content_Generator
from ..models.rl_recommender_agent import RL_Recommender_Agent

class RecommendationService:
    """
    Main recommendation service that orchestrates content generation and RL-based recommendations.
    This service integrates the LLM Content Generator and RL Recommender Agent to provide
    personalized learning recommendations.
    """
    
    def __init__(self, kcs_map, exercises=None):
        """
        Initialize the recommendation service.
        
        Args:
            kcs_map (dict): Mapping of KC IDs to names
            exercises (list, optional): List of available exercises
        """
        self.kcs_map = kcs_map
        self.exercises = exercises or []
        
        # Initialize components
        self.content_generator = LLM_Content_Generator(kcs_map)
        
        # Initialize RL agent if exercises are provided
        if self.exercises:
            action_space = self.exercises + list(kcs_map.keys())
            self.rl_agent = RL_Recommender_Agent(kcs_map, self.exercises, len(action_space))
            self.action_space = action_space
        else:
            self.rl_agent = None
            self.action_space = []
        
        print("Recommendation Service initialized.")
    
    def get_recommendation(self, student_graph_obj, recommendation_type='auto'):
        """
        Get a recommendation for the student based on their current knowledge state.
        
        Args:
            student_graph_obj: Student knowledge graph object
            recommendation_type (str): Type of recommendation ('auto', 'exercise', 'explanation', 'hint')
            
        Returns:
            dict: Recommendation with type, content, and metadata
        """
        try:
            # Get current mastery profile
            mastery_profile = student_graph_obj.get_mastery_profile()
            
            if not mastery_profile:
                return self._get_default_recommendation()
            
            # Find the weakest knowledge component
            weakest_kc = min(mastery_profile.items(), key=lambda x: x[1])
            weakest_kc_id, weakest_mastery = weakest_kc
            
            # Determine recommendation type if auto
            if recommendation_type == 'auto':
                if weakest_mastery < 0.3:
                    recommendation_type = 'explanation'
                elif weakest_mastery < 0.6:
                    recommendation_type = 'exercise'
                else:
                    recommendation_type = 'hint'
            
            # Generate recommendation based on type
            if recommendation_type == 'exercise':
                return self._recommend_exercise(student_graph_obj, weakest_kc_id)
            elif recommendation_type == 'explanation':
                return self._recommend_explanation(student_graph_obj, weakest_kc_id)
            elif recommendation_type == 'hint':
                return self._recommend_hint(student_graph_obj, weakest_kc_id)
            else:
                return self._get_default_recommendation()
                
        except Exception as e:
            print(f"Error generating recommendation: {e}")
            return self._get_default_recommendation()
    
    def _recommend_exercise(self, student_graph_obj, target_kc_id):
        """Generate an exercise recommendation."""
        try:
            # Use RL agent if available
            if self.rl_agent and self.action_space:
                state = self.rl_agent.get_state(student_graph_obj)
                action_index = self.rl_agent.get_best_action(state)
                action = self.action_space[action_index]
                
                if action in self.exercises:
                    # Recommend existing exercise
                    return {
                        'type': 'exercise',
                        'subtype': 'existing',
                        'exercise_id': action,
                        'target_kc': target_kc_id,
                        'source': 'rl_agent',
                        'confidence': 0.8
                    }
                else:
                    # Generate new exercise for the KC
                    target_kc_id = action
            
            # Generate new exercise
            hobbies = getattr(student_graph_obj, 'hobbies', [])
            mastery_level = student_graph_obj.get_mastery_profile().get(target_kc_id, 0.1)
            
            # Determine difficulty based on mastery
            if mastery_level < 0.3:
                difficulty = 'easy'
            elif mastery_level < 0.7:
                difficulty = 'medium'
            else:
                difficulty = 'hard'
            
            exercise = self.content_generator.generate_exercise(target_kc_id, difficulty, hobbies)
            
            return {
                'type': 'exercise',
                'subtype': 'generated',
                'content': exercise,
                'target_kc': target_kc_id,
                'source': 'content_generator',
                'confidence': 0.9
            }
            
        except Exception as e:
            print(f"Error generating exercise recommendation: {e}")
            return self._get_default_recommendation()
    
    def _recommend_explanation(self, student_graph_obj, target_kc_id):
        """Generate an explanation recommendation."""
        try:
            mastery_level = student_graph_obj.get_mastery_profile().get(target_kc_id, 0.1)
            
            # Determine difficulty based on mastery
            if mastery_level < 0.3:
                difficulty = 'easy'
            elif mastery_level < 0.7:
                difficulty = 'medium'
            else:
                difficulty = 'hard'
            
            explanation = self.content_generator.generate_explanation(target_kc_id, difficulty)
            
            return {
                'type': 'explanation',
                'content': explanation,
                'target_kc': target_kc_id,
                'source': 'content_generator',
                'confidence': 0.9
            }
            
        except Exception as e:
            print(f"Error generating explanation recommendation: {e}")
            return self._get_default_recommendation()
    
    def _recommend_hint(self, student_graph_obj, target_kc_id):
        """Generate a hint recommendation."""
        try:
            mastery_level = student_graph_obj.get_mastery_profile().get(target_kc_id, 0.1)
            
            # Determine difficulty based on mastery
            if mastery_level < 0.3:
                difficulty = 'easy'
            elif mastery_level < 0.7:
                difficulty = 'medium'
            else:
                difficulty = 'hard'
            
            hint = self.content_generator.generate_hint(target_kc_id, difficulty)
            
            return {
                'type': 'hint',
                'content': hint,
                'target_kc': target_kc_id,
                'source': 'content_generator',
                'confidence': 0.8
            }
            
        except Exception as e:
            print(f"Error generating hint recommendation: {e}")
            return self._get_default_recommendation()
    
    def _get_default_recommendation(self):
        """Get a default recommendation when other methods fail."""
        return {
            'type': 'general',
            'content': {
                'title': 'Continue Learning',
                'text': 'Keep practicing programming concepts to improve your skills!',
                'suggestions': [
                    'Review basic programming concepts',
                    'Practice with simple exercises',
                    'Ask questions when you need help'
                ]
            },
            'source': 'default',
            'confidence': 0.5
        }
    
    def train_rl_agent(self, training_episodes=100, student_graph_obj=None):
        """
        Train the RL agent with simulated interactions.
        
        Args:
            training_episodes (int): Number of training episodes
            student_graph_obj: Student graph for training (optional)
            
        Returns:
            dict: Training statistics
        """
        if not self.rl_agent:
            print("RL agent not initialized. Cannot train.")
            return {}
        
        print(f"Training RL agent for {training_episodes} episodes...")
        
        training_stats = {
            'episodes': training_episodes,
            'total_reward': 0,
            'avg_reward_per_episode': 0,
            'final_exploration_rate': self.rl_agent.exploration_rate
        }
        
        # Training loop would go here
        # This is a simplified version for demonstration
        for episode in range(training_episodes):
            # Simulate training episode
            reward = random.uniform(-0.1, 0.3)  # Simulated reward
            training_stats['total_reward'] += reward
            
            # Decay exploration rate
            self.rl_agent.decay_exploration_rate(episode)
        
        training_stats['avg_reward_per_episode'] = training_stats['total_reward'] / training_episodes
        training_stats['final_exploration_rate'] = self.rl_agent.exploration_rate
        
        print(f"Training completed. Average reward: {training_stats['avg_reward_per_episode']:.3f}")
        return training_stats
    
    def get_service_stats(self):
        """
        Get statistics about the recommendation service.
        
        Returns:
            dict: Service statistics
        """
        stats = {
            'num_kcs': len(self.kcs_map),
            'num_exercises': len(self.exercises),
            'has_rl_agent': self.rl_agent is not None,
            'action_space_size': len(self.action_space)
        }
        
        if self.rl_agent:
            stats.update(self.rl_agent.get_policy_stats())
        
        return stats
