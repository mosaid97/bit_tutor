"""
SQKT Integration Service

Integrates SQKT (Sequential Question-based Knowledge Tracing) with Neo4j knowledge graph.
Provides high-level interfaces for:
- Recording student interactions (submissions, questions, responses)
- Predicting student performance
- Updating knowledge states
- Training and evaluating the model
"""

from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime


class SQKTIntegrationService:
    """
    Service for integrating SQKT with Neo4j knowledge graph
    """
    
    def __init__(self, sqkt_tracer, neo4j_service=None):
        """
        Initialize SQKT integration service
        
        Args:
            sqkt_tracer: SQKT_KnowledgeTracer instance
            neo4j_service: Neo4j service for graph operations
        """
        self.sqkt = sqkt_tracer
        self.neo4j = neo4j_service
        self.exercise_id_map = {}  # Maps exercise names to IDs
        self.skill_id_map = {}  # Maps skill/concept names to IDs
        self.next_exercise_id = 1
        self.next_skill_id = 1
        
        print("✅ SQKT Integration Service initialized")
    
    def record_submission(
        self,
        student_id: str,
        exercise_name: str,
        skill_name: str,
        is_correct: bool,
        submission_text: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> Dict:
        """
        Record a student submission
        
        Args:
            student_id: Student identifier
            exercise_name: Exercise/problem name
            skill_name: Skill/concept name
            is_correct: Whether submission was correct
            submission_text: Submission code/text (optional)
            timestamp: Submission timestamp (optional)
            
        Returns:
            Dictionary with interaction details and prediction
        """
        # Get or create exercise and skill IDs
        exercise_id = self._get_or_create_exercise_id(exercise_name)
        skill_id = self._get_or_create_skill_id(skill_name)
        
        # Create interaction record
        interaction = {
            'exercise_id': exercise_id,
            'skill_id': skill_id,
            'response': 1 if is_correct else 0,
            'interaction_type': 1,  # 1 = submission
            'timestamp': timestamp or datetime.now(),
            'submission_text': submission_text
        }
        
        # Store in Neo4j if available
        if self.neo4j:
            self._store_interaction_in_neo4j(student_id, interaction, 'SUBMISSION')
        
        # Get student's interaction history
        history = self._get_student_history(student_id)
        history.append(interaction)
        
        # Predict next performance
        if len(history) > 1:
            knowledge_state, prediction = self.sqkt.predict_knowledge_state(
                history,
                return_predictions=True
            )
            current_prediction = prediction[-1].item()
        else:
            current_prediction = 0.5  # Neutral for first interaction
        
        return {
            'interaction': interaction,
            'predicted_performance': current_prediction,
            'knowledge_state_updated': True
        }
    
    def record_student_question(
        self,
        student_id: str,
        exercise_name: str,
        skill_name: str,
        question_text: str,
        timestamp: Optional[datetime] = None
    ) -> Dict:
        """
        Record a student question to educator
        
        Args:
            student_id: Student identifier
            exercise_name: Related exercise name
            skill_name: Related skill name
            question_text: Question text
            timestamp: Question timestamp (optional)
            
        Returns:
            Dictionary with interaction details
        """
        exercise_id = self._get_or_create_exercise_id(exercise_name)
        skill_id = self._get_or_create_skill_id(skill_name)
        
        interaction = {
            'exercise_id': exercise_id,
            'skill_id': skill_id,
            'response': 0,  # Not applicable for questions
            'interaction_type': 2,  # 2 = student question
            'timestamp': timestamp or datetime.now(),
            'question_text': question_text
        }
        
        if self.neo4j:
            self._store_interaction_in_neo4j(student_id, interaction, 'STUDENT_QUESTION')
        
        return {'interaction': interaction, 'recorded': True}
    
    def record_educator_response(
        self,
        student_id: str,
        exercise_name: str,
        skill_name: str,
        response_text: str,
        timestamp: Optional[datetime] = None
    ) -> Dict:
        """
        Record an educator response to student question
        
        Args:
            student_id: Student identifier
            exercise_name: Related exercise name
            skill_name: Related skill name
            response_text: Educator response text
            timestamp: Response timestamp (optional)
            
        Returns:
            Dictionary with interaction details
        """
        exercise_id = self._get_or_create_exercise_id(exercise_name)
        skill_id = self._get_or_create_skill_id(skill_name)
        
        interaction = {
            'exercise_id': exercise_id,
            'skill_id': skill_id,
            'response': 0,  # Not applicable for educator responses
            'interaction_type': 3,  # 3 = educator response
            'timestamp': timestamp or datetime.now(),
            'response_text': response_text
        }
        
        if self.neo4j:
            self._store_interaction_in_neo4j(student_id, interaction, 'EDUCATOR_RESPONSE')
        
        return {'interaction': interaction, 'recorded': True}
    
    def predict_performance(
        self,
        student_id: str,
        exercise_name: str,
        skill_name: str
    ) -> float:
        """
        Predict student performance on an exercise
        
        Args:
            student_id: Student identifier
            exercise_name: Exercise name
            skill_name: Skill name
            
        Returns:
            Predicted probability of correct response (0-1)
        """
        exercise_id = self._get_or_create_exercise_id(exercise_name)
        skill_id = self._get_or_create_skill_id(skill_name)
        
        history = self._get_student_history(student_id)
        
        if not history:
            return 0.5  # Neutral prediction for new students
        
        prediction = self.sqkt.predict_next_performance(
            history,
            exercise_id,
            skill_id
        )
        
        return prediction
    
    def get_knowledge_state(
        self,
        student_id: str
    ) -> Dict:
        """
        Get current knowledge state for a student
        
        Args:
            student_id: Student identifier
            
        Returns:
            Dictionary with knowledge state information
        """
        history = self._get_student_history(student_id)
        
        if not history:
            return {
                'student_id': student_id,
                'num_interactions': 0,
                'knowledge_state': None,
                'overall_mastery': 0.0
            }
        
        knowledge_state, predictions = self.sqkt.predict_knowledge_state(
            history,
            return_predictions=True
        )
        
        # Calculate overall mastery as average of recent predictions
        recent_predictions = predictions[-10:].mean().item() if len(predictions) > 0 else 0.5
        
        return {
            'student_id': student_id,
            'num_interactions': len(history),
            'knowledge_state': knowledge_state[-1].cpu().numpy().tolist() if len(knowledge_state) > 0 else None,
            'overall_mastery': recent_predictions,
            'recent_performance': predictions[-5:].cpu().numpy().tolist() if len(predictions) >= 5 else []
        }
    
    def _get_or_create_exercise_id(self, exercise_name: str) -> int:
        """Get or create exercise ID"""
        if exercise_name not in self.exercise_id_map:
            self.exercise_id_map[exercise_name] = self.next_exercise_id
            self.next_exercise_id += 1
        return self.exercise_id_map[exercise_name]
    
    def _get_or_create_skill_id(self, skill_name: str) -> int:
        """Get or create skill ID"""
        if skill_name not in self.skill_id_map:
            self.skill_id_map[skill_name] = self.next_skill_id
            self.next_skill_id += 1
        return self.skill_id_map[skill_name]
    
    def _get_student_history(self, student_id: str) -> List[Dict]:
        """
        Get student interaction history from Neo4j or cache
        
        Args:
            student_id: Student identifier
            
        Returns:
            List of interaction dictionaries
        """
        if not self.neo4j:
            return []
        
        # Query Neo4j for student interactions
        query = """
        MATCH (s:Student {student_id: $student_id})-[r:INTERACTED]->(i:Interaction)
        RETURN i.exercise_id as exercise_id,
               i.skill_id as skill_id,
               i.response as response,
               i.interaction_type as interaction_type,
               i.timestamp as timestamp
        ORDER BY i.timestamp
        """
        
        try:
            results = self.neo4j.graph.query(query, {'student_id': student_id})
            history = []
            for record in results:
                history.append({
                    'exercise_id': record.get('exercise_id', 0),
                    'skill_id': record.get('skill_id', 0),
                    'response': record.get('response', 0),
                    'interaction_type': record.get('interaction_type', 1)
                })
            return history
        except Exception as e:
            print(f"Error fetching student history: {e}")
            return []
    
    def _store_interaction_in_neo4j(
        self,
        student_id: str,
        interaction: Dict,
        interaction_label: str
    ):
        """Store interaction in Neo4j"""
        if not self.neo4j:
            return
        
        query = """
        MATCH (s:Student {student_id: $student_id})
        CREATE (i:Interaction {
            exercise_id: $exercise_id,
            skill_id: $skill_id,
            response: $response,
            interaction_type: $interaction_type,
            timestamp: $timestamp
        })
        CREATE (s)-[:INTERACTED]->(i)
        RETURN i
        """
        
        try:
            self.neo4j.graph.query(query, {
                'student_id': student_id,
                'exercise_id': interaction['exercise_id'],
                'skill_id': interaction['skill_id'],
                'response': interaction['response'],
                'interaction_type': interaction['interaction_type'],
                'timestamp': interaction['timestamp'].isoformat()
            })
        except Exception as e:
            print(f"Error storing interaction in Neo4j: {e}")

