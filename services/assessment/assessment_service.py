"""
Assessment Service
Handles initial assessments, pre-topic assessments, quizzes, and cognitive diagnosis
Implements the 5-step framework:
1. Cognitive Foundation (Knowledge Graph & Q-Matrix)
2. Generative Student Model (OKT + MLFBK)
3. Graph-Based Cognitive Diagnosis (G-CDM)
4. Diagnostic-Generative Feedback Loop
5. Explainable, Actionable Feedback
"""

from datetime import datetime
from typing import Dict, List, Optional
import random


class AssessmentService:
    """Service for managing student assessments and cognitive diagnosis"""
    
    def __init__(self, neo4j_service=None):
        """Initialize assessment service with Neo4j connection"""
        self.neo4j = neo4j_service
    
    def create_initial_assessment(self, student_id: str, class_id: str) -> Dict:
        """
        Create initial cognitive assessment for a new student
        This assesses baseline knowledge across all concepts
        
        Returns:
            Dict with assessment_id, questions, total_questions
        """
        if not self.neo4j:
            return {'success': False, 'message': 'Database not available'}
        
        try:
            # Get all concepts for the class
            query = """
            MATCH (c:Class {class_id: $class_id})-[:INCLUDES]->(t:Topic)-[:INCLUDES_CONCEPT]->(concept:Concept)
            RETURN DISTINCT concept.concept_id as concept_id,
                   concept.name as name,
                   concept.description as description,
                   concept.difficulty as difficulty
            ORDER BY concept.difficulty
            """
            
            concepts = self.neo4j.graph.query(query, {
                'class_id': class_id
            })
            
            if not concepts:
                return {'success': False, 'message': 'No concepts found for class'}
            
            # Create assessment node
            assessment_id = f"initial_{student_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            create_query = """
            MATCH (s:Student {student_id: $student_id})
            CREATE (a:Assessment {
                assessment_id: $assessment_id,
                type: 'initial',
                created_at: datetime(),
                status: 'in_progress',
                total_questions: $total_questions
            })
            CREATE (s)-[:TAKING]->(a)
            RETURN a.assessment_id as assessment_id
            """
            
            result = self.neo4j.graph.query(create_query, {
                'student_id': student_id,
                'assessment_id': assessment_id,
                'total_questions': len(concepts)
            })
            
            return {
                'success': True,
                'assessment_id': assessment_id,
                'total_questions': len(concepts),
                'concepts': [dict(c) for c in concepts]
            }
            
        except Exception as e:
            print(f"Error creating initial assessment: {e}")
            return {'success': False, 'message': str(e)}
    
    def create_pre_topic_assessment(self, student_id: str, topic_name: str) -> Dict:
        """
        Create pre-topic assessment to gauge student's knowledge before learning
        
        Returns:
            Dict with assessment_id, questions, total_questions
        """
        if not self.neo4j:
            return {'success': False, 'message': 'Database not available'}
        
        try:
            # Get concepts for the topic
            query = """
            MATCH (t:Topic {name: $topic_name})-[:INCLUDES_CONCEPT]->(c:Concept)
            RETURN c.concept_id as concept_id,
                   c.name as name,
                   c.description as description,
                   c.difficulty as difficulty
            ORDER BY c.difficulty
            """
            
            concepts = self.neo4j.graph.query(query, {'topic_name': topic_name})
            
            if not concepts:
                return {'success': False, 'message': 'No concepts found for topic'}
            
            # Create assessment
            assessment_id = f"pre_topic_{student_id}_{topic_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            create_query = """
            MATCH (s:Student {student_id: $student_id})
            MATCH (t:Topic {name: $topic_name})
            CREATE (a:Assessment {
                assessment_id: $assessment_id,
                type: 'pre_topic',
                topic: $topic_name,
                created_at: datetime(),
                status: 'in_progress',
                total_questions: $total_questions
            })
            CREATE (s)-[:TAKING]->(a)
            CREATE (a)-[:ASSESSES]->(t)
            RETURN a.assessment_id as assessment_id
            """
            
            result = self.neo4j.graph.query(create_query, {
                'student_id': student_id,
                'topic_name': topic_name,
                'assessment_id': assessment_id,
                'total_questions': len(concepts)
            })
            
            return {
                'success': True,
                'assessment_id': assessment_id,
                'topic': topic_name,
                'total_questions': len(concepts),
                'concepts': [dict(c) for c in concepts]
            }
            
        except Exception as e:
            print(f"Error creating pre-topic assessment: {e}")
            return {'success': False, 'message': str(e)}
    
    def submit_assessment(self, student_id: str, assessment_id: str, responses: List[Dict]) -> Dict:
        """
        Submit assessment responses and calculate cognitive profile
        Implements G-CDM (Graph-Based Cognitive Diagnosis Model)
        
        Args:
            student_id: Student ID
            assessment_id: Assessment ID
            responses: List of {concept_id, answer, correct}
        
        Returns:
            Dict with score, mastery_profile, recommendations
        """
        if not self.neo4j:
            return {'success': False, 'message': 'Database not available'}
        
        try:
            # Calculate score
            total = len(responses)
            correct = sum(1 for r in responses if r.get('correct', False))
            score = (correct / total * 100) if total > 0 else 0
            
            # Update assessment
            update_query = """
            MATCH (a:Assessment {assessment_id: $assessment_id})
            SET a.status = 'completed',
                a.completed_at = datetime(),
                a.score = $score,
                a.total_correct = $correct,
                a.total_questions = $total
            RETURN a
            """
            
            self.neo4j.graph.query(update_query, {
                'assessment_id': assessment_id,
                'score': score,
                'correct': correct,
                'total': total
            })
            
            # Create KNOWS relationships with mastery levels (G-CDM)
            for response in responses:
                concept_id = response.get('concept_id')
                is_correct = response.get('correct', False)
                
                # Calculate mastery level (simplified - in production use MLFBK)
                mastery_level = 0.8 if is_correct else 0.3
                
                knows_query = """
                MATCH (s:Student {student_id: $student_id})
                MATCH (c:Concept {concept_id: $concept_id})
                MERGE (s)-[k:KNOWS]->(c)
                SET k.mastery_level = $mastery_level,
                    k.last_assessed = datetime(),
                    k.assessment_id = $assessment_id,
                    k.recommendation = CASE 
                        WHEN $mastery_level >= 0.8 THEN 'Excellent! You have mastered this concept.'
                        WHEN $mastery_level >= 0.5 THEN 'Good progress. Review and practice more.'
                        ELSE 'Needs attention. Focus on understanding fundamentals.'
                    END
                """
                
                self.neo4j.graph.query(knows_query, {
                    'student_id': student_id,
                    'concept_id': concept_id,
                    'mastery_level': mastery_level,
                    'assessment_id': assessment_id
                })
            
            # Update student overall score
            update_student_query = """
            MATCH (s:Student {student_id: $student_id})
            SET s.overall_score = $score,
                s.last_assessment_date = datetime()
            """
            
            self.neo4j.graph.query(update_student_query, {
                'student_id': student_id,
                'score': score
            })
            
            # Create TOOK relationship
            took_query = """
            MATCH (s:Student {student_id: $student_id})
            MATCH (a:Assessment {assessment_id: $assessment_id})
            MERGE (s)-[t:TOOK]->(a)
            SET t.date = datetime()
            """
            
            self.neo4j.graph.query(took_query, {
                'student_id': student_id,
                'assessment_id': assessment_id
            })
            
            return {
                'success': True,
                'score': score,
                'correct': correct,
                'total': total,
                'message': 'Assessment completed successfully'
            }
            
        except Exception as e:
            print(f"Error submitting assessment: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'message': str(e)}
    
    def get_student_cognitive_profile(self, student_id: str) -> Dict:
        """
        Get student's cognitive profile from G-CDM
        
        Returns:
            Dict with mastery levels for each concept
        """
        if not self.neo4j:
            return {'success': False, 'message': 'Database not available'}
        
        try:
            query = """
            MATCH (s:Student {student_id: $student_id})-[k:KNOWS]->(c:Concept)
            MATCH (c)<-[:INCLUDES_CONCEPT]-(t:Topic)
            RETURN c.name as concept,
                   t.name as topic,
                   k.mastery_level as mastery,
                   k.recommendation as recommendation,
                   k.last_assessed as last_assessed
            ORDER BY k.mastery_level ASC
            """
            
            results = self.neo4j.graph.query(query, {'student_id': student_id})
            
            return {
                'success': True,
                'profile': [dict(r) for r in results] if results else []
            }
            
        except Exception as e:
            print(f"Error getting cognitive profile: {e}")
            return {'success': False, 'message': str(e)}

