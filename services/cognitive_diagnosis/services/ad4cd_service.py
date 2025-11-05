"""
AD4CD Integration Service

Integrates AD4CD (Anomaly Detection for Cognitive Diagnosis) with Neo4j knowledge graph
and SQKT knowledge tracing for comprehensive student assessment.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json


class AD4CDIntegrationService:
    """
    Service for integrating AD4CD with Neo4j and SQKT
    
    Provides:
    - Anomaly detection in student responses
    - Enhanced cognitive diagnosis
    - Cheating/guessing detection
    - Confidence-weighted mastery updates
    """
    
    def __init__(self, ad4cd_diagnosis, neo4j_service=None, sqkt_service=None):
        """
        Initialize AD4CD integration service
        
        Args:
            ad4cd_diagnosis: AD4CD_CognitiveDiagnosis instance
            neo4j_service: Neo4j service for graph operations
            sqkt_service: SQKT service for knowledge tracing
        """
        self.ad4cd = ad4cd_diagnosis
        self.neo4j = neo4j_service
        self.sqkt = sqkt_service
        
        # ID mappings
        self.student_id_map = {}
        self.exercise_id_map = {}
        self.concept_id_map = {}
        self.next_student_id = 1
        self.next_exercise_id = 1
        self.next_concept_id = 1
        
        print("✅ AD4CD Integration Service initialized")
    
    def diagnose_response(
        self,
        student_id: str,
        exercise_name: str,
        concept_name: str,
        response: int,
        timestamp: Optional[datetime] = None
    ) -> Dict:
        """
        Diagnose a student response with anomaly detection
        
        Args:
            student_id: Student identifier
            exercise_name: Exercise/question name
            concept_name: Concept name
            response: Student response (0=incorrect, 1=correct)
            timestamp: Response timestamp
            
        Returns:
            Dictionary with diagnosis results
        """
        # Get or create numeric IDs
        student_num_id = self._get_or_create_student_id(student_id)
        exercise_num_id = self._get_or_create_exercise_id(exercise_name)
        concept_num_id = self._get_or_create_concept_id(concept_name)
        
        # Run AD4CD diagnosis
        diagnosis = self.ad4cd.diagnose_with_anomaly_detection(
            student_id=student_num_id,
            exercise_id=exercise_num_id,
            concept_id=concept_num_id,
            response=response
        )
        
        # Store in Neo4j if available
        if self.neo4j:
            self._store_diagnosis_in_neo4j(
                student_id, exercise_name, concept_name, response, diagnosis, timestamp
            )
        
        # Update SQKT if available and response is not anomalous
        if self.sqkt and not diagnosis['is_anomaly']:
            self.sqkt.record_submission(
                student_id=student_id,
                exercise_name=exercise_name,
                skill_name=concept_name,
                is_correct=bool(response)
            )
        
        # Add metadata
        diagnosis['student_id'] = student_id
        diagnosis['exercise_name'] = exercise_name
        diagnosis['concept_name'] = concept_name
        diagnosis['response'] = response
        diagnosis['timestamp'] = timestamp or datetime.now()
        
        return diagnosis
    
    def get_student_anomaly_report(self, student_id: str) -> Dict:
        """
        Get anomaly report for a student
        
        Args:
            student_id: Student identifier
            
        Returns:
            Dictionary with anomaly statistics
        """
        if not self.neo4j:
            return {
                'student_id': student_id,
                'total_responses': 0,
                'anomalous_responses': 0,
                'anomaly_rate': 0.0,
                'cheating_incidents': 0,
                'guessing_incidents': 0,
                'careless_mistakes': 0
            }
        
        # Query Neo4j for student's diagnoses
        query = """
        MATCH (s:Student {student_id: $student_id})-[:HAS_DIAGNOSIS]->(d:Diagnosis)
        RETURN d.is_anomaly as is_anomaly,
               d.diagnosis as diagnosis,
               d.anomaly_score as anomaly_score
        """
        
        try:
            results = self.neo4j.graph.query(query, {'student_id': student_id})
            
            total = len(results)
            anomalous = sum(1 for r in results if r.get('is_anomaly', False))
            cheating = sum(1 for r in results if r.get('diagnosis') == 'possible_cheating')
            guessing = sum(1 for r in results if r.get('diagnosis') == 'guessing')
            careless = sum(1 for r in results if r.get('diagnosis') == 'careless_mistake')
            
            return {
                'student_id': student_id,
                'total_responses': total,
                'anomalous_responses': anomalous,
                'anomaly_rate': anomalous / total if total > 0 else 0.0,
                'cheating_incidents': cheating,
                'guessing_incidents': guessing,
                'careless_mistakes': careless,
                'avg_anomaly_score': sum(r.get('anomaly_score', 0) for r in results) / total if total > 0 else 0.0
            }
        except Exception as e:
            print(f"Error getting anomaly report: {e}")
            return {
                'student_id': student_id,
                'total_responses': 0,
                'anomalous_responses': 0,
                'anomaly_rate': 0.0,
                'cheating_incidents': 0,
                'guessing_incidents': 0,
                'careless_mistakes': 0,
                'error': str(e)
            }
    
    def update_mastery_with_confidence(
        self,
        student_id: str,
        concept_name: str,
        performance: float,
        confidence: float
    ):
        """
        Update student mastery with confidence weighting
        
        Args:
            student_id: Student identifier
            concept_name: Concept name
            performance: Performance score (0-1)
            confidence: Confidence in the assessment (0-1)
        """
        if not self.neo4j:
            return
        
        # Confidence-weighted mastery update
        weighted_performance = performance * confidence
        
        query = """
        MATCH (s:Student {student_id: $student_id})
        MATCH (c:Concept {name: $concept_name})
        MERGE (s)-[k:KNOWS]->(c)
        ON CREATE SET k.mastery_level = $mastery,
                     k.confidence = $confidence,
                     k.last_updated = $timestamp
        ON MATCH SET k.mastery_level = (k.mastery_level * 0.7 + $mastery * 0.3),
                    k.confidence = $confidence,
                    k.last_updated = $timestamp
        RETURN k.mastery_level as mastery_level
        """
        
        try:
            self.neo4j.graph.query(query, {
                'student_id': student_id,
                'concept_name': concept_name,
                'mastery': weighted_performance,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            print(f"Error updating mastery: {e}")
    
    def get_filtered_performance_data(
        self,
        student_id: str,
        filter_anomalies: bool = True
    ) -> List[Dict]:
        """
        Get student performance data with optional anomaly filtering
        
        Args:
            student_id: Student identifier
            filter_anomalies: Whether to filter out anomalous responses
            
        Returns:
            List of performance records
        """
        if not self.neo4j:
            return []
        
        query = """
        MATCH (s:Student {student_id: $student_id})-[:HAS_DIAGNOSIS]->(d:Diagnosis)
        WHERE ($filter_anomalies = false OR d.is_anomaly = false)
        RETURN d.exercise_name as exercise,
               d.concept_name as concept,
               d.response as response,
               d.predicted_performance as predicted,
               d.is_anomaly as is_anomaly,
               d.diagnosis as diagnosis,
               d.timestamp as timestamp
        ORDER BY d.timestamp
        """
        
        try:
            results = self.neo4j.graph.query(query, {
                'student_id': student_id,
                'filter_anomalies': filter_anomalies
            })
            
            return [dict(r) for r in results]
        except Exception as e:
            print(f"Error getting performance data: {e}")
            return []
    
    def _get_or_create_student_id(self, student_id: str) -> int:
        """Get or create numeric student ID"""
        if student_id not in self.student_id_map:
            self.student_id_map[student_id] = self.next_student_id
            self.next_student_id += 1
        return self.student_id_map[student_id]
    
    def _get_or_create_exercise_id(self, exercise_name: str) -> int:
        """Get or create numeric exercise ID"""
        if exercise_name not in self.exercise_id_map:
            self.exercise_id_map[exercise_name] = self.next_exercise_id
            self.next_exercise_id += 1
        return self.exercise_id_map[exercise_name]
    
    def _get_or_create_concept_id(self, concept_name: str) -> int:
        """Get or create numeric concept ID"""
        if concept_name not in self.concept_id_map:
            self.concept_id_map[concept_name] = self.next_concept_id
            self.next_concept_id += 1
        return self.concept_id_map[concept_name]
    
    def _store_diagnosis_in_neo4j(
        self,
        student_id: str,
        exercise_name: str,
        concept_name: str,
        response: int,
        diagnosis: Dict,
        timestamp: Optional[datetime]
    ):
        """Store diagnosis results in Neo4j"""
        if not self.neo4j:
            return
        
        query = """
        MATCH (s:Student {student_id: $student_id})
        CREATE (d:Diagnosis {
            exercise_name: $exercise_name,
            concept_name: $concept_name,
            response: $response,
            predicted_performance: $predicted_performance,
            anomaly_score: $anomaly_score,
            is_anomaly: $is_anomaly,
            confidence: $confidence,
            diagnosis: $diagnosis,
            timestamp: $timestamp
        })
        CREATE (s)-[:HAS_DIAGNOSIS]->(d)
        RETURN d
        """
        
        try:
            self.neo4j.graph.query(query, {
                'student_id': student_id,
                'exercise_name': exercise_name,
                'concept_name': concept_name,
                'response': response,
                'predicted_performance': diagnosis['predicted_performance'],
                'anomaly_score': diagnosis['anomaly_score'],
                'is_anomaly': diagnosis['is_anomaly'],
                'confidence': diagnosis['confidence'],
                'diagnosis': diagnosis['diagnosis'],
                'timestamp': (timestamp or datetime.now()).isoformat()
            })
            
            # Update mastery with confidence weighting
            if not diagnosis['is_anomaly']:
                self.update_mastery_with_confidence(
                    student_id,
                    concept_name,
                    diagnosis['predicted_performance'],
                    diagnosis['confidence']
                )
        except Exception as e:
            print(f"Error storing diagnosis in Neo4j: {e}")

