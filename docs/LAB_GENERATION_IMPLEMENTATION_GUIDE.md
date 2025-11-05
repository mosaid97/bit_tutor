# Personalized Lab Generation - Implementation Guide
## Step-by-Step Implementation of AI Model Integration

**Version**: 1.0  
**Date**: November 4, 2025  
**Purpose**: Practical guide for implementing personalized lab generation using SQKT, G-CDM+AD4CD, and RL Agent outputs

---

## 🎯 Implementation Overview

This guide shows how to:
1. Add required methods to existing AI services
2. Create a unified lab generation service
3. Integrate with the lab generator
4. Test the complete pipeline

---

## 📝 Step 1: Extend Knowledge Tracing Service (SQKT)

### File: `services/knowledge_tracing/services/sqkt_service.py`

Add these methods to `SQKTIntegrationService` class:

```python
def get_concept_predictions(self, student_id: str, concepts: List[str]) -> Dict:
    """
    Get mastery predictions for specific concepts
    
    Args:
        student_id: Student identifier
        concepts: List of concept names
        
    Returns:
        Dictionary with predictions for each concept
    """
    history = self._get_student_history(student_id)
    
    if not history:
        return {concept: {
            'current_mastery': 0.5,
            'predicted_next': 0.5,
            'confidence': 0.0,
            'trend': 'unknown'
        } for concept in concepts}
    
    predictions = {}
    
    for concept in concepts:
        # Filter history for this concept
        concept_history = [h for h in history if h.get('skill_name') == concept]
        
        if not concept_history:
            predictions[concept] = {
                'current_mastery': 0.5,
                'predicted_next': 0.5,
                'confidence': 0.0,
                'trend': 'unknown'
            }
            continue
        
        # Get recent performance
        recent_responses = [h['response'] for h in concept_history[-5:]]
        current_mastery = sum(recent_responses) / len(recent_responses) if recent_responses else 0.5
        
        # Predict next performance using SQKT
        skill_id = self._get_or_create_skill_id(concept)
        next_exercise_id = self.next_exercise_id
        
        predicted_next = self.sqkt.predict_next_performance(
            history, next_exercise_id, skill_id
        )
        
        # Calculate trend
        if len(recent_responses) >= 3:
            first_half = sum(recent_responses[:len(recent_responses)//2]) / (len(recent_responses)//2)
            second_half = sum(recent_responses[len(recent_responses)//2:]) / (len(recent_responses) - len(recent_responses)//2)
            
            if second_half > first_half + 0.1:
                trend = 'improving'
            elif second_half < first_half - 0.1:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'unknown'
        
        predictions[concept] = {
            'current_mastery': float(current_mastery),
            'predicted_next': float(predicted_next),
            'confidence': min(len(concept_history) / 10.0, 1.0),  # More data = higher confidence
            'trend': trend
        }
    
    return predictions


def get_skill_gaps(self, student_id: str, concepts: List[str], target_mastery: float = 0.7) -> List[Dict]:
    """
    Identify skill gaps for a student
    
    Args:
        student_id: Student identifier
        concepts: List of concepts to analyze
        target_mastery: Target mastery level (default 0.7)
        
    Returns:
        List of skill gaps sorted by priority
    """
    predictions = self.get_concept_predictions(student_id, concepts)
    
    gaps = []
    for concept, pred in predictions.items():
        current = pred['current_mastery']
        gap_size = max(0, target_mastery - current)
        
        if gap_size > 0:
            # Priority based on gap size and trend
            if pred['trend'] == 'declining':
                priority_multiplier = 1.5
            elif pred['trend'] == 'improving':
                priority_multiplier = 0.8
            else:
                priority_multiplier = 1.0
            
            priority_score = gap_size * priority_multiplier
            
            # Determine priority level
            if priority_score > 0.3:
                priority = 'high'
                recommended_exercises = 3
            elif priority_score > 0.15:
                priority = 'medium'
                recommended_exercises = 2
            else:
                priority = 'low'
                recommended_exercises = 1
            
            gaps.append({
                'concept': concept,
                'current_mastery': current,
                'target_mastery': target_mastery,
                'gap_size': gap_size,
                'priority': priority,
                'priority_score': priority_score,
                'recommended_exercises': recommended_exercises,
                'trend': pred['trend']
            })
    
    # Sort by priority score (highest first)
    gaps.sort(key=lambda x: x['priority_score'], reverse=True)
    
    return gaps


def get_learning_trajectory(self, student_id: str) -> Dict:
    """
    Get student's learning trajectory and velocity
    
    Args:
        student_id: Student identifier
        
    Returns:
        Dictionary with trajectory information
    """
    history = self._get_student_history(student_id)
    
    if not history:
        return {
            'num_interactions': 0,
            'recent_performance': [],
            'overall_mastery': 0.0,
            'learning_velocity': 0.0,
            'consistency_score': 0.0
        }
    
    # Get recent performance
    recent_responses = [h['response'] for h in history[-10:]]
    recent_performance = [sum(recent_responses[max(0, i-4):i+1]) / min(5, i+1) 
                         for i in range(len(recent_responses))]
    
    # Calculate overall mastery
    overall_mastery = sum(recent_responses) / len(recent_responses) if recent_responses else 0.0
    
    # Calculate learning velocity (improvement rate)
    if len(recent_performance) >= 5:
        first_half_avg = sum(recent_performance[:len(recent_performance)//2]) / (len(recent_performance)//2)
        second_half_avg = sum(recent_performance[len(recent_performance)//2:]) / (len(recent_performance) - len(recent_performance)//2)
        learning_velocity = (second_half_avg - first_half_avg) / (len(recent_performance)//2)
    else:
        learning_velocity = 0.0
    
    # Calculate consistency (inverse of standard deviation)
    if len(recent_performance) >= 3:
        mean_perf = sum(recent_performance) / len(recent_performance)
        variance = sum((p - mean_perf) ** 2 for p in recent_performance) / len(recent_performance)
        std_dev = variance ** 0.5
        consistency_score = max(0, 1.0 - std_dev)
    else:
        consistency_score = 0.5
    
    return {
        'num_interactions': len(history),
        'recent_performance': recent_performance[-5:],
        'overall_mastery': overall_mastery,
        'learning_velocity': learning_velocity,
        'consistency_score': consistency_score
    }
```

---

## 🧠 Step 2: Extend Cognitive Diagnosis Service (AD4CD)

### File: `services/cognitive_diagnosis/services/ad4cd_service.py`

Add these methods to `AD4CDIntegrationService` class:

```python
def get_mastery_profile(self, student_id: str, concepts: List[str]) -> Dict:
    """
    Get detailed mastery profile for concepts
    
    Args:
        student_id: Student identifier
        concepts: List of concepts to analyze
        
    Returns:
        Dictionary with mastery profile for each concept
    """
    if not self.neo4j:
        # Fallback without Neo4j
        return {concept: {
            'mastery_level': 0.5,
            'confidence': 0.0,
            'last_assessed': None,
            'num_assessments': 0,
            'mastery_category': 'developing'
        } for concept in concepts}
    
    profile = {}
    
    for concept in concepts:
        query = """
        MATCH (s:Student {student_id: $student_id})-[k:KNOWS]->(c:Concept {name: $concept})
        OPTIONAL MATCH (s)-[:HAS_DIAGNOSIS]->(d:Diagnosis)-[:FOR_CONCEPT]->(c)
        RETURN k.mastery_level as mastery,
               k.confidence as confidence,
               k.last_updated as last_assessed,
               count(d) as num_assessments
        """
        
        try:
            result = self.neo4j.graph.query(query, {
                'student_id': student_id,
                'concept': concept
            })
            
            if result:
                record = result[0]
                mastery = record.get('mastery', 0.5)
                
                # Determine category
                if mastery >= 0.85:
                    category = 'expert'
                elif mastery >= 0.70:
                    category = 'proficient'
                elif mastery >= 0.40:
                    category = 'developing'
                else:
                    category = 'novice'
                
                profile[concept] = {
                    'mastery_level': mastery,
                    'confidence': record.get('confidence', 0.5),
                    'last_assessed': record.get('last_assessed'),
                    'num_assessments': record.get('num_assessments', 0),
                    'mastery_category': category
                }
            else:
                profile[concept] = {
                    'mastery_level': 0.5,
                    'confidence': 0.0,
                    'last_assessed': None,
                    'num_assessments': 0,
                    'mastery_category': 'developing'
                }
        except Exception as e:
            print(f"Error getting mastery for {concept}: {e}")
            profile[concept] = {
                'mastery_level': 0.5,
                'confidence': 0.0,
                'last_assessed': None,
                'num_assessments': 0,
                'mastery_category': 'developing'
            }
    
    return profile


def get_diagnostic_insights(self, student_id: str, concepts: List[str]) -> Dict:
    """
    Get diagnostic insights including strengths, weaknesses, and learning style
    
    Args:
        student_id: Student identifier
        concepts: List of concepts to analyze
        
    Returns:
        Dictionary with diagnostic insights
    """
    mastery_profile = self.get_mastery_profile(student_id, concepts)
    
    # Identify strengths (mastery >= 0.75)
    strengths = [
        {
            'concept': concept,
            'mastery': data['mastery_level'],
            'evidence': f"Mastery level: {data['mastery_level']:.2f} ({data['mastery_category']})"
        }
        for concept, data in mastery_profile.items()
        if data['mastery_level'] >= 0.75
    ]
    
    # Identify weaknesses (mastery < 0.60)
    weaknesses = [
        {
            'concept': concept,
            'mastery': data['mastery_level'],
            'evidence': f"Mastery level: {data['mastery_level']:.2f} ({data['mastery_category']})",
            'misconceptions': []  # Would be populated from detailed analysis
        }
        for concept, data in mastery_profile.items()
        if data['mastery_level'] < 0.60
    ]
    
    # Determine learning style (would use more sophisticated analysis in production)
    avg_mastery = sum(d['mastery_level'] for d in mastery_profile.values()) / len(mastery_profile) if mastery_profile else 0.5
    
    if avg_mastery > 0.7:
        preferred_difficulty = 'hard'
    elif avg_mastery > 0.4:
        preferred_difficulty = 'medium'
    else:
        preferred_difficulty = 'easy'
    
    return {
        'strengths': strengths,
        'weaknesses': weaknesses,
        'learning_style': 'hands-on',  # Default, would be determined from interaction patterns
        'preferred_difficulty': preferred_difficulty,
        'avg_mastery': avg_mastery
    }
```

---

## 🎯 Step 3: Extend Recommendation Service

### File: `services/recommendation/services/recommendation_service.py`

Add these methods to `RecommendationService` class:

```python
def get_content_recommendations(self, student_id: str, concepts: List[str], student_graph_obj=None) -> List[Dict]:
    """
    Get ranked content recommendations for concepts
    
    Args:
        student_id: Student identifier
        concepts: List of concepts
        student_graph_obj: Student graph object (optional)
        
    Returns:
        List of content recommendations sorted by learning gain
    """
    recommendations = []
    
    # Get mastery profile if student_graph_obj provided
    if student_graph_obj:
        mastery_profile = student_graph_obj.get_mastery_profile()
    else:
        mastery_profile = {concept: 0.5 for concept in concepts}
    
    for concept in concepts:
        mastery = mastery_profile.get(concept, 0.5)
        
        # Determine content type and difficulty based on mastery
        if mastery < 0.4:
            content_type = 'explanation'
            difficulty = 'easy'
            learning_gain = 0.20
            estimated_time = 15
        elif mastery < 0.7:
            content_type = 'exercise'
            difficulty = 'medium'
            learning_gain = 0.15
            estimated_time = 20
        else:
            content_type = 'challenge'
            difficulty = 'hard'
            learning_gain = 0.10
            estimated_time = 25
        
        # Engagement score (higher for concepts with medium mastery)
        engagement_score = 1.0 - abs(mastery - 0.5) * 2
        
        recommendations.append({
            'concept': concept,
            'content_type': content_type,
            'difficulty': difficulty,
            'estimated_time': estimated_time,
            'learning_gain': learning_gain,
            'engagement_score': engagement_score,
            'current_mastery': mastery
        })
    
    # Sort by learning gain (highest first)
    recommendations.sort(key=lambda x: x['learning_gain'], reverse=True)
    
    return recommendations


def get_hobby_personalization(self, student_id: str, concepts: List[str]) -> Dict:
    """
    Get hobby-based personalization data
    
    Args:
        student_id: Student identifier
        concepts: List of concepts
        
    Returns:
        Dictionary with personalization data
    """
    # In production, this would query student hobbies from database
    # For now, return example structure
    
    # Example hobby mappings
    hobby_examples = {
        'gaming': {
            'NoSQL Databases': 'game leaderboard storage',
            'Data Modeling': 'player inventory systems',
            'CAP Theorem': 'multiplayer game consistency',
            'Replication': 'game state synchronization'
        },
        'music': {
            'NoSQL Databases': 'music streaming catalogs',
            'Data Modeling': 'playlist organization',
            'CAP Theorem': 'distributed music libraries',
            'Replication': 'offline music sync'
        },
        'sports': {
            'NoSQL Databases': 'sports statistics tracking',
            'Data Modeling': 'team roster management',
            'CAP Theorem': 'live score updates',
            'Replication': 'multi-stadium data sync'
        }
    }
    
    # Default hobbies (would be fetched from student profile)
    hobbies = ['gaming', 'music']
    
    preferred_examples = []
    for concept in concepts:
        for hobby in hobbies:
            if concept in hobby_examples.get(hobby, {}):
                preferred_examples.append({
                    'concept': concept,
                    'example_context': hobby_examples[hobby][concept],
                    'hobby': hobby,
                    'relevance_score': 0.9
                })
                break
    
    return {
        'hobbies': hobbies,
        'interests': ['game development', 'audio processing'],
        'preferred_examples': preferred_examples
    }
```

---

## 🔧 Step 4: Create Unified Lab Generation Service

### File: `services/content_generation/services/personalized_lab_generator.py`

```python
"""
Personalized Lab Generator Service

Integrates SQKT, G-CDM+AD4CD, and RL Agent outputs to generate
personalized lab exercises based on student's learning state.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime


class PersonalizedLabGenerator:
    """
    Generates personalized lab exercises using AI model outputs
    """
    
    def __init__(self, sqkt_service, ad4cd_service, recommendation_service, lab_generator):
        """
        Initialize personalized lab generator
        
        Args:
            sqkt_service: SQKT knowledge tracing service
            ad4cd_service: AD4CD cognitive diagnosis service
            recommendation_service: RL recommendation service
            lab_generator: Base lab generator
        """
        self.sqkt = sqkt_service
        self.ad4cd = ad4cd_service
        self.recommendation = recommendation_service
        self.lab_generator = lab_generator
        
        print("✅ Personalized Lab Generator initialized")
    
    def generate_personalized_lab(
        self,
        student_id: str,
        topic_name: str,
        concepts: List[str],
        theory_data: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Generate a personalized lab for a student
        
        Args:
            student_id: Student identifier
            topic_name: Topic name
            concepts: List of concepts in the topic
            theory_data: Optional theory data
            
        Returns:
            Personalized lab dictionary
        """
        print(f"Generating personalized lab for {student_id} on {topic_name}...")
        
        # Step 1: Get Knowledge Tracing outputs
        kt_predictions = self.sqkt.get_concept_predictions(student_id, concepts)
        skill_gaps = self.sqkt.get_skill_gaps(student_id, concepts)
        trajectory = self.sqkt.get_learning_trajectory(student_id)
        
        # Step 2: Get Cognitive Diagnosis outputs
        mastery_profile = self.ad4cd.get_mastery_profile(student_id, concepts)
        anomaly_report = self.ad4cd.get_anomaly_report(student_id)
        diagnostic_insights = self.ad4cd.get_diagnostic_insights(student_id, concepts)
        
        # Step 3: Get Recommendation outputs
        content_recs = self.recommendation.get_content_recommendations(student_id, concepts)
        personalization = self.recommendation.get_hobby_personalization(student_id, concepts)
        
        # Step 4: Determine lab parameters
        lab_difficulty = self._determine_difficulty(trajectory, mastery_profile)
        prioritized_concepts = self._prioritize_concepts(skill_gaps, content_recs)
        scaffolding_level = self._determine_scaffolding(trajectory, anomaly_report)
        
        # Step 5: Generate lab structure
        lab = {
            'lab_id': f"lab_{topic_name.replace(' ', '_').lower()}_{student_id}_{datetime.now().strftime('%Y%m%d')}",
            'title': f"Personalized Lab: {topic_name}",
            'student_id': student_id,
            'topic_name': topic_name,
            'difficulty': lab_difficulty,
            'estimated_time': self._calculate_time(prioritized_concepts, trajectory),
            'personalization_score': self._calculate_personalization_score(personalization),
            'sections': [],
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'kt_mastery_avg': trajectory['overall_mastery'],
                'cd_confidence_avg': sum(m['confidence'] for m in mastery_profile.values()) / len(mastery_profile) if mastery_profile else 0.0,
                'anomaly_rate': anomaly_report.get('anomaly_rate', 0.0),
                'personalization_applied': ['hobby_contexts', 'difficulty_adjustment', 'scaffolding']
            }
        }
        
        # Step 6: Generate sections for each prioritized concept
        for i, concept_data in enumerate(prioritized_concepts[:5]):  # Limit to top 5 concepts
            concept = concept_data['concept']
            
            section = self._generate_lab_section(
                section_id=i + 1,
                concept=concept,
                mastery_data=mastery_profile.get(concept, {}),
                kt_prediction=kt_predictions.get(concept, {}),
                scaffolding_level=scaffolding_level,
                personalization=personalization,
                anomaly_rate=anomaly_report.get('anomaly_rate', 0.0)
            )
            
            lab['sections'].append(section)
        
        return lab
    
    def _determine_difficulty(self, trajectory: Dict, mastery_profile: Dict) -> str:
        """Determine overall lab difficulty"""
        avg_mastery = trajectory.get('overall_mastery', 0.5)
        learning_velocity = trajectory.get('learning_velocity', 0.0)
        
        # Adjust based on velocity
        if learning_velocity > 0.1:
            avg_mastery += 0.1  # Student improving, can handle harder
        elif learning_velocity < -0.1:
            avg_mastery -= 0.1  # Student struggling, need easier
        
        if avg_mastery >= 0.75:
            return 'hard'
        elif avg_mastery >= 0.50:
            return 'medium'
        else:
            return 'easy'
    
    def _prioritize_concepts(self, skill_gaps: List[Dict], content_recs: List[Dict]) -> List[Dict]:
        """Prioritize concepts for lab sections"""
        # Combine skill gaps and content recommendations
        concept_scores = {}
        
        for gap in skill_gaps:
            concept = gap['concept']
            concept_scores[concept] = {
                'concept': concept,
                'priority_score': gap['priority_score'],
                'gap_size': gap['gap_size'],
                'recommended_exercises': gap['recommended_exercises']
            }
        
        for rec in content_recs:
            concept = rec['concept']
            if concept in concept_scores:
                concept_scores[concept]['priority_score'] += rec['learning_gain']
            else:
                concept_scores[concept] = {
                    'concept': concept,
                    'priority_score': rec['learning_gain'],
                    'gap_size': 0.0,
                    'recommended_exercises': 1
                }
        
        # Sort by priority score
        prioritized = sorted(concept_scores.values(), key=lambda x: x['priority_score'], reverse=True)
        
        return prioritized
    
    def _determine_scaffolding(self, trajectory: Dict, anomaly_report: Dict) -> str:
        """Determine scaffolding level"""
        consistency = trajectory.get('consistency_score', 0.5)
        anomaly_rate = anomaly_report.get('anomaly_rate', 0.0)
        
        # High anomaly rate or low consistency = more scaffolding
        if anomaly_rate > 0.15 or consistency < 0.4:
            return 'high'
        elif anomaly_rate > 0.08 or consistency < 0.7:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_time(self, concepts: List[Dict], trajectory: Dict) -> int:
        """Calculate estimated time in minutes"""
        base_time_per_concept = 15
        num_concepts = min(len(concepts), 5)
        
        # Adjust based on learning velocity
        velocity = trajectory.get('learning_velocity', 0.0)
        if velocity > 0.1:
            time_multiplier = 0.8  # Faster learner
        elif velocity < -0.1:
            time_multiplier = 1.2  # Needs more time
        else:
            time_multiplier = 1.0
        
        return int(base_time_per_concept * num_concepts * time_multiplier)
    
    def _calculate_personalization_score(self, personalization: Dict) -> float:
        """Calculate how personalized the lab is"""
        num_personalized = len(personalization.get('preferred_examples', []))
        total_concepts = 5  # Assuming max 5 concepts
        
        return min(num_personalized / total_concepts, 1.0)
    
    def _generate_lab_section(
        self,
        section_id: int,
        concept: str,
        mastery_data: Dict,
        kt_prediction: Dict,
        scaffolding_level: str,
        personalization: Dict,
        anomaly_rate: float
    ) -> Dict:
        """Generate a single lab section"""
        mastery = mastery_data.get('mastery_level', 0.5)
        category = mastery_data.get('mastery_category', 'developing')
        
        # Get personalized context
        personalized_context = None
        for example in personalization.get('preferred_examples', []):
            if example['concept'] == concept:
                personalized_context = example['example_context']
                break
        
        # Determine section difficulty
        if mastery >= 0.75:
            section_difficulty = 'easy'  # Review only
            exercise_type = 'challenge'
        elif mastery >= 0.50:
            section_difficulty = 'medium'
            exercise_type = 'standard'
        else:
            section_difficulty = 'hard'  # Needs learning
            exercise_type = 'guided'
        
        # Determine time based on mastery
        if mastery >= 0.75:
            estimated_time = 10
        elif mastery >= 0.50:
            estimated_time = 15
        else:
            estimated_time = 20
        
        section = {
            'section_id': section_id,
            'concept': concept,
            'title': f"{concept}" + (f" in {personalized_context}" if personalized_context else ""),
            'difficulty': section_difficulty,
            'scaffolding_level': scaffolding_level,
            'estimated_time': estimated_time,
            'mastery_level': mastery,
            'mastery_category': category,
            
            'introduction': {
                'text': f"In this section, you'll work with {concept}" + 
                       (f" in the context of {personalized_context}." if personalized_context else "."),
                'personalized': personalized_context is not None,
                'hobby_context': personalized_context
            },
            
            'exercises': self._generate_exercises(
                concept=concept,
                exercise_type=exercise_type,
                scaffolding_level=scaffolding_level,
                anomaly_rate=anomaly_rate,
                personalized_context=personalized_context
            )
        }
        
        return section
    
    def _generate_exercises(
        self,
        concept: str,
        exercise_type: str,
        scaffolding_level: str,
        anomaly_rate: float,
        personalized_context: Optional[str]
    ) -> List[Dict]:
        """Generate exercises for a section"""
        exercises = []
        
        # Determine number of hints based on scaffolding
        if scaffolding_level == 'high':
            num_hints = 3
        elif scaffolding_level == 'medium':
            num_hints = 2
        else:
            num_hints = 0
        
        # Add verification questions if anomaly rate is high
        verification_questions = []
        if anomaly_rate > 0.15:
            verification_questions = [
                f"Explain your approach to solving this {concept} problem",
                f"What trade-offs did you consider?"
            ]
        
        exercise = {
            'exercise_id': 1,
            'type': exercise_type,
            'prompt': f"Apply {concept}" + (f" to {personalized_context}" if personalized_context else ""),
            'hints': [f"Hint {i+1} for {concept}" for i in range(num_hints)],
            'verification_questions': verification_questions,
            'personalized': personalized_context is not None
        }
        
        if exercise_type == 'challenge':
            exercise['extension'] = f"Advanced challenge: Optimize your {concept} solution"
        
        exercises.append(exercise)
        
        return exercises
```

---

**Status**: ✅ Implementation Guide Complete  
**Next Steps**: 
1. Add these methods to respective service files
2. Create PersonalizedLabGenerator service
3. Integrate with lab generation routes
4. Test with real student data

