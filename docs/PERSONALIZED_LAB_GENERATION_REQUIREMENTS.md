# Personalized Lab Generation Requirements
## AI Model Outputs for Concept-Based Lab Personalization

**Version**: 1.0  
**Date**: November 4, 2025  
**Purpose**: Define required outputs from Knowledge Tracing, Cognitive Diagnosis, and Recommendation models for generating personalized lab exercises

---

## 🎯 Overview

Personalized lab generation requires integrating outputs from three AI models:
1. **SQKT + MLFBK** (Knowledge Tracing) - Student learning trajectory
2. **G-CDM + AD4CD** (Cognitive Diagnosis) - Concept mastery & anomaly detection
3. **RL Recommendation Agent** - Adaptive content selection

Each model provides specific outputs that inform lab difficulty, content focus, scaffolding level, and personalization based on student hobbies/interests.

---

## 📊 1. Knowledge Tracing Model Outputs (SQKT + MLFBK)

### Required Outputs for Lab Generation

#### A. **Knowledge State Vector**
```python
{
    'student_id': 'student_123',
    'knowledge_state': [0.72, 0.45, 0.89, 0.34, ...],  # Hidden state vector (dim: 128)
    'timestamp': '2025-11-04T10:30:00'
}
```
**Purpose**: Represents student's current understanding across all concepts
**Usage in Lab**: 
- Determines overall lab difficulty
- Identifies which concepts need more practice
- Guides scaffolding level (high state = less hints, low state = more guidance)

#### B. **Concept-Level Mastery Predictions**
```python
{
    'student_id': 'student_123',
    'concept_predictions': {
        'NoSQL Basics': {
            'current_mastery': 0.72,
            'predicted_next': 0.75,
            'confidence': 0.85,
            'trend': 'improving'  # improving, stable, declining
        },
        'CAP Theorem': {
            'current_mastery': 0.45,
            'predicted_next': 0.48,
            'confidence': 0.78,
            'trend': 'stable'
        },
        'Data Modeling': {
            'current_mastery': 0.89,
            'predicted_next': 0.91,
            'confidence': 0.92,
            'trend': 'improving'
        }
    }
}
```
**Purpose**: Tracks mastery for each concept in the topic
**Usage in Lab**:
- **High mastery (>0.7)**: Advanced exercises, minimal hints
- **Medium mastery (0.4-0.7)**: Standard exercises, moderate scaffolding
- **Low mastery (<0.4)**: Foundational exercises, extensive guidance

#### C. **Learning Trajectory**
```python
{
    'student_id': 'student_123',
    'trajectory': {
        'num_interactions': 45,
        'recent_performance': [0.6, 0.7, 0.65, 0.75, 0.8],  # Last 5 interactions
        'overall_mastery': 0.68,
        'learning_velocity': 0.12,  # Rate of improvement per interaction
        'consistency_score': 0.85   # How consistent performance is
    }
}
```
**Purpose**: Shows learning progress over time
**Usage in Lab**:
- **High velocity**: Challenge with harder problems
- **Low velocity**: Provide more foundational practice
- **Low consistency**: Add review sections before new concepts

#### D. **Skill Gap Analysis**
```python
{
    'student_id': 'student_123',
    'skill_gaps': [
        {
            'concept': 'CAP Theorem',
            'current_mastery': 0.45,
            'target_mastery': 0.70,
            'gap_size': 0.25,
            'priority': 'high',
            'recommended_exercises': 3
        },
        {
            'concept': 'Replication',
            'current_mastery': 0.52,
            'target_mastery': 0.70,
            'gap_size': 0.18,
            'priority': 'medium',
            'recommended_exercises': 2
        }
    ]
}
```
**Purpose**: Identifies specific weaknesses
**Usage in Lab**:
- Prioritize exercises for high-priority gaps
- Allocate more lab steps to weak concepts
- Skip or minimize exercises for mastered concepts

---

## 🧠 2. Cognitive Diagnosis Model Outputs (G-CDM + AD4CD)

### Required Outputs for Lab Generation

#### A. **Concept Mastery Profile**
```python
{
    'student_id': 'student_123',
    'mastery_profile': {
        'NoSQL Basics': {
            'mastery_level': 0.72,
            'confidence': 0.88,
            'last_assessed': '2025-11-04T09:15:00',
            'num_assessments': 5,
            'mastery_category': 'proficient'  # novice, developing, proficient, expert
        },
        'CAP Theorem': {
            'mastery_level': 0.45,
            'confidence': 0.75,
            'last_assessed': '2025-11-04T09:20:00',
            'num_assessments': 3,
            'mastery_category': 'developing'
        }
    }
}
```
**Purpose**: Fine-grained assessment of concept understanding
**Usage in Lab**:
- **Expert (>0.85)**: Skip basic exercises, provide extension challenges
- **Proficient (0.70-0.85)**: Standard exercises with optional challenges
- **Developing (0.40-0.70)**: Guided exercises with scaffolding
- **Novice (<0.40)**: Step-by-step tutorials with extensive hints

#### B. **Anomaly Detection Results**
```python
{
    'student_id': 'student_123',
    'anomaly_report': {
        'total_responses': 45,
        'anomalous_responses': 3,
        'anomaly_rate': 0.067,
        'anomaly_types': {
            'cheating_incidents': 1,      # Correct answer without understanding
            'guessing_incidents': 2,      # Random correct answers
            'careless_mistakes': 0        # Wrong despite high mastery
        },
        'avg_anomaly_score': 0.15,
        'reliability_score': 0.93  # How reliable is student's performance data
    }
}
```
**Purpose**: Identifies unreliable performance data
**Usage in Lab**:
- **High anomaly rate (>0.15)**: Add verification questions, require explanations
- **Cheating detected**: Include conceptual questions, not just code
- **Guessing detected**: Add multiple checkpoints, require reasoning
- **Low reliability**: Don't trust mastery scores, use conservative difficulty

#### C. **Diagnostic Insights**
```python
{
    'student_id': 'student_123',
    'diagnostic_insights': {
        'strengths': [
            {
                'concept': 'Data Modeling',
                'mastery': 0.89,
                'evidence': 'Consistently correct on complex problems'
            }
        ],
        'weaknesses': [
            {
                'concept': 'CAP Theorem',
                'mastery': 0.45,
                'evidence': 'Struggles with trade-off scenarios',
                'misconceptions': ['Thinks all three properties can be achieved simultaneously']
            }
        ],
        'learning_style': 'visual',  # visual, hands-on, theoretical
        'preferred_difficulty': 'medium'
    }
}
```
**Purpose**: Provides actionable insights about student learning
**Usage in Lab**:
- **Strengths**: Build on these in advanced sections
- **Weaknesses**: Target with focused exercises
- **Misconceptions**: Include exercises that directly address them
- **Learning style**: Adapt lab format (diagrams for visual, code for hands-on)

#### D. **Confidence-Weighted Performance**
```python
{
    'student_id': 'student_123',
    'confidence_weighted_performance': {
        'NoSQL Basics': {
            'raw_performance': 0.75,
            'confidence': 0.88,
            'weighted_performance': 0.66,  # raw * confidence
            'reliability': 'high'
        },
        'CAP Theorem': {
            'raw_performance': 0.60,
            'confidence': 0.45,
            'weighted_performance': 0.27,  # Low confidence = unreliable
            'reliability': 'low'
        }
    }
}
```
**Purpose**: Adjusts performance by confidence in assessment
**Usage in Lab**:
- **High confidence**: Trust the mastery level
- **Low confidence**: Be conservative, provide more support
- **Weighted performance**: Use this for difficulty selection, not raw scores

---

## 🎯 3. Recommendation Model Outputs (RL Agent)

### Required Outputs for Lab Generation

#### A. **Next Best Action**
```python
{
    'student_id': 'student_123',
    'recommendation': {
        'action_type': 'exercise',  # exercise, explanation, hint, review
        'target_concept': 'CAP Theorem',
        'difficulty': 'medium',
        'priority': 0.85,  # 0-1, how important this action is
        'expected_reward': 0.23,  # Expected learning gain
        'confidence': 0.78
    }
}
```
**Purpose**: Suggests optimal next learning activity
**Usage in Lab**:
- **Action type = exercise**: Include hands-on coding task
- **Action type = explanation**: Add theory section before exercise
- **Action type = hint**: Provide scaffolding hints
- **Action type = review**: Include review questions before new content

#### B. **Personalized Content Recommendations**
```python
{
    'student_id': 'student_123',
    'content_recommendations': [
        {
            'concept': 'CAP Theorem',
            'content_type': 'exercise',
            'difficulty': 'medium',
            'estimated_time': 15,  # minutes
            'learning_gain': 0.18,
            'engagement_score': 0.82
        },
        {
            'concept': 'Replication',
            'content_type': 'explanation',
            'difficulty': 'easy',
            'estimated_time': 10,
            'learning_gain': 0.12,
            'engagement_score': 0.75
        }
    ]
}
```
**Purpose**: Ranks content by expected learning value
**Usage in Lab**:
- Order lab sections by learning_gain (highest first)
- Allocate time based on estimated_time
- Include high engagement_score activities to maintain motivation

#### C. **Adaptive Difficulty Adjustment**
```python
{
    'student_id': 'student_123',
    'difficulty_adjustment': {
        'current_difficulty': 'medium',
        'recommended_difficulty': 'medium-hard',
        'adjustment_reason': 'Student showing consistent improvement',
        'confidence_in_adjustment': 0.82,
        'gradual_increase': True  # vs. sudden jump
    }
}
```
**Purpose**: Suggests when to increase/decrease difficulty
**Usage in Lab**:
- **Gradual increase**: Add one harder problem at the end
- **Sudden jump**: Change entire lab difficulty level
- **Decrease**: Provide more scaffolding, simpler problems

#### D. **Hobby-Based Personalization**
```python
{
    'student_id': 'student_123',
    'personalization': {
        'hobbies': ['gaming', 'music'],
        'interests': ['game development', 'audio processing'],
        'preferred_examples': [
            {
                'concept': 'NoSQL Databases',
                'example_context': 'game leaderboard storage',
                'relevance_score': 0.95
            },
            {
                'concept': 'Data Modeling',
                'example_context': 'music playlist organization',
                'relevance_score': 0.88
            }
        ]
    }
}
```
**Purpose**: Personalizes examples based on student interests
**Usage in Lab**:
- Replace generic examples with hobby-related ones
- Frame problems in contexts student cares about
- Increase engagement and motivation

---

## 🔧 4. Integration: How to Use These Outputs in Lab Generation

### Lab Generation Pipeline

```python
def generate_personalized_lab(student_id, topic_name, concepts):
    """
    Generate personalized lab using all three AI model outputs
    """
    
    # Step 1: Get Knowledge Tracing outputs
    kt_outputs = sqkt_service.get_knowledge_state(student_id)
    skill_gaps = sqkt_service.get_skill_gaps(student_id, concepts)
    trajectory = sqkt_service.get_learning_trajectory(student_id)
    
    # Step 2: Get Cognitive Diagnosis outputs
    cd_outputs = ad4cd_service.get_mastery_profile(student_id, concepts)
    anomaly_report = ad4cd_service.get_anomaly_report(student_id)
    diagnostic_insights = ad4cd_service.get_diagnostic_insights(student_id)
    
    # Step 3: Get Recommendation outputs
    rec_outputs = recommendation_service.get_recommendation(student_id)
    content_recs = recommendation_service.get_content_recommendations(student_id, concepts)
    personalization = recommendation_service.get_hobby_personalization(student_id)
    
    # Step 4: Determine lab parameters
    lab_difficulty = determine_difficulty(kt_outputs, cd_outputs, rec_outputs)
    lab_concepts = prioritize_concepts(skill_gaps, cd_outputs, content_recs)
    scaffolding_level = determine_scaffolding(trajectory, anomaly_report)
    example_contexts = get_personalized_contexts(personalization, lab_concepts)
    
    # Step 5: Generate lab structure
    lab = {
        'title': f"Personalized Lab: {topic_name}",
        'difficulty': lab_difficulty,
        'estimated_time': calculate_time(lab_concepts, trajectory),
        'sections': []
    }
    
    # Step 6: Add sections based on recommendations
    for concept in lab_concepts:
        section = generate_lab_section(
            concept=concept,
            mastery=cd_outputs['mastery_profile'][concept]['mastery_level'],
            scaffolding=scaffolding_level,
            example_context=example_contexts.get(concept),
            recommended_action=rec_outputs.get('action_type')
        )
        lab['sections'].append(section)
    
    return lab
```

### Example Output Structure

```python
{
    'lab_id': 'lab_nosql_student123_20251104',
    'title': 'Personalized Lab: NoSQL Databases',
    'student_id': 'student_123',
    'difficulty': 'medium',
    'estimated_time': 45,  # minutes
    'personalization_score': 0.92,
    
    'sections': [
        {
            'section_id': 1,
            'concept': 'CAP Theorem',
            'title': 'Understanding CAP Trade-offs in Game Leaderboards',
            'difficulty': 'medium',
            'scaffolding_level': 'high',  # Due to low mastery
            'estimated_time': 15,
            
            'introduction': {
                'text': 'In online gaming, leaderboards must handle...',
                'personalized': True,
                'hobby_context': 'gaming'
            },
            
            'exercises': [
                {
                    'exercise_id': 1,
                    'type': 'guided',  # Due to low mastery
                    'prompt': 'Design a leaderboard system that prioritizes...',
                    'hints': [
                        'Consider what happens during network partition',
                        'Think about consistency vs. availability trade-off'
                    ],
                    'verification_questions': [  # Due to anomaly detection
                        'Explain why you chose this approach',
                        'What trade-offs did you make?'
                    ]
                }
            ]
        },
        
        {
            'section_id': 2,
            'concept': 'Data Modeling',
            'title': 'Modeling Music Playlists in NoSQL',
            'difficulty': 'easy',  # High mastery, so review only
            'scaffolding_level': 'low',
            'estimated_time': 10,
            
            'exercises': [
                {
                    'exercise_id': 1,
                    'type': 'challenge',  # High mastery = challenge
                    'prompt': 'Optimize a music playlist data model for...',
                    'hints': [],  # No hints for high mastery
                    'extension': 'Add support for collaborative playlists'
                }
            ]
        }
    ],
    
    'metadata': {
        'generated_at': '2025-11-04T10:30:00',
        'kt_mastery_avg': 0.68,
        'cd_confidence_avg': 0.82,
        'anomaly_rate': 0.067,
        'personalization_applied': ['hobby_contexts', 'difficulty_adjustment', 'scaffolding']
    }
}
```

---

## 📋 Summary: Required API Methods

### Knowledge Tracing Service
```python
sqkt_service.get_knowledge_state(student_id) → knowledge_state_vector
sqkt_service.get_concept_predictions(student_id, concepts) → mastery_predictions
sqkt_service.get_learning_trajectory(student_id) → trajectory_data
sqkt_service.get_skill_gaps(student_id, concepts) → gap_analysis
```

### Cognitive Diagnosis Service
```python
ad4cd_service.get_mastery_profile(student_id, concepts) → mastery_profile
ad4cd_service.get_anomaly_report(student_id) → anomaly_data
ad4cd_service.get_diagnostic_insights(student_id) → insights
ad4cd_service.get_confidence_weighted_performance(student_id) → weighted_scores
```

### Recommendation Service
```python
recommendation_service.get_recommendation(student_id) → next_action
recommendation_service.get_content_recommendations(student_id, concepts) → ranked_content
recommendation_service.get_difficulty_adjustment(student_id) → difficulty_params
recommendation_service.get_hobby_personalization(student_id) → personalization_data
```

---

**Status**: ✅ Ready for Implementation  
**Next Steps**: Implement these methods in respective services and integrate into lab generator

