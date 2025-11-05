# Personalized Lab Generation - Inputs & Outputs Summary

**Version**: 1.0  
**Date**: November 4, 2025  
**Purpose**: Consolidated reference for all inputs and outputs in personalized lab generation

---

## 📥 INPUTS

### 1. Student Context (Required)
```python
{
    'student_id': 'student_123',
    'topic_name': 'NoSQL Databases',
    'concepts': ['CAP Theorem', 'Data Modeling', 'Replication', 'Sharding'],
    'hobbies': ['gaming', 'music'],
    'interests': ['game development', 'audio processing']
}
```

### 2. Knowledge Tracing Outputs (SQKT + MLFBK)

#### A. Concept Predictions
```python
{
    'CAP Theorem': {
        'current_mastery': 0.45,      # 0-1 scale
        'predicted_next': 0.48,       # Expected after next interaction
        'confidence': 0.78,           # How confident (0-1)
        'trend': 'stable'             # improving, stable, declining
    },
    'Data Modeling': {
        'current_mastery': 0.89,
        'predicted_next': 0.91,
        'confidence': 0.92,
        'trend': 'improving'
    }
}
```

#### B. Skill Gaps
```python
[
    {
        'concept': 'CAP Theorem',
        'current_mastery': 0.45,
        'target_mastery': 0.70,
        'gap_size': 0.25,
        'priority': 'high',           # high, medium, low
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
```

#### C. Learning Trajectory
```python
{
    'num_interactions': 45,
    'recent_performance': [0.6, 0.7, 0.65, 0.75, 0.8],  # Last 5
    'overall_mastery': 0.68,
    'learning_velocity': 0.12,      # Rate of improvement
    'consistency_score': 0.85       # How consistent (0-1)
}
```

### 3. Cognitive Diagnosis Outputs (G-CDM + AD4CD)

#### A. Mastery Profile
```python
{
    'CAP Theorem': {
        'mastery_level': 0.45,
        'confidence': 0.75,
        'last_assessed': '2025-11-04T09:20:00',
        'num_assessments': 3,
        'mastery_category': 'developing'  # novice, developing, proficient, expert
    },
    'Data Modeling': {
        'mastery_level': 0.89,
        'confidence': 0.92,
        'last_assessed': '2025-11-04T09:15:00',
        'num_assessments': 5,
        'mastery_category': 'expert'
    }
}
```

#### B. Anomaly Report
```python
{
    'total_responses': 45,
    'anomalous_responses': 3,
    'anomaly_rate': 0.067,          # 6.7%
    'anomaly_types': {
        'cheating_incidents': 1,
        'guessing_incidents': 2,
        'careless_mistakes': 0
    },
    'reliability_score': 0.93       # How reliable is data
}
```

#### C. Diagnostic Insights
```python
{
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
            'misconceptions': ['Thinks all three properties achievable']
        }
    ],
    'learning_style': 'hands-on',   # visual, hands-on, theoretical
    'preferred_difficulty': 'medium'
}
```

### 4. Recommendation Outputs (RL Agent)

#### A. Content Recommendations
```python
[
    {
        'concept': 'CAP Theorem',
        'content_type': 'exercise',     # exercise, explanation, hint, review
        'difficulty': 'medium',
        'estimated_time': 15,           # minutes
        'learning_gain': 0.18,          # Expected improvement
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
```

#### B. Hobby Personalization
```python
{
    'hobbies': ['gaming', 'music'],
    'interests': ['game development', 'audio processing'],
    'preferred_examples': [
        {
            'concept': 'CAP Theorem',
            'example_context': 'game leaderboard consistency',
            'hobby': 'gaming',
            'relevance_score': 0.95
        },
        {
            'concept': 'Data Modeling',
            'example_context': 'music playlist organization',
            'hobby': 'music',
            'relevance_score': 0.88
        }
    ]
}
```

---

## 📤 OUTPUTS

### Generated Personalized Lab Structure

```python
{
    # Lab Metadata
    'lab_id': 'lab_nosql_student123_20251104',
    'title': 'Personalized Lab: NoSQL Databases',
    'student_id': 'student_123',
    'topic_name': 'NoSQL Databases',
    'difficulty': 'medium',                    # easy, medium, hard
    'estimated_time': 45,                      # minutes
    'personalization_score': 0.92,             # 0-1, how personalized
    
    # Lab Sections (one per concept)
    'sections': [
        {
            'section_id': 1,
            'concept': 'CAP Theorem',
            'title': 'CAP Theorem in Game Leaderboard Consistency',
            'difficulty': 'medium',
            'scaffolding_level': 'high',       # high, medium, low
            'estimated_time': 20,              # minutes
            'mastery_level': 0.45,
            'mastery_category': 'developing',
            
            # Introduction
            'introduction': {
                'text': 'In online gaming, leaderboards must handle millions of players...',
                'personalized': True,
                'hobby_context': 'gaming'
            },
            
            # Exercises
            'exercises': [
                {
                    'exercise_id': 1,
                    'type': 'guided',          # tutorial, guided, standard, challenge
                    'prompt': 'Design a leaderboard system that prioritizes availability during network partitions',
                    'hints': [
                        'Consider what happens during network partition',
                        'Think about consistency vs. availability trade-off',
                        'Review CAP theorem principles'
                    ],
                    'verification_questions': [],  # Empty if anomaly rate low
                    'personalized': True
                }
            ]
        },
        
        {
            'section_id': 2,
            'concept': 'Data Modeling',
            'title': 'Data Modeling for Player Inventory Systems',
            'difficulty': 'easy',              # Review only (high mastery)
            'scaffolding_level': 'low',
            'estimated_time': 10,
            'mastery_level': 0.89,
            'mastery_category': 'expert',
            
            'introduction': {
                'text': 'Apply your data modeling expertise to player inventory systems...',
                'personalized': True,
                'hobby_context': 'gaming'
            },
            
            'exercises': [
                {
                    'exercise_id': 1,
                    'type': 'challenge',       # No hints for expert
                    'prompt': 'Optimize a player inventory data model for 1M+ items',
                    'hints': [],               # Expert level = no hints
                    'verification_questions': [],
                    'extension': 'Add support for item trading between players',
                    'personalized': True
                }
            ]
        },
        
        {
            'section_id': 3,
            'concept': 'Replication',
            'title': 'Replication in Game State Synchronization',
            'difficulty': 'medium',
            'scaffolding_level': 'medium',
            'estimated_time': 15,
            'mastery_level': 0.52,
            'mastery_category': 'developing',
            
            'introduction': {
                'text': 'Multiplayer games require real-time state synchronization...',
                'personalized': True,
                'hobby_context': 'gaming'
            },
            
            'exercises': [
                {
                    'exercise_id': 1,
                    'type': 'standard',
                    'prompt': 'Implement replication for multiplayer game state',
                    'hints': [
                        'Consider master-slave vs. multi-master',
                        'Think about conflict resolution'
                    ],
                    'verification_questions': [],
                    'personalized': True
                }
            ]
        }
    ],
    
    # Metadata
    'metadata': {
        'generated_at': '2025-11-04T10:30:00',
        'kt_mastery_avg': 0.68,
        'cd_confidence_avg': 0.82,
        'anomaly_rate': 0.067,
        'personalization_applied': [
            'hobby_contexts',
            'difficulty_adjustment',
            'scaffolding'
        ]
    }
}
```

---

## 🔄 INPUT → OUTPUT MAPPING

### Mastery Level → Exercise Type & Hints

| Mastery | Category | Exercise Type | Hints | Time | Difficulty |
|---------|----------|---------------|-------|------|------------|
| **0.85+** | Expert | Challenge | 0 | 10 min | Easy (review) |
| **0.70-0.85** | Proficient | Standard | 1 | 15 min | Easy |
| **0.40-0.70** | Developing | Guided | 2 | 20 min | Medium |
| **<0.40** | Novice | Tutorial | 3 | 25 min | Hard |

### Anomaly Rate → Verification

| Anomaly Rate | Action |
|--------------|--------|
| **>0.15** | Add 2-3 verification questions requiring explanations |
| **0.08-0.15** | Add 1 verification question |
| **<0.08** | No verification needed |

### Learning Velocity → Difficulty Adjustment

| Velocity | Adjustment |
|----------|------------|
| **>0.10** | Increase difficulty by 1 level |
| **-0.10 to 0.10** | Keep current difficulty |
| **<-0.10** | Decrease difficulty by 1 level |

### Skill Gap → Priority & Exercises

| Gap Size | Priority | Exercises | Time Allocation |
|----------|----------|-----------|-----------------|
| **>0.30** | High | 3 | 25 min |
| **0.15-0.30** | Medium | 2 | 20 min |
| **<0.15** | Low | 1 | 15 min |

---

## 🎯 QUICK REFERENCE: API METHODS

### Required Service Methods

```python
# Knowledge Tracing Service (SQKT)
sqkt_service.get_concept_predictions(student_id, concepts)
sqkt_service.get_skill_gaps(student_id, concepts, target_mastery=0.7)
sqkt_service.get_learning_trajectory(student_id)

# Cognitive Diagnosis Service (AD4CD)
ad4cd_service.get_mastery_profile(student_id, concepts)
ad4cd_service.get_anomaly_report(student_id)
ad4cd_service.get_diagnostic_insights(student_id, concepts)

# Recommendation Service (RL Agent)
recommendation_service.get_content_recommendations(student_id, concepts)
recommendation_service.get_hobby_personalization(student_id, concepts)
```

### Lab Generation Call

```python
# Initialize generator
lab_generator = PersonalizedLabGenerator(
    sqkt_service=sqkt_service,
    ad4cd_service=ad4cd_service,
    recommendation_service=recommendation_service,
    lab_generator=base_lab_generator
)

# Generate personalized lab
lab = lab_generator.generate_personalized_lab(
    student_id='student_123',
    topic_name='NoSQL Databases',
    concepts=['CAP Theorem', 'Data Modeling', 'Replication', 'Sharding'],
    theory_data=None  # Optional
)
```

---

## 📊 EXAMPLE: Complete Flow

### Input
```python
student_id = 'student_123'
topic = 'NoSQL Databases'
concepts = ['CAP Theorem', 'Data Modeling', 'Replication']
hobbies = ['gaming']
```

### AI Model Outputs
```python
# SQKT
mastery = {'CAP Theorem': 0.45, 'Data Modeling': 0.89, 'Replication': 0.52}
velocity = 0.12  # Improving
gaps = [{'concept': 'CAP Theorem', 'gap_size': 0.25, 'priority': 'high'}]

# G-CDM+AD4CD
categories = {'CAP Theorem': 'developing', 'Data Modeling': 'expert'}
anomaly_rate = 0.067  # Low

# RL Agent
recommendations = [{'concept': 'CAP Theorem', 'learning_gain': 0.25}]
personalization = {'CAP Theorem': 'game leaderboards'}
```

### Output Lab
```python
{
    'title': 'Personalized Lab: NoSQL Databases',
    'difficulty': 'medium',
    'estimated_time': 45,
    'sections': [
        {
            'concept': 'CAP Theorem',
            'title': 'CAP Theorem in Game Leaderboards',
            'difficulty': 'medium',
            'scaffolding_level': 'high',
            'exercises': [{'type': 'guided', 'hints': 3}]
        },
        {
            'concept': 'Data Modeling',
            'title': 'Data Modeling for Player Inventory',
            'difficulty': 'easy',
            'scaffolding_level': 'low',
            'exercises': [{'type': 'challenge', 'hints': 0}]
        }
    ]
}
```

---

**Status**: ✅ Complete Reference  
**Usage**: Use this as quick reference for implementing personalized lab generation  
**Related Docs**: 
- `PERSONALIZED_LAB_GENERATION_REQUIREMENTS.md` - Detailed requirements
- `LAB_GENERATION_IMPLEMENTATION_GUIDE.md` - Implementation guide
- `LAB_GENERATION_ARCHITECTURE_DIAGRAM.md` - Architecture diagrams

