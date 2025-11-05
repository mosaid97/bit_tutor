#!/usr/bin/env python3
"""
Create Demo Students: Roma and Moha

This script creates two demo students with unique portfolios, different strengths/weaknesses,
and assessment results. It also verifies the knowledge graph pipelines are connected properly.
"""

import sys
import os
import random
import hashlib
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.knowledge_graph.services.dynamic_graph_manager import DynamicGraphManager
from services.content_generation.services.question_generator import QuestionGenerator


def hash_password(password):
    """Hash password using SHA-256 (same as StudentAuthService)"""
    return hashlib.sha256(password.encode()).hexdigest()


def create_demo_students():
    """Create two demo students with unique portfolios."""
    
    print("\n" + "="*80)
    print("🎓 CREATING DEMO STUDENTS: ROMA & MOHA")
    print("="*80 + "\n")
    
    # Initialize services
    print("📊 Initializing services...")
    graph_manager = DynamicGraphManager()
    question_generator = QuestionGenerator(use_llm=False)
    
    if not graph_manager.neo4j or not graph_manager.neo4j.graph:
        print("❌ Error: Neo4j connection not available")
        return False
    
    print("✅ Services initialized\n")
    
    # Get the Big Data Analysis class
    class_query = """
    MATCH (c:Class {name: 'Big Data Analysis'})
    RETURN c.class_id as class_id, c.name as name
    LIMIT 1
    """
    class_result = graph_manager.neo4j.graph.query(class_query)
    
    if not class_result:
        print("❌ Error: Big Data Analysis class not found")
        return False
    
    class_id = class_result[0]['class_id']
    class_name = class_result[0]['name']
    print(f"✅ Found class: {class_name} (ID: {class_id})\n")
    
    # Get all topics and concepts
    topics_query = """
    MATCH (c:Class {class_id: $class_id})-[:INCLUDES]->(t:Topic)
    OPTIONAL MATCH (t)-[:INCLUDES_CONCEPT]->(concept:Concept)
    WITH t, collect(DISTINCT concept.name) as concepts
    RETURN t.name as topic_name, t.topic_id as topic_id, concepts
    ORDER BY t.order
    """
    topics_result = graph_manager.neo4j.graph.query(topics_query, {'class_id': class_id})
    
    if not topics_result:
        print("❌ Error: No topics found for class")
        return False
    
    topics = [dict(t) for t in topics_result]
    print(f"✅ Found {len(topics)} topics with concepts\n")
    
    # Create Student 1: Roma
    print("👤 Creating Student 1: ROMA")
    print("-" * 80)
    
    roma_data = {
        'student_id': 'student_roma',
        'name': 'Roma',
        'email': 'roma@example.com',
        'password_hash': hash_password('roma123'),  # Properly hashed password
        'hobbies': ['gaming', 'technology', 'music'],
        'interests': ['artificial intelligence', 'data science', 'machine learning'],
        'registration_date': (datetime.now() - timedelta(days=30)).isoformat(),
        'grades': [],
        'overall_score': 0.0,
        'streak_days': 15,
        'total_practice_hours': 45.5,
        'class_assessment_completed': True,
        'class_assessment_score': 78.5,
        'class_assessment_date': (datetime.now() - timedelta(days=28)).isoformat(),
        'status': 'active'
    }
    
    # Create Roma in Neo4j
    create_roma_query = """
    MERGE (s:Student {student_id: $student_id})
    SET s.name = $name,
        s.email = $email,
        s.password_hash = $password_hash,
        s.hobbies = $hobbies,
        s.interests = $interests,
        s.registration_date = $registration_date,
        s.grades = $grades,
        s.overall_score = $overall_score,
        s.streak_days = $streak_days,
        s.total_practice_hours = $total_practice_hours,
        s.class_assessment_completed = $class_assessment_completed,
        s.class_assessment_score = $class_assessment_score,
        s.class_assessment_date = $class_assessment_date,
        s.status = $status
    WITH s
    MATCH (c:Class {class_id: $class_id})
    MERGE (s)-[:REGISTERED_IN]->(c)
    RETURN s.student_id as student_id
    """
    
    graph_manager.neo4j.graph.query(create_roma_query, {**roma_data, 'class_id': class_id})
    print(f"✅ Created student: {roma_data['name']} ({roma_data['student_id']})")
    print(f"   Email: {roma_data['email']}")
    print(f"   Hobbies: {', '.join(roma_data['hobbies'])}")
    print(f"   Interests: {', '.join(roma_data['interests'])}")
    print(f"   Class Assessment Score: {roma_data['class_assessment_score']}%\n")
    
    # Create Student 2: Moha
    print("👤 Creating Student 2: MOHA")
    print("-" * 80)
    
    moha_data = {
        'student_id': 'student_moha',
        'name': 'Moha',
        'email': 'moha@example.com',
        'password_hash': hash_password('moha123'),  # Properly hashed password
        'hobbies': ['sports', 'reading', 'photography'],
        'interests': ['big data', 'cloud computing', 'database systems'],
        'registration_date': (datetime.now() - timedelta(days=25)).isoformat(),
        'grades': [],
        'overall_score': 0.0,
        'streak_days': 12,
        'total_practice_hours': 38.0,
        'class_assessment_completed': True,
        'class_assessment_score': 82.0,
        'class_assessment_date': (datetime.now() - timedelta(days=23)).isoformat(),
        'status': 'active'
    }
    
    # Create Moha in Neo4j
    create_moha_query = """
    MERGE (s:Student {student_id: $student_id})
    SET s.name = $name,
        s.email = $email,
        s.password_hash = $password_hash,
        s.hobbies = $hobbies,
        s.interests = $interests,
        s.registration_date = $registration_date,
        s.grades = $grades,
        s.overall_score = $overall_score,
        s.streak_days = $streak_days,
        s.total_practice_hours = $total_practice_hours,
        s.class_assessment_completed = $class_assessment_completed,
        s.class_assessment_score = $class_assessment_score,
        s.class_assessment_date = $class_assessment_date,
        s.status = $status
    WITH s
    MATCH (c:Class {class_id: $class_id})
    MERGE (s)-[:REGISTERED_IN]->(c)
    RETURN s.student_id as student_id
    """
    
    graph_manager.neo4j.graph.query(create_moha_query, {**moha_data, 'class_id': class_id})
    print(f"✅ Created student: {moha_data['name']} ({moha_data['student_id']})")
    print(f"   Email: {moha_data['email']}")
    print(f"   Hobbies: {', '.join(moha_data['hobbies'])}")
    print(f"   Interests: {', '.join(moha_data['interests'])}")
    print(f"   Class Assessment Score: {moha_data['class_assessment_score']}%\n")
    
    # Create unique mastery profiles for each student
    print("📊 Creating unique mastery profiles...")
    print("-" * 80)
    
    # Roma's strengths: NoSQL concepts, weak in algorithms
    # Moha's strengths: Algorithms, weak in NoSQL theory
    
    all_concepts = []
    for topic in topics:
        for concept in topic['concepts']:
            if concept:  # Skip empty concepts
                all_concepts.append({
                    'topic': topic['topic_name'],
                    'concept': concept
                })
    
    print(f"Found {len(all_concepts)} concepts across all topics\n")
    
    # Create mastery for Roma (strong in NoSQL, weak in algorithms)
    roma_mastery_count = 0
    for item in all_concepts:
        concept = item['concept']
        topic = item['topic']
        
        # Determine mastery based on topic
        if 'NoSQL' in topic or 'Database' in topic:
            mastery = random.uniform(0.75, 0.95)  # Strong
        elif 'Algorithm' in topic:
            mastery = random.uniform(0.35, 0.55)  # Weak
        else:
            mastery = random.uniform(0.60, 0.80)  # Average
        
        # Create KNOWS relationship
        knows_query = """
        MATCH (s:Student {student_id: $student_id})
        MATCH (c:Concept {name: $concept_name})
        MERGE (s)-[k:KNOWS]->(c)
        SET k.mastery_level = $mastery,
            k.last_assessed = $last_assessed,
            k.assessment_type = 'class_assessment'
        """
        
        graph_manager.neo4j.graph.query(knows_query, {
            'student_id': 'student_roma',
            'concept_name': concept,
            'mastery': mastery,
            'last_assessed': datetime.now().isoformat()
        })
        roma_mastery_count += 1
    
    print(f"✅ Created {roma_mastery_count} mastery relationships for Roma")
    print(f"   Strengths: NoSQL concepts (75-95% mastery)")
    print(f"   Weaknesses: Algorithms (35-55% mastery)\n")
    
    # Create mastery for Moha (strong in algorithms, weak in NoSQL theory)
    moha_mastery_count = 0
    for item in all_concepts:
        concept = item['concept']
        topic = item['topic']
        
        # Determine mastery based on topic
        if 'Algorithm' in topic:
            mastery = random.uniform(0.80, 0.95)  # Strong
        elif 'NoSQL' in topic and 'Theory' in concept:
            mastery = random.uniform(0.40, 0.60)  # Weak
        else:
            mastery = random.uniform(0.65, 0.85)  # Average
        
        # Create KNOWS relationship
        knows_query = """
        MATCH (s:Student {student_id: $student_id})
        MATCH (c:Concept {name: $concept_name})
        MERGE (s)-[k:KNOWS]->(c)
        SET k.mastery_level = $mastery,
            k.last_assessed = $last_assessed,
            k.assessment_type = 'class_assessment'
        """
        
        graph_manager.neo4j.graph.query(knows_query, {
            'student_id': 'student_moha',
            'concept_name': concept,
            'mastery': mastery,
            'last_assessed': datetime.now().isoformat()
        })
        moha_mastery_count += 1
    
    print(f"✅ Created {moha_mastery_count} mastery relationships for Moha")
    print(f"   Strengths: Algorithms (80-95% mastery)")
    print(f"   Weaknesses: NoSQL Theory (40-60% mastery)\n")

    # Add grades for both students
    print("📝 Adding assessment grades...")
    print("-" * 80)

    # Roma's grades (varied performance)
    roma_grades = [
        {
            'type': 'quiz',
            'topic': topics[0]['topic_name'],
            'score': 72.5,
            'percentage': 72.5,
            'date': (datetime.now() - timedelta(days=20)).isoformat()
        },
        {
            'type': 'lab',
            'topic': topics[0]['topic_name'],
            'score': 85.0,
            'percentage': 85.0,
            'date': (datetime.now() - timedelta(days=18)).isoformat()
        },
        {
            'type': 'quiz',
            'topic': topics[1]['topic_name'],
            'score': 68.0,
            'percentage': 68.0,
            'date': (datetime.now() - timedelta(days=15)).isoformat()
        },
        {
            'type': 'assessment',
            'topic': topics[2]['topic_name'],
            'score': 78.5,
            'percentage': 78.5,
            'date': (datetime.now() - timedelta(days=10)).isoformat()
        }
    ]

    # Moha's grades (different pattern)
    moha_grades = [
        {
            'type': 'quiz',
            'topic': topics[0]['topic_name'],
            'score': 65.0,
            'percentage': 65.0,
            'date': (datetime.now() - timedelta(days=19)).isoformat()
        },
        {
            'type': 'lab',
            'topic': topics[1]['topic_name'],
            'score': 90.0,
            'percentage': 90.0,
            'date': (datetime.now() - timedelta(days=16)).isoformat()
        },
        {
            'type': 'quiz',
            'topic': topics[2]['topic_name'],
            'score': 88.0,
            'percentage': 88.0,
            'date': (datetime.now() - timedelta(days=12)).isoformat()
        },
        {
            'type': 'assessment',
            'topic': topics[3]['topic_name'],
            'score': 82.0,
            'percentage': 82.0,
            'date': (datetime.now() - timedelta(days=8)).isoformat()
        }
    ]

    # Update Roma's grades (convert to JSON string for Neo4j)
    import json

    update_roma_grades = """
    MATCH (s:Student {student_id: 'student_roma'})
    SET s.grades_json = $grades_json,
        s.overall_score = $overall_score
    """

    roma_overall = sum(g['score'] for g in roma_grades) / len(roma_grades)
    graph_manager.neo4j.graph.query(update_roma_grades, {
        'grades_json': json.dumps(roma_grades),
        'overall_score': round(roma_overall, 2)
    })

    print(f"✅ Added {len(roma_grades)} grades for Roma (Overall: {roma_overall:.1f}%)")

    # Update Moha's grades (convert to JSON string for Neo4j)
    update_moha_grades = """
    MATCH (s:Student {student_id: 'student_moha'})
    SET s.grades_json = $grades_json,
        s.overall_score = $overall_score
    """

    moha_overall = sum(g['score'] for g in moha_grades) / len(moha_grades)
    graph_manager.neo4j.graph.query(update_moha_grades, {
        'grades_json': json.dumps(moha_grades),
        'overall_score': round(moha_overall, 2)
    })

    print(f"✅ Added {len(moha_grades)} grades for Moha (Overall: {moha_overall:.1f}%)\n")

    return True


if __name__ == "__main__":
    success = create_demo_students()
    
    if success:
        print("\n" + "="*80)
        print("✅ DEMO STUDENTS CREATED SUCCESSFULLY!")
        print("="*80)
        print("\nYou can now login as:")
        print("  1. Roma - roma@example.com / roma123")
        print("  2. Moha - moha@example.com / moha123")
        print("\nEach student has unique strengths and weaknesses!")
    else:
        print("\n❌ Failed to create demo students")
        sys.exit(1)

