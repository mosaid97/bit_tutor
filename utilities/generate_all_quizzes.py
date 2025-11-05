#!/usr/bin/env python3
"""
Generate Quizzes for All Topics

This script generates one quiz per topic (37 quizzes total) and stores them in Neo4j.
Each quiz has 15 questions pulled from the concepts in that topic.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.knowledge_graph.services.dynamic_graph_manager import DynamicGraphManager
from services.knowledge_graph.services.lab_tutor_loader import get_lab_tutor_loader
from services.content_generation.services.quiz_generator import QuizGenerator


def generate_all_quizzes():
    """Generate quizzes for all topics in the knowledge graph."""
    
    print("🚀 Generating Quizzes for All Topics")
    print("=" * 60)
    
    # Initialize services
    print("📦 Initializing services...")
    graph_manager = DynamicGraphManager()
    lab_tutor_loader = get_lab_tutor_loader()
    quiz_generator = QuizGenerator()
    
    if not graph_manager.neo4j:
        print("❌ Error: Neo4j not available")
        return 1
    
    # Get all topics
    print("📚 Loading topics from lab_tutor...")
    topics = lab_tutor_loader.get_all_topics()
    print(f"✅ Found {len(topics)} topics")
    
    # Check existing quizzes
    existing_query = """
    MATCH (q:Quiz)
    RETURN q.topic_name as topic_name
    """
    existing_results = graph_manager.neo4j.graph.query(existing_query)
    existing_topics = {r['topic_name'] for r in existing_results} if existing_results else set()
    
    print(f"📊 Existing quizzes: {len(existing_topics)}")
    print(f"🎯 Need to generate: {len(topics) - len(existing_topics)} quizzes")
    print("-" * 60)
    
    # Generate quizzes
    generated = 0
    skipped = 0
    failed = 0
    
    for i, topic in enumerate(topics, 1):
        topic_name = topic['name']
        
        # Skip if quiz already exists
        if topic_name in existing_topics:
            print(f"[{i}/{len(topics)}] ⏭️  Skipping {topic_name} (already exists)")
            skipped += 1
            continue
        
        try:
            print(f"[{i}/{len(topics)}] 🔨 Generating quiz for: {topic_name}")

            # Get concepts from Neo4j
            concept_query = """
            MATCH (t:Topic {name: $topic_name})-[:INCLUDES_CONCEPT]->(c:Concept)
            RETURN c.name as name, c.description as description
            """
            concept_results = graph_manager.neo4j.graph.query(concept_query, {'topic_name': topic_name})
            # Quiz generator expects List[str], not List[Dict]
            concepts = [c['name'] for c in concept_results] if concept_results else []

            # Get theories from Neo4j
            theory_query = """
            MATCH (t:Topic {name: $topic_name})-[:HAS_THEORY]->(th:Theory)
            RETURN th.name as name, th.compressed_text as text
            """
            theory_results = graph_manager.neo4j.graph.query(theory_query, {'topic_name': topic_name})
            theories = [{'name': t['name'], 'text': t.get('text', '')} for t in theory_results] if theory_results else []

            if not concepts:
                print(f"  ⚠️  No concepts found for {topic_name}, skipping")
                skipped += 1
                continue
            
            # Generate quiz
            quiz_data = quiz_generator.generate_quiz_for_topic(
                topic_name=topic_name,
                concepts=concepts,
                theory_data=theories,
                num_questions=15
            )
            
            # Store in Neo4j - serialize questions as JSON string
            create_query = """
            MERGE (q:Quiz {quiz_id: $quiz_id})
            SET q.topic_name = $topic_name,
                q.title = $title,
                q.description = $description,
                q.total_questions = $total_questions,
                q.passing_score = $passing_score,
                q.time_limit = $time_limit,
                q.questions_json = $questions_json,
                q.created_at = datetime()

            WITH q
            MATCH (t:Topic {name: $topic_name})
            MERGE (q)-[:TESTS]->(t)

            RETURN q.quiz_id as quiz_id
            """

            # Serialize questions to JSON string
            import json
            questions_json = json.dumps(quiz_data['questions'])

            params = {
                'quiz_id': f"quiz_{topic_name.lower().replace(' ', '_')}",
                'topic_name': topic_name,
                'title': quiz_data['title'],
                'description': quiz_data['description'],
                'total_questions': quiz_data['total_questions'],
                'passing_score': quiz_data['passing_score'],
                'time_limit': quiz_data['time_limit'],
                'questions_json': questions_json
            }
            
            result = graph_manager.neo4j.graph.query(create_query, params)
            
            if result:
                print(f"  ✅ Created quiz with {len(quiz_data['questions'])} questions")
                generated += 1
            else:
                print(f"  ⚠️  Failed to create quiz in Neo4j")
                failed += 1
                
        except Exception as e:
            print(f"  ❌ Error generating quiz: {e}")
            failed += 1
            continue
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 Quiz Generation Complete!")
    print(f"✅ Generated: {generated}")
    print(f"⏭️  Skipped: {skipped}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total quizzes in database: {len(existing_topics) + generated}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(generate_all_quizzes())

