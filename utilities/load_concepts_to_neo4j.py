#!/usr/bin/env python3
"""
Load Concepts to Neo4j

This script loads all concepts from lab_tutor theories into Neo4j
and creates proper relationships: Topic -> Theory -> Concept
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.knowledge_graph.services.dynamic_graph_manager import DynamicGraphManager
from services.knowledge_graph.services.lab_tutor_loader import get_lab_tutor_loader


def load_concepts_to_neo4j():
    """Load all concepts from lab_tutor into Neo4j."""
    
    print("🚀 Loading Concepts to Neo4j")
    print("=" * 60)
    
    # Initialize services
    print("📦 Initializing services...")
    graph_manager = DynamicGraphManager()
    lab_tutor_loader = get_lab_tutor_loader()
    
    if not graph_manager.neo4j:
        print("❌ Error: Neo4j not available")
        return 1
    
    # Get all theories (which contain concepts)
    print("📚 Loading theories from lab_tutor...")
    theories = lab_tutor_loader.data.get('theories', [])
    print(f"✅ Found {len(theories)} theories")
    
    # Process each theory
    total_concepts = 0
    total_topics = 0
    topics_created = set()
    
    for i, theory in enumerate(theories, 1):
        theory_name = theory.get('name', f'Theory {i}')
        topic_name = theory.get('topic', theory_name)
        concepts = theory.get('concepts', [])
        
        print(f"\n[{i}/{len(theories)}] Processing: {theory_name}")
        print(f"  Topic: {topic_name}")
        print(f"  Concepts: {len(concepts)}")
        
        if not concepts:
            print(f"  ⏭️  No concepts, skipping")
            continue
        
        try:
            # Create/update Topic node
            if topic_name not in topics_created:
                topic_query = """
                MERGE (t:Topic {name: $topic_name})
                SET t.description = COALESCE(t.description, $description)
                RETURN t.name as name
                """
                graph_manager.neo4j.graph.query(topic_query, {
                    'topic_name': topic_name,
                    'description': f"Learn about {topic_name}"
                })
                topics_created.add(topic_name)
                total_topics += 1
            
            # Create/update Theory node and link to Topic
            theory_query = """
            MERGE (th:Theory {name: $theory_name})
            SET th.summary = $summary,
                th.keywords = $keywords,
                th.compressed_text = $compressed_text
            
            WITH th
            MATCH (t:Topic {name: $topic_name})
            MERGE (t)-[:HAS_THEORY]->(th)
            
            RETURN th.name as name
            """
            graph_manager.neo4j.graph.query(theory_query, {
                'theory_name': theory_name,
                'topic_name': topic_name,
                'summary': theory.get('summary', ''),
                'keywords': json.dumps(theory.get('keywords', [])),
                'compressed_text': theory.get('original_text', '')[:5000]  # Limit size
            })
            
            # Create Concept nodes and link to Theory and Topic
            for concept in concepts:
                concept_name = concept.get('name', '')
                if not concept_name:
                    continue
                
                concept_query = """
                MERGE (c:Concept {name: $concept_name})
                SET c.concept_id = $concept_id,
                    c.definition = $definition,
                    c.description = $description,
                    c.text_evidence = $text_evidence
                
                WITH c
                MATCH (th:Theory {name: $theory_name})
                MERGE (th)-[:CONSISTS_OF]->(c)
                
                WITH c
                MATCH (t:Topic {name: $topic_name})
                MERGE (t)-[:INCLUDES_CONCEPT]->(c)
                
                RETURN c.name as name
                """
                
                graph_manager.neo4j.graph.query(concept_query, {
                    'concept_name': concept_name,
                    'concept_id': concept_name.lower().replace(' ', '_'),
                    'theory_name': theory_name,
                    'topic_name': topic_name,
                    'definition': concept.get('definition', ''),
                    'description': concept.get('definition', ''),
                    'text_evidence': concept.get('text_evidence', '')
                })
                
                total_concepts += 1
            
            print(f"  ✅ Created {len(concepts)} concepts")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 Concept Loading Complete!")
    print(f"📊 Topics created: {total_topics}")
    print(f"📚 Theories processed: {len(theories)}")
    print(f"💡 Concepts created: {total_concepts}")
    print("=" * 60)
    
    # Verify
    verify_query = """
    MATCH (t:Topic)
    OPTIONAL MATCH (t)-[:INCLUDES_CONCEPT]->(c:Concept)
    RETURN t.name as topic, count(c) as concept_count
    ORDER BY concept_count DESC
    """
    results = graph_manager.neo4j.graph.query(verify_query)
    
    print("\n📊 Verification - Topics with Concepts:")
    for r in results[:10]:
        print(f"  {r['topic']}: {r['concept_count']} concepts")
    
    return 0


if __name__ == "__main__":
    exit(load_concepts_to_neo4j())

