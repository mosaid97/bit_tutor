#!/usr/bin/env python3
"""
Visualize the complete knowledge graph structure from Neo4j
"""

from neo4j import GraphDatabase
import json

# Neo4j connection
driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'ktcd_password123'))

def get_graph_statistics():
    """Get comprehensive statistics about the knowledge graph"""
    with driver.session() as session:
        print("=" * 80)
        print("KNOWLEDGE GRAPH STATISTICS")
        print("=" * 80)
        
        # Node counts
        print("\n📊 NODE COUNTS:")
        print("-" * 80)
        node_query = """
        MATCH (n)
        RETURN labels(n)[0] as label, count(n) as count
        ORDER BY count DESC
        """
        result = session.run(node_query)
        total_nodes = 0
        for record in result:
            count = record['count']
            total_nodes += count
            print(f"  {record['label']:30s}: {count:5d}")
        print(f"  {'TOTAL NODES':30s}: {total_nodes:5d}")
        
        # Relationship counts
        print("\n🔗 RELATIONSHIP COUNTS:")
        print("-" * 80)
        rel_query = """
        MATCH ()-[r]->()
        RETURN type(r) as type, count(r) as count
        ORDER BY count DESC
        """
        result = session.run(rel_query)
        total_rels = 0
        for record in result:
            count = record['count']
            total_rels += count
            print(f"  {record['type']:30s}: {count:5d}")
        print(f"  {'TOTAL RELATIONSHIPS':30s}: {total_rels:5d}")
        
        # Class structure
        print("\n🎓 CLASS STRUCTURE:")
        print("-" * 80)
        class_query = """
        MATCH (c:Class)
        OPTIONAL MATCH (c)-[:INCLUDES_TOPIC]->(t:Topic)
        RETURN c.name as class_name, 
               c.class_id as class_id,
               count(DISTINCT t) as topic_count
        """
        result = session.run(class_query)
        for record in result:
            print(f"  Class: {record['class_name']}")
            print(f"    ID: {record['class_id']}")
            print(f"    Topics: {record['topic_count']}")
            print()
        
        # Topic details
        print("\n📚 TOPIC DETAILS:")
        print("-" * 80)
        topic_query = """
        MATCH (t:Topic)
        OPTIONAL MATCH (t)-[:INCLUDES_CONCEPT]->(c:Concept)
        OPTIONAL MATCH (t)-[:HAS_THEORY]->(th:Theory)
        OPTIONAL MATCH (th)-[:EXPLAINED_BY]->(v:Video)
        OPTIONAL MATCH (l:Lab)-[:PRACTICES]->(t)
        OPTIONAL MATCH (q:Quiz)-[:TESTS]->(t)
        RETURN t.name as topic_name,
               count(DISTINCT c) as concept_count,
               count(DISTINCT th) as theory_count,
               count(DISTINCT v) as video_count,
               count(DISTINCT l) as lab_count,
               count(DISTINCT q) as quiz_count
        ORDER BY t.name
        """
        result = session.run(topic_query)
        for record in result:
            print(f"  📖 {record['topic_name']}")
            print(f"     Concepts: {record['concept_count']}, Theories: {record['theory_count']}, Videos: {record['video_count']}")
            print(f"     Labs: {record['lab_count']}, Quizzes: {record['quiz_count']}")
            print()
        
        # Student data
        print("\n👥 STUDENT DATA:")
        print("-" * 80)
        student_query = """
        MATCH (s:Student)
        OPTIONAL MATCH (s)-[:REGISTERED_IN]->(c:Class)
        OPTIONAL MATCH (s)-[:ENROLLED_IN]->(t:Topic)
        OPTIONAL MATCH (s)-[:KNOWS]->(con:Concept)
        OPTIONAL MATCH (s)-[:EARNED]->(g:Grade)
        OPTIONAL MATCH (score:StudentScore {student_id: s.student_id})
        RETURN s.name as name,
               s.student_id as student_id,
               s.email as email,
               c.name as class_name,
               count(DISTINCT t) as enrolled_topics,
               count(DISTINCT con) as concepts_known,
               count(DISTINCT g) as grade_count,
               count(DISTINCT score) as score_count
        """
        result = session.run(student_query)
        for record in result:
            print(f"  👤 {record['name']} ({record['student_id']})")
            print(f"     Email: {record['email']}")
            print(f"     Class: {record['class_name']}")
            print(f"     Enrolled Topics: {record['enrolled_topics']}")
            print(f"     Concepts Known: {record['concepts_known']}")
            print(f"     Grades (Grade nodes): {record['grade_count']}")
            print(f"     Scores (StudentScore nodes): {record['score_count']}")
            print()

def get_graph_structure():
    """Get the hierarchical structure of the knowledge graph"""
    with driver.session() as session:
        print("\n" + "=" * 80)
        print("KNOWLEDGE GRAPH STRUCTURE")
        print("=" * 80)
        
        structure_query = """
        MATCH (c:Class)-[:INCLUDES_TOPIC]->(t:Topic)
        OPTIONAL MATCH (t)-[:INCLUDES_CONCEPT]->(con:Concept)
        RETURN c.name as class_name,
               t.name as topic_name,
               collect(DISTINCT con.name) as concepts
        ORDER BY c.name, t.name
        """
        result = session.run(structure_query)
        
        current_class = None
        for record in result:
            if current_class != record['class_name']:
                current_class = record['class_name']
                print(f"\n🎓 CLASS: {current_class}")
                print("=" * 80)
            
            print(f"\n  📖 TOPIC: {record['topic_name']}")
            concepts = record['concepts']
            if concepts:
                print(f"     Concepts ({len(concepts)}):")
                for i, concept in enumerate(concepts[:10], 1):  # Show first 10
                    print(f"       {i}. {concept}")
                if len(concepts) > 10:
                    print(f"       ... and {len(concepts) - 10} more")

def check_data_completeness():
    """Check if all expected data is present"""
    with driver.session() as session:
        print("\n" + "=" * 80)
        print("DATA COMPLETENESS CHECK")
        print("=" * 80)
        
        # Check for topics without videos
        print("\n⚠️  TOPICS WITHOUT VIDEOS:")
        print("-" * 80)
        no_video_query = """
        MATCH (t:Topic)
        WHERE NOT EXISTS {
            MATCH (t)-[:HAS_THEORY]->(:Theory)-[:EXPLAINED_BY]->(:Video)
        }
        RETURN t.name as topic_name
        """
        result = session.run(no_video_query)
        topics_without_videos = [record['topic_name'] for record in result]
        if topics_without_videos:
            for topic in topics_without_videos:
                print(f"  ❌ {topic}")
        else:
            print("  ✅ All topics have videos!")
        
        # Check for topics without labs
        print("\n⚠️  TOPICS WITHOUT LABS:")
        print("-" * 80)
        no_lab_query = """
        MATCH (t:Topic)
        WHERE NOT EXISTS {
            MATCH (:Lab)-[:PRACTICES]->(t)
        }
        RETURN t.name as topic_name
        """
        result = session.run(no_lab_query)
        topics_without_labs = [record['topic_name'] for record in result]
        if topics_without_labs:
            for topic in topics_without_labs:
                print(f"  ❌ {topic}")
        else:
            print("  ✅ All topics have labs!")
        
        # Check for topics without quizzes
        print("\n⚠️  TOPICS WITHOUT QUIZZES:")
        print("-" * 80)
        no_quiz_query = """
        MATCH (t:Topic)
        WHERE NOT EXISTS {
            MATCH (:Quiz)-[:TESTS]->(t)
        }
        RETURN t.name as topic_name
        """
        result = session.run(no_quiz_query)
        topics_without_quizzes = [record['topic_name'] for record in result]
        if topics_without_quizzes:
            for topic in topics_without_quizzes:
                print(f"  ❌ {topic}")
        else:
            print("  ✅ All topics have quizzes!")
        
        # Check for concepts without theory
        print("\n⚠️  CONCEPTS WITHOUT THEORY:")
        print("-" * 80)
        no_theory_query = """
        MATCH (c:Concept)
        WHERE NOT EXISTS {
            MATCH (:Theory)-[:CONSISTS_OF]->(c)
        }
        RETURN c.name as concept_name
        LIMIT 10
        """
        result = session.run(no_theory_query)
        concepts_without_theory = [record['concept_name'] for record in result]
        if concepts_without_theory:
            for concept in concepts_without_theory:
                print(f"  ❌ {concept}")
            print(f"  ... (showing first 10)")
        else:
            print("  ✅ All concepts have theory!")

def export_graph_schema():
    """Export the graph schema as a Mermaid diagram"""
    print("\n" + "=" * 80)
    print("GRAPH SCHEMA (Mermaid Format)")
    print("=" * 80)
    print("\n```mermaid")
    print("graph TD")
    print("    Class[Class]")
    print("    Topic[Topic]")
    print("    Concept[Concept]")
    print("    Theory[Theory]")
    print("    Video[Video]")
    print("    Lab[Lab]")
    print("    Quiz[Quiz]")
    print("    Question[Question]")
    print("    Student[Student]")
    print("    Grade[Grade]")
    print("    StudentScore[StudentScore]")
    print("    ")
    print("    Class -->|INCLUDES_TOPIC| Topic")
    print("    Topic -->|INCLUDES_CONCEPT| Concept")
    print("    Topic -->|HAS_THEORY| Theory")
    print("    Theory -->|CONSISTS_OF| Concept")
    print("    Theory -->|EXPLAINED_BY| Video")
    print("    Lab -->|PRACTICES| Topic")
    print("    Quiz -->|TESTS| Topic")
    print("    Question -->|TESTS| Concept")
    print("    Student -->|REGISTERED_IN| Class")
    print("    Student -->|ENROLLED_IN| Topic")
    print("    Student -->|KNOWS| Concept")
    print("    Student -->|EARNED| Grade")
    print("    Grade -->|FOR_TOPIC| Topic")
    print("```")

if __name__ == "__main__":
    try:
        get_graph_statistics()
        get_graph_structure()
        check_data_completeness()
        export_graph_schema()
        
        print("\n" + "=" * 80)
        print("✅ KNOWLEDGE GRAPH VISUALIZATION COMPLETE")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        driver.close()

