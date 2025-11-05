#!/usr/bin/env python3
"""
Visualize Roma and Moha in the Knowledge Graph
Shows their mastery profiles, grades, and relationships
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.knowledge_graph.services.dynamic_graph_manager import DynamicGraphManager

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")

def visualize_student_profile(graph_manager, student_id, student_name):
    """Visualize a student's complete profile"""
    
    print_header(f"📊 {student_name}'s Profile in Knowledge Graph")
    
    # Get student basic info
    query = """
    MATCH (s:Student {student_id: $student_id})
    RETURN s.name as name, s.email as email, s.overall_score as overall_score,
           s.hobbies as hobbies, s.interests as interests,
           s.class_assessment_score as class_score,
           s.streak_days as streak, s.total_practice_hours as hours,
           s.grades_json as grades_json
    """
    result = graph_manager.neo4j.graph.query(query, {'student_id': student_id})
    
    if result:
        student = result[0]
        print(f"👤 Name: {student['name']}")
        print(f"📧 Email: {student['email']}")
        print(f"📊 Overall Score: {student['overall_score']}%")
        print(f"📝 Class Assessment: {student['class_score']}%")
        print(f"🔥 Streak: {student['streak']} days")
        print(f"⏱️  Practice Hours: {student['hours']} hours")
        print(f"🎮 Hobbies: {', '.join(student['hobbies'])}")
        print(f"💡 Interests: {', '.join(student['interests'])}")
        
        # Parse and display grades
        if student['grades_json']:
            import json
            grades = json.loads(student['grades_json'])
            print(f"\n📈 Grades ({len(grades)} recorded):")
            for i, grade in enumerate(grades, 1):
                print(f"  {i}. {grade['type'].upper()} - {grade['topic']}: {grade['score']}%")
    
    # Get class registration
    query = """
    MATCH (s:Student {student_id: $student_id})-[:REGISTERED_IN]->(c:Class)
    RETURN c.name as class_name, c.class_id as class_id
    """
    result = graph_manager.neo4j.graph.query(query, {'student_id': student_id})
    if result:
        print(f"\n🎓 Registered in Class: {result[0]['class_name']}")
    
    # Get mastery statistics
    query = """
    MATCH (s:Student {student_id: $student_id})-[k:KNOWS]->(c:Concept)
    RETURN count(c) as total_concepts,
           avg(k.mastery_level) as avg_mastery,
           max(k.mastery_level) as max_mastery,
           min(k.mastery_level) as min_mastery
    """
    result = graph_manager.neo4j.graph.query(query, {'student_id': student_id})
    if result:
        stats = result[0]
        print(f"\n📚 Mastery Statistics:")
        print(f"  Total Concepts Tracked: {stats['total_concepts']}")
        print(f"  Average Mastery: {stats['avg_mastery']*100:.1f}%")
        print(f"  Highest Mastery: {stats['max_mastery']*100:.1f}%")
        print(f"  Lowest Mastery: {stats['min_mastery']*100:.1f}%")
    
    # Get top strengths
    query = """
    MATCH (s:Student {student_id: $student_id})-[k:KNOWS]->(c:Concept)
    RETURN c.name as concept, k.mastery_level as mastery
    ORDER BY k.mastery_level DESC
    LIMIT 10
    """
    result = graph_manager.neo4j.graph.query(query, {'student_id': student_id})
    if result:
        print(f"\n💪 Top 10 Strengths:")
        for i, item in enumerate(result, 1):
            mastery_pct = item['mastery'] * 100
            bar = "█" * int(mastery_pct / 5)
            print(f"  {i:2d}. {item['concept'][:40]:40s} {mastery_pct:5.1f}% {bar}")
    
    # Get top weaknesses
    query = """
    MATCH (s:Student {student_id: $student_id})-[k:KNOWS]->(c:Concept)
    RETURN c.name as concept, k.mastery_level as mastery
    ORDER BY k.mastery_level ASC
    LIMIT 10
    """
    result = graph_manager.neo4j.graph.query(query, {'student_id': student_id})
    if result:
        print(f"\n⚠️  Top 10 Areas for Improvement:")
        for i, item in enumerate(result, 1):
            mastery_pct = item['mastery'] * 100
            bar = "░" * int(mastery_pct / 5)
            print(f"  {i:2d}. {item['concept'][:40]:40s} {mastery_pct:5.1f}% {bar}")

def compare_students(graph_manager):
    """Compare Roma and Moha side by side"""
    
    print_header("🔄 Roma vs Moha Comparison")
    
    # Get mastery comparison for key topics
    query = """
    MATCH (roma:Student {student_id: 'student_roma'})-[kr:KNOWS]->(c:Concept)
    MATCH (moha:Student {student_id: 'student_moha'})-[km:KNOWS]->(c)
    MATCH (c)<-[:INCLUDES_CONCEPT]-(t:Topic)
    RETURN t.name as topic,
           avg(kr.mastery_level) as roma_avg,
           avg(km.mastery_level) as moha_avg
    ORDER BY abs(avg(kr.mastery_level) - avg(km.mastery_level)) DESC
    LIMIT 10
    """
    result = graph_manager.neo4j.graph.query(query)
    
    if result:
        print("📊 Topic Mastery Comparison (Biggest Differences):\n")
        print(f"{'Topic':<50} {'Roma':>8} {'Moha':>8} {'Diff':>8}")
        print("-" * 80)
        
        for item in result:
            roma_pct = item['roma_avg'] * 100
            moha_pct = item['moha_avg'] * 100
            diff = roma_pct - moha_pct
            
            # Visual indicator
            if diff > 10:
                indicator = "Roma ✓"
            elif diff < -10:
                indicator = "Moha ✓"
            else:
                indicator = "Similar"
            
            topic_short = item['topic'][:48]
            print(f"{topic_short:<50} {roma_pct:7.1f}% {moha_pct:7.1f}% {diff:+7.1f}% {indicator}")
    
    # Overall comparison
    query = """
    MATCH (roma:Student {student_id: 'student_roma'})
    MATCH (moha:Student {student_id: 'student_moha'})
    RETURN roma.overall_score as roma_score,
           moha.overall_score as moha_score,
           roma.streak_days as roma_streak,
           moha.streak_days as moha_streak,
           roma.total_practice_hours as roma_hours,
           moha.total_practice_hours as moha_hours
    """
    result = graph_manager.neo4j.graph.query(query)
    
    if result:
        comp = result[0]
        print(f"\n📈 Overall Performance:")
        print(f"  Overall Score:    Roma {comp['roma_score']}%  vs  Moha {comp['moha_score']}%")
        print(f"  Streak Days:      Roma {comp['roma_streak']} days  vs  Moha {comp['moha_streak']} days")
        print(f"  Practice Hours:   Roma {comp['roma_hours']} hrs  vs  Moha {comp['moha_hours']} hrs")

def show_knowledge_graph_stats(graph_manager):
    """Show overall knowledge graph statistics"""
    
    print_header("📊 Knowledge Graph Statistics")
    
    # Node counts
    queries = {
        'Classes': "MATCH (n:Class) RETURN count(n) as count",
        'Topics': "MATCH (n:Topic) RETURN count(n) as count",
        'Concepts': "MATCH (n:Concept) RETURN count(n) as count",
        'Questions': "MATCH (n:Question) RETURN count(n) as count",
        'Students': "MATCH (n:Student) RETURN count(n) as count",
        'Labs': "MATCH (n:Lab) RETURN count(n) as count",
        'Quizzes': "MATCH (n:Quiz) RETURN count(n) as count",
    }
    
    print("📦 Node Counts:")
    for label, query in queries.items():
        result = graph_manager.neo4j.graph.query(query)
        count = result[0]['count'] if result else 0
        print(f"  {label:<15} {count:>5}")
    
    # Relationship counts
    rel_queries = {
        'INCLUDES': "MATCH ()-[r:INCLUDES]->() RETURN count(r) as count",
        'INCLUDES_CONCEPT': "MATCH ()-[r:INCLUDES_CONCEPT]->() RETURN count(r) as count",
        'TESTS': "MATCH ()-[r:TESTS]->() RETURN count(r) as count",
        'REGISTERED_IN': "MATCH ()-[r:REGISTERED_IN]->() RETURN count(r) as count",
        'KNOWS': "MATCH ()-[r:KNOWS]->() RETURN count(r) as count",
    }
    
    print("\n🔗 Relationship Counts:")
    for rel_type, query in rel_queries.items():
        result = graph_manager.neo4j.graph.query(query)
        count = result[0]['count'] if result else 0
        print(f"  {rel_type:<20} {count:>5}")

def main():
    """Main visualization function"""
    
    print("\n" + "🚀 " * 20)
    print("  KNOWLEDGE GRAPH VISUALIZATION - ROMA & MOHA")
    print("🚀 " * 20)
    
    # Initialize graph manager
    graph_manager = DynamicGraphManager()
    
    # Show overall stats
    show_knowledge_graph_stats(graph_manager)
    
    # Visualize Roma
    visualize_student_profile(graph_manager, 'student_roma', 'Roma')
    
    # Visualize Moha
    visualize_student_profile(graph_manager, 'student_moha', 'Moha')
    
    # Compare students
    compare_students(graph_manager)
    
    print("\n" + "✅ " * 20)
    print("  VISUALIZATION COMPLETE!")
    print("✅ " * 20 + "\n")

if __name__ == "__main__":
    main()

