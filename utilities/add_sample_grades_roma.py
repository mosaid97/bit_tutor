#!/usr/bin/env python3
"""
Add sample grades for student Roma to test the grades display
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.knowledge_graph.services.dynamic_graph_manager import DynamicGraphManager
import json
from datetime import datetime, timedelta

def add_sample_grades():
    """Add sample grades for student Roma"""
    
    graph_manager = DynamicGraphManager()
    
    if not graph_manager.neo4j:
        print("❌ Neo4j not available")
        return
    
    student_id = "student_roma"
    
    # Create sample grades
    sample_grades = [
        {
            "type": "quiz",
            "topic_name": "Introduction to NoSQL Databases for Big Data",
            "score": 85,
            "max_score": 100,
            "percentage": 85.0,
            "date": (datetime.now() - timedelta(days=5)).isoformat()
        },
        {
            "type": "lab",
            "topic_name": "Core Concepts of NoSQL Databases: The CAP Theorem and BASE Model",
            "score": 92,
            "max_score": 100,
            "percentage": 92.0,
            "date": (datetime.now() - timedelta(days=3)).isoformat()
        },
        {
            "type": "quiz",
            "topic_name": "Types of NoSQL Databases",
            "score": 78,
            "max_score": 100,
            "percentage": 78.0,
            "date": (datetime.now() - timedelta(days=2)).isoformat()
        },
        {
            "type": "assessment",
            "topic_name": "Fundamentals of Big Data Storing Systems and Data Modeling",
            "score": 88,
            "max_score": 100,
            "percentage": 88.0,
            "date": (datetime.now() - timedelta(days=1)).isoformat()
        },
        {
            "type": "lab",
            "topic_name": "Overview of Popular Data Analysis Algorithms",
            "score": 95,
            "max_score": 100,
            "percentage": 95.0,
            "date": datetime.now().isoformat()
        }
    ]
    
    # Convert to JSON
    grades_json = json.dumps(sample_grades)
    
    # Calculate overall score
    overall_score = sum(g['percentage'] for g in sample_grades) / len(sample_grades)
    
    # Update student with grades
    update_query = """
    MATCH (s:Student {student_id: $student_id})
    SET s.grades = $grades_json,
        s.total_grades = $total_grades,
        s.overall_score = $overall_score
    RETURN s.student_id as student_id, s.overall_score as overall_score
    """
    
    try:
        result = graph_manager.neo4j.graph.query(update_query, {
            'student_id': student_id,
            'grades_json': grades_json,
            'total_grades': len(sample_grades),
            'overall_score': overall_score
        })
        
        if result:
            print(f"✅ Added {len(sample_grades)} grades for student Roma")
            print(f"📊 Overall score: {overall_score:.1f}%")
            print(f"\nGrades added:")
            for grade in sample_grades:
                print(f"  - {grade['type'].upper()}: {grade['topic_name'][:50]}... ({grade['percentage']}%)")
        else:
            print("❌ Failed to update student")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    print("🎓 Adding sample grades for student Roma...")
    add_sample_grades()
    print("\n✅ Done!")

