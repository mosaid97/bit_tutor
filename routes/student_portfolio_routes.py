# routes/student_portfolio_routes.py

"""
Student Portfolio Routes

This module provides routes for:
1. Student portfolio/dashboard
2. Pre-topic assessments
3. Adaptive learning paths
4. Progress tracking
"""

from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for
from datetime import datetime

# Create Blueprint
student_portfolio_bp = Blueprint('student_portfolio', __name__, url_prefix='/student')

# Initialize managers (will be set by app)
graph_manager = None
assessment_engine = None


def init_student_portfolio_routes(app, dynamic_graph_manager, assessment_eng):
    """
    Initialize student portfolio routes with the Flask app and managers.
    
    Args:
        app: Flask application instance
        dynamic_graph_manager: Instance of DynamicGraphManager
        assessment_eng: Instance of AssessmentEngine
    """
    global graph_manager, assessment_engine
    graph_manager = dynamic_graph_manager
    assessment_engine = assessment_eng
    app.register_blueprint(student_portfolio_bp)
    print("✅ Student portfolio routes initialized")


# ==================== PORTFOLIO/DASHBOARD ====================

@student_portfolio_bp.route('/<student_id>/portfolio')
def portfolio(student_id):
    """
    Student portfolio page showing comprehensive progress and achievements.
    """
    if graph_manager:
        portfolio_data = graph_manager.get_student_portfolio(student_id)
    else:
        portfolio_data = {
            "student_id": student_id,
            "name": "Student",
            "overall_progress": {
                "total_concepts": 0,
                "mastered_concepts": 0,
                "average_mastery": 0.0,
                "completion_percentage": 0.0
            },
            "topic_progress": {},
            "recent_assessments": []
        }
    
    return render_template('student/portfolio.html',
                         student_id=student_id,
                         portfolio=portfolio_data)


@student_portfolio_bp.route('/api/<student_id>/portfolio/data')
def get_portfolio_data(student_id):
    """API endpoint for portfolio data."""
    if graph_manager:
        portfolio_data = graph_manager.get_student_portfolio(student_id)
    else:
        portfolio_data = {
            "student_id": student_id,
            "overall_progress": {
                "total_concepts": 0,
                "mastered_concepts": 0,
                "average_mastery": 0.0
            }
        }
    
    return jsonify(portfolio_data)


# ==================== PRE-TOPIC ASSESSMENTS ====================

@student_portfolio_bp.route('/<student_id>/assessment/pre-topic/<topic_name>')
def pre_topic_assessment(student_id, topic_name):
    """
    Display pre-topic assessment page.
    """
    # Get concepts for this topic
    if graph_manager and graph_manager.neo4j:
        query = """
        MATCH (t:TOPIC {name: $topic_name})-[:HAS_CONCEPT]->(c:CONCEPT)
        RETURN c.name as name,
               c.definition as definition,
               c.keywords as keywords
        """
        results = graph_manager.neo4j.graph.query(query, {"topic_name": topic_name})
        
        concepts = []
        for record in results:
            concepts.append({
                "name": record.get('name'),
                "definition": record.get('definition', ''),
                "keywords": record.get('keywords', [])
            })
    else:
        concepts = []
    
    # Generate assessment
    if assessment_engine and concepts:
        assessment = assessment_engine.generate_pre_topic_assessment(topic_name, concepts)
    else:
        assessment = {
            "assessment_id": f"pre_topic_{topic_name}",
            "topic_name": topic_name,
            "questions": [],
            "total_questions": 0
        }
    
    return render_template('student/pre_topic_assessment.html',
                         student_id=student_id,
                         topic_name=topic_name,
                         assessment=assessment)


@student_portfolio_bp.route('/api/<student_id>/assessment/generate', methods=['POST'])
def generate_assessment(student_id):
    """
    API endpoint to generate a pre-topic assessment.
    """
    data = request.get_json()
    topic_name = data.get('topic_name')
    
    if not topic_name:
        return jsonify({"error": "Topic name is required"}), 400
    
    # Get concepts for this topic
    if graph_manager and graph_manager.neo4j:
        query = """
        MATCH (t:TOPIC {name: $topic_name})-[:HAS_CONCEPT]->(c:CONCEPT)
        RETURN c.name as name,
               c.definition as definition,
               c.keywords as keywords
        """
        results = graph_manager.neo4j.graph.query(query, {"topic_name": topic_name})
        
        concepts = []
        for record in results:
            concepts.append({
                "name": record.get('name'),
                "definition": record.get('definition', ''),
                "keywords": record.get('keywords', [])
            })
    else:
        return jsonify({"error": "No concepts found for this topic"}), 404
    
    # Generate assessment
    if assessment_engine:
        assessment = assessment_engine.generate_pre_topic_assessment(topic_name, concepts)
        return jsonify(assessment), 201
    else:
        return jsonify({"error": "Assessment engine not available"}), 500


@student_portfolio_bp.route('/api/<student_id>/assessment/submit', methods=['POST'])
def submit_assessment(student_id):
    """
    API endpoint to submit and evaluate an assessment.
    """
    data = request.get_json()
    
    assessment_id = data.get('assessment_id')
    answers = data.get('answers', {})
    
    if not assessment_id or not answers:
        return jsonify({"error": "Assessment ID and answers are required"}), 400
    
    # Evaluate assessment
    if assessment_engine:
        results = assessment_engine.evaluate_assessment(assessment_id, student_id, answers)
        
        # Generate adaptive learning path
        topic_name = results.get('assessment_id', '').split('_')[2] if '_' in results.get('assessment_id', '') else 'Unknown'
        learning_path = assessment_engine.generate_adaptive_learning_path(student_id, topic_name, results)
        
        return jsonify({
            "results": results,
            "learning_path": learning_path
        }), 200
    else:
        return jsonify({"error": "Assessment engine not available"}), 500


# ==================== ADAPTIVE LEARNING PATHS ====================

@student_portfolio_bp.route('/<student_id>/learning-path/<topic_name>')
def learning_path(student_id, topic_name):
    """
    Display adaptive learning path for a topic.
    """
    # This would typically be generated after an assessment
    # For now, we'll show a placeholder
    return render_template('student/learning_path.html',
                         student_id=student_id,
                         topic_name=topic_name)


@student_portfolio_bp.route('/api/<student_id>/learning-path/<topic_name>')
def get_learning_path(student_id, topic_name):
    """
    API endpoint to get the adaptive learning path for a topic.
    """
    # Get the most recent assessment results for this topic
    if graph_manager and graph_manager.neo4j:
        query = """
        MATCH (s:Student {student_id: $student_id})-[att:ATTEMPTED]->(a:Assessment {topic_name: $topic_name})
        RETURN a.assessment_id as assessment_id,
               att.score as score,
               att.completed_at as completed_at
        ORDER BY att.completed_at DESC
        LIMIT 1
        """
        results = graph_manager.neo4j.graph.query(query, {
            "student_id": student_id,
            "topic_name": topic_name
        })
        
        if results:
            record = results[0]
            assessment_results = {
                "overall_score": record.get('score', 0),
                "concept_mastery": {}  # Would need to fetch this separately
            }
            
            if assessment_engine:
                learning_path = assessment_engine.generate_adaptive_learning_path(
                    student_id, topic_name, assessment_results
                )
                return jsonify(learning_path), 200
    
    # Default response if no assessment found
    return jsonify({
        "student_id": student_id,
        "topic_name": topic_name,
        "path_type": "standard",
        "recommendations": [],
        "next_steps": ["Take the pre-topic assessment to get personalized recommendations"]
    }), 200


# ==================== GRADED QUIZZES ====================

@student_portfolio_bp.route('/<student_id>/quiz/<topic_name>')
def graded_quiz(student_id, topic_name):
    """
    Display graded quiz page for a topic.
    """
    # Get concepts for this topic
    if graph_manager and graph_manager.neo4j:
        query = """
        MATCH (t:TOPIC {name: $topic_name})-[:HAS_CONCEPT]->(c:CONCEPT)
        RETURN c.name as name,
               c.definition as definition,
               c.keywords as keywords
        """
        results = graph_manager.neo4j.graph.query(query, {"topic_name": topic_name})
        
        concepts = []
        for record in results:
            concepts.append({
                "name": record.get('name'),
                "definition": record.get('definition', ''),
                "keywords": record.get('keywords', [])
            })
    else:
        concepts = []
    
    # Generate graded quiz (similar to pre-topic but marked as graded)
    if assessment_engine and concepts:
        assessment = assessment_engine.generate_pre_topic_assessment(topic_name, concepts, questions_per_concept=5)
        assessment['type'] = 'graded_quiz'
        assessment['assessment_id'] = f"graded_quiz_{topic_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    else:
        assessment = {
            "assessment_id": f"graded_quiz_{topic_name}",
            "topic_name": topic_name,
            "type": "graded_quiz",
            "questions": [],
            "total_questions": 0
        }
    
    return render_template('student/graded_quiz.html',
                         student_id=student_id,
                         topic_name=topic_name,
                         quiz=assessment)


@student_portfolio_bp.route('/api/<student_id>/quiz/submit', methods=['POST'])
def submit_graded_quiz(student_id):
    """
    API endpoint to submit and grade a quiz.
    """
    data = request.get_json()

    quiz_id = data.get('quiz_id')
    answers = data.get('answers', {})
    topic_name = data.get('topic_name', '')

    if not quiz_id or not answers:
        return jsonify({"error": "Quiz ID and answers are required"}), 400

    # Evaluate quiz (same as assessment but this contributes to grade)
    if assessment_engine:
        results = assessment_engine.evaluate_assessment(quiz_id, student_id, answers)
        results['graded'] = True
        results['contributes_to_grade'] = True

        # Record grade in student attributes
        if graph_manager and graph_manager.neo4j:
            try:
                from datetime import datetime
                import json

                score = results.get('score', 0)
                max_score = results.get('total_questions', 0)
                percentage = (score / max_score * 100) if max_score > 0 else 0

                # Get current grades
                get_grades_query = """
                MATCH (s:Student {student_id: $student_id})
                RETURN s.grades as grades_json
                """
                result = graph_manager.neo4j.graph.query(get_grades_query, {
                    'student_id': student_id
                })

                grades = []
                if result and len(result) > 0:
                    grades_json = result[0].get('grades_json')
                    if grades_json:
                        try:
                            grades = json.loads(grades_json)
                        except:
                            grades = []

                # Add new grade
                grade_id = f"grade_{student_id}_quiz_{int(datetime.now().timestamp())}"
                new_grade = {
                    'grade_id': grade_id,
                    'type': 'quiz',
                    'score': score,
                    'max_score': max_score,
                    'percentage': percentage,
                    'date': datetime.now().isoformat(),
                    'topic_name': topic_name,
                    'quiz_id': quiz_id
                }
                grades.append(new_grade)

                # Calculate overall score
                overall_score = sum(g.get('percentage', 0) for g in grades) / len(grades) if grades else 0

                # Update student
                update_query = """
                MATCH (s:Student {student_id: $student_id})
                SET s.grades = $grades_json,
                    s.total_grades = $total_grades,
                    s.overall_score = $overall_score,
                    s.last_updated = datetime()
                RETURN s.overall_score as overall_score
                """

                graph_manager.neo4j.graph.query(update_query, {
                    'student_id': student_id,
                    'grades_json': json.dumps(grades),
                    'total_grades': len(grades),
                    'overall_score': overall_score
                })

                results['grade_recorded'] = True
                results['grade_id'] = grade_id
            except Exception as e:
                print(f"Error recording grade: {e}")
                results['grade_recorded'] = False

        return jsonify(results), 200
    else:
        return jsonify({"error": "Assessment engine not available"}), 500


# ==================== PROGRESS TRACKING ====================

@student_portfolio_bp.route('/api/<student_id>/progress/concepts')
def get_concept_progress(student_id):
    """
    API endpoint to get student's progress on all concepts.
    """
    if graph_manager and graph_manager.neo4j:
        query = """
        MATCH (s:Student {student_id: $student_id})-[m:HAS_MASTERY]->(c:CONCEPT)
        OPTIONAL MATCH (c)<-[:HAS_CONCEPT]-(t:TOPIC)
        RETURN c.name as concept,
               t.name as topic,
               m.level as mastery,
               m.updated_at as last_updated
        ORDER BY t.name, c.name
        """
        results = graph_manager.neo4j.graph.query(query, {"student_id": student_id})
        
        progress = []
        for record in results:
            progress.append({
                "concept": record.get('concept'),
                "topic": record.get('topic', 'General'),
                "mastery": record.get('mastery', 0),
                "last_updated": record.get('last_updated')
            })
        
        return jsonify({"progress": progress}), 200
    else:
        return jsonify({"progress": []}), 200

