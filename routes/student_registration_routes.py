"""
Student Registration Routes
Handles student registration, class enrollment, and hobby collection
"""

from flask import Blueprint, render_template, request, jsonify, redirect, url_for, session
from datetime import datetime
import uuid

# Create blueprint
student_registration_bp = Blueprint('student_registration', __name__, url_prefix='/register')

# Global reference to graph manager (will be set by main app)
graph_manager = None


def init_registration_routes(gm):
    """Initialize routes with graph manager"""
    global graph_manager
    graph_manager = gm


@student_registration_bp.route('/', methods=['GET'])
def registration_form():
    """Display student registration form"""
    # Get available classes
    classes = []
    if graph_manager and graph_manager.neo4j:
        query = """
        MATCH (c:Class)
        WHERE c.status = 'active'
        RETURN c.class_id as class_id,
               c.name as name,
               c.description as description,
               c.subject as subject,
               c.semester as semester,
               c.year as year
        ORDER BY c.year DESC, c.semester
        """
        try:
            result = graph_manager.neo4j.graph.query(query)
            if result:
                classes = [dict(record) for record in result]
        except Exception as e:
            print(f"Error fetching classes: {e}")

    return render_template('student/registration.html', classes=classes)


@student_registration_bp.route('/submit', methods=['POST'])
def submit_registration():
    """Process student registration"""
    try:
        data = request.get_json()

        # Validate required fields
        required_fields = ['name', 'email', 'password', 'class_id', 'hobbies', 'interests']
        for field in required_fields:
            if field not in data:
                return jsonify({"success": False, "error": f"Missing required field: {field}"}), 400

        # Check if email already exists
        if graph_manager and graph_manager.neo4j:
            check_query = """
            MATCH (s:Student {email: $email})
            RETURN s.student_id as student_id
            LIMIT 1
            """
            result = graph_manager.neo4j.graph.query(check_query, {'email': data['email']})
            if result and len(result) > 0:
                return jsonify({"success": False, "error": "Email already registered"}), 400

        # Generate student ID
        student_id = f"student_{uuid.uuid4().hex[:8]}"

        # Hash password
        import hashlib
        password_hash = hashlib.sha256(data['password'].encode()).hexdigest()

        # Create student node
        if graph_manager and graph_manager.neo4j:
            create_student_query = """
            CREATE (s:Student {
                student_id: $student_id,
                name: $name,
                email: $email,
                password_hash: $password_hash,
                registration_date: datetime(),
                hobbies: $hobbies,
                interests: $interests,
                learning_style: $learning_style,
                initial_assessment_completed: false,
                grades: '[]',
                total_grades: 0,
                overall_score: 0.0,
                streak_days: 0,
                total_practice_hours: 0.0,
                last_updated: datetime()
            })
            RETURN s
            """

            params = {
                'student_id': student_id,
                'name': data['name'],
                'email': data['email'],
                'password_hash': password_hash,
                'hobbies': data['hobbies'],
                'interests': data['interests'],
                'learning_style': data.get('learning_style', 'visual')
            }

            graph_manager.neo4j.graph.query(create_student_query, params)

            # Register student in class
            register_query = """
            MATCH (s:Student {student_id: $student_id})
            MATCH (c:Class {class_id: $class_id})
            CREATE (s)-[:REGISTERED_IN]->(c)
            """

            graph_manager.neo4j.graph.query(
                register_query,
                {'student_id': student_id, 'class_id': data['class_id']}
            )

            # Create hobby relationships
            for hobby_name in data['hobbies']:
                hobby_query = """
                MATCH (s:Student {student_id: $student_id})
                MERGE (h:Hobby {name: $hobby_name})
                ON CREATE SET h.hobby_id = 'hobby_' + toLower(replace($hobby_name, ' ', '_')),
                              h.category = 'general',
                              h.description = $hobby_name
                CREATE (s)-[:HAS]->(h)
                """

                graph_manager.neo4j.graph.query(
                    hobby_query,
                    {'student_id': student_id, 'hobby_name': hobby_name}
                )
        
        return jsonify({
            "success": True,
            "student_id": student_id,
            "message": "Registration successful!",
            "redirect_url": url_for('student_registration.initial_assessment', student_id=student_id)
        })
    
    except Exception as e:
        print(f"Error in registration: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@student_registration_bp.route('/<student_id>/initial-assessment', methods=['GET'])
def initial_assessment(student_id):
    """Display initial comprehensive assessment"""
    # Get student info
    student = None
    if graph_manager and graph_manager.neo4j:
        query = """
        MATCH (s:Student {student_id: $student_id})
        RETURN s.name as name,
               s.email as email,
               s.hobbies as hobbies,
               s.interests as interests
        """
        result = graph_manager.neo4j.graph.query(query, {'student_id': student_id})
        if result:
            student = dict(result[0])

    if not student:
        return "Student not found", 404

    # Get class info
    class_info = None
    if graph_manager and graph_manager.neo4j:
        query = """
        MATCH (s:Student {student_id: $student_id})-[:REGISTERED_IN]->(c:Class)
        RETURN c.name as class_name,
               c.description as description
        """
        result = graph_manager.neo4j.graph.query(query, {'student_id': student_id})
        if result:
            class_info = dict(result[0])
    
    return render_template('student/initial_assessment.html',
                         student_id=student_id,
                         student=student,
                         class_info=class_info)


@student_registration_bp.route('/<student_id>/initial-assessment/questions', methods=['GET'])
def get_initial_assessment_questions(student_id):
    """Get questions for initial comprehensive assessment"""
    try:
        # Get all topics in the class
        topics = []
        if graph_manager and graph_manager.neo4j:
            query = """
            MATCH (s:Student {student_id: $student_id})-[:REGISTERED_IN]->(c:Class)-[:INCLUDES]->(t:Topic)
            RETURN t.topic_id as topic_id,
                   t.name as name
            ORDER BY t.order
            """
            result = graph_manager.neo4j.graph.query(query, {'student_id': student_id})
            if result:
                topics = [dict(record) for record in result]

        # Get 2 questions per topic from question banks
        questions = []
        question_id = 1

        for topic in topics:
            topic_query = """
            MATCH (t:Topic {topic_id: $topic_id})-[:INCLUDES_CONCEPT]->(c:Concept)
            MATCH (c)<-[:BELONGS_TO]-(qb:QuestionBank)
            WITH qb, c
            LIMIT 2
            RETURN qb.questions as questions,
                   c.name as concept_name
            """
            result = graph_manager.neo4j.graph.query(
                topic_query,
                {'topic_id': topic['topic_id']}
            )
            
            if result:
                import json
                for record in result:
                    question_list = json.loads(record['questions']) if isinstance(record['questions'], str) else record['questions']
                    if question_list and len(question_list) > 0:
                        q = question_list[0]  # Take first question
                        questions.append({
                            'id': question_id,
                            'question': q['question'],
                            'options': q['options'],
                            'correct_answer': q['correct_answer'],
                            'topic': topic['name'],
                            'concept': record['concept_name']
                        })
                        question_id += 1
        
        return jsonify({
            "success": True,
            "questions": questions,
            "total_questions": len(questions),
            "time_limit": len(questions) * 90  # 90 seconds per question
        })
    
    except Exception as e:
        print(f"Error getting assessment questions: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@student_registration_bp.route('/<student_id>/initial-assessment/submit', methods=['POST'])
def submit_initial_assessment(student_id):
    """Submit initial assessment results"""
    try:
        data = request.get_json()
        answers = data.get('answers', {})
        time_spent = data.get('time_spent', 0)
        
        # Calculate score
        correct_count = 0
        total_questions = len(answers)
        
        for question_id, answer_data in answers.items():
            if answer_data.get('selected_answer') == answer_data.get('correct_answer'):
                correct_count += 1
        
        score = (correct_count / total_questions * 100) if total_questions > 0 else 0
        
        # Update student node
        if graph_manager and graph_manager.neo4j:
            update_query = """
            MATCH (s:Student {student_id: $student_id})
            SET s.initial_assessment_completed = true,
                s.initial_assessment_score = $score,
                s.initial_assessment_date = datetime()
            RETURN s
            """

            graph_manager.neo4j.graph.query(
                update_query,
                {'student_id': student_id, 'score': score}
            )

            # Create assessment score record
            score_id = f"score_{uuid.uuid4().hex[:8]}"
            score_query = """
            CREATE (ss:StudentScore {
                score_id: $score_id,
                student_id: $student_id,
                assessment_type: 'initial_comprehensive',
                item_id: 'initial_assessment',
                score: $score,
                max_score: 100.0,
                percentage: $score,
                completed_at: datetime(),
                time_spent: $time_spent
            })
            WITH ss
            MATCH (s:Student {student_id: $student_id})
            CREATE (s)-[:ACHIEVED]->(ss)
            """

            graph_manager.neo4j.graph.query(
                score_query,
                {
                    'score_id': score_id,
                    'student_id': student_id,
                    'score': score,
                    'time_spent': time_spent
                }
            )
        
        return jsonify({
            "success": True,
            "score": score,
            "correct_count": correct_count,
            "total_questions": total_questions,
            "redirect_url": url_for('student_nexus', student_id=student_id)
        })
    
    except Exception as e:
        print(f"Error submitting assessment: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@student_registration_bp.route('/check-email', methods=['POST'])
def check_email():
    """Check if email is already registered"""
    try:
        data = request.get_json()
        email = data.get('email')
        
        if not email:
            return jsonify({"exists": False})
        
        if graph_manager and graph_manager.neo4j:
            query = """
            MATCH (s:Student {email: $email})
            RETURN s.student_id as student_id
            LIMIT 1
            """
            result = graph_manager.neo4j.graph.query(query, {'email': email})

            if result:
                return jsonify({"exists": True, "student_id": result[0]['student_id']})
        
        return jsonify({"exists": False})
    
    except Exception as e:
        print(f"Error checking email: {e}")
        return jsonify({"exists": False})

