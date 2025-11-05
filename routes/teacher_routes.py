"""
Teacher Routes
Handles teacher authentication, class management, and student monitoring
"""

from flask import Blueprint, render_template, request, jsonify, redirect, url_for, session
from datetime import datetime
import hashlib
import uuid
from services.auth.password_manager import PasswordManager

# Create blueprint
teacher_bp = Blueprint('teacher', __name__, url_prefix='/teacher')

# Global reference to graph manager
graph_manager = None

def init_teacher_routes(gm):
    """Initialize routes with graph manager"""
    global graph_manager
    graph_manager = gm
    return teacher_bp

@teacher_bp.route('/login', methods=['GET'])
def login_page():
    """Display teacher login form"""
    return render_template('teacher/login.html')

@teacher_bp.route('/login', methods=['POST'])
def login_submit():
    """Process teacher login with secure password verification"""
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')

        # Validate input
        if not email or not password:
            return jsonify({'success': False, 'message': 'Email and password required'}), 400

        # SECURITY FIX: Get teacher by email only (don't include password in query)
        query = """
        MATCH (t:Teacher {email: $email})
        RETURN t.teacher_id as teacher_id,
               t.name as name,
               t.email as email,
               t.password_hash as password_hash
        """

        result = graph_manager.neo4j.graph.query(query, {'email': email})

        if result and len(result) > 0:
            teacher = result[0]

            # SECURITY FIX: Verify password using bcrypt
            if PasswordManager.verify_password(password, teacher.get('password_hash', '')):
                # Login successful
                session['teacher_id'] = teacher['teacher_id']
                session['teacher_name'] = teacher['name']
                session['teacher_email'] = teacher['email']

                return jsonify({
                    'success': True,
                    'message': 'Login successful',
                    'redirect': url_for('teacher.dashboard')
                })

        # Invalid credentials (don't specify which field is wrong for security)
        return jsonify({'success': False, 'message': 'Invalid email or password'}), 401

    except Exception as e:
        print(f"⚠️  Login error: {e}")
        # Don't expose internal errors to users
        return jsonify({'success': False, 'message': 'An error occurred. Please try again.'}), 500

@teacher_bp.route('/logout')
def logout():
    """Logout teacher"""
    session.clear()
    return redirect(url_for('teacher.login_page'))

@teacher_bp.route('/dashboard')
def dashboard():
    """Teacher dashboard with overview"""
    if 'teacher_id' not in session:
        return redirect(url_for('teacher.login_page'))
    
    teacher_id = session['teacher_id']
    
    # Get teacher's classes
    classes_query = """
    MATCH (t:Teacher {teacher_id: $teacher_id})-[:TEACHES]->(c:Class)
    OPTIONAL MATCH (c)<-[:REGISTERED_IN]-(s:Student)
    WITH c, count(DISTINCT s) as student_count
    OPTIONAL MATCH (c)-[:INCLUDES]->(topic:Topic)
    RETURN c.class_id as class_id,
           c.name as name,
           c.description as description,
           c.semester as semester,
           c.year as year,
           student_count,
           count(DISTINCT topic) as topic_count
    ORDER BY c.year DESC, c.semester
    """
    
    classes = []
    try:
        result = graph_manager.neo4j.graph.query(classes_query, {'teacher_id': teacher_id})
        if result:
            classes = [dict(r) for r in result]
    except Exception as e:
        print(f"Error fetching classes: {e}")
    
    # Get overall statistics
    stats_query = """
    MATCH (t:Teacher {teacher_id: $teacher_id})-[:TEACHES]->(c:Class)<-[:REGISTERED_IN]-(s:Student)
    RETURN count(DISTINCT s) as total_students,
           count(DISTINCT c) as total_classes,
           avg(s.overall_score) as avg_score
    """
    
    stats = {
        'total_students': 0,
        'total_classes': 0,
        'avg_score': 0
    }
    
    try:
        result = graph_manager.neo4j.graph.query(stats_query, {'teacher_id': teacher_id})
        if result and len(result) > 0:
            stats = dict(result[0])
    except Exception as e:
        print(f"Error fetching stats: {e}")
    
    return render_template('teacher/dashboard.html',
                         teacher_name=session['teacher_name'],
                         classes=classes,
                         stats=stats)

@teacher_bp.route('/classes/create', methods=['GET'])
def create_class_page():
    """Display create class form"""
    if 'teacher_id' not in session:
        return redirect(url_for('teacher.login_page'))
    
    return render_template('teacher/create_class.html',
                         teacher_name=session['teacher_name'])

@teacher_bp.route('/classes/create', methods=['POST'])
def create_class_submit():
    """Create a new class"""
    if 'teacher_id' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401
    
    try:
        data = request.get_json()
        teacher_id = session['teacher_id']
        
        # Validate required fields
        required = ['name', 'description', 'subject', 'semester', 'year']
        for field in required:
            if field not in data:
                return jsonify({'success': False, 'error': f'Missing {field}'}), 400
        
        # Generate class ID
        class_id = f"class_{uuid.uuid4().hex[:12]}"
        
        # Create class
        create_query = """
        CREATE (c:Class {
            class_id: $class_id,
            name: $name,
            description: $description,
            subject: $subject,
            semester: $semester,
            year: $year,
            status: 'active',
            created_at: datetime()
        })
        WITH c
        MATCH (t:Teacher {teacher_id: $teacher_id})
        CREATE (t)-[:TEACHES]->(c)
        RETURN c.class_id as class_id, c.name as name
        """
        
        result = graph_manager.neo4j.graph.query(create_query, {
            'class_id': class_id,
            'name': data['name'],
            'description': data['description'],
            'subject': data['subject'],
            'semester': data['semester'],
            'year': int(data['year']),
            'teacher_id': teacher_id
        })
        
        if result:
            return jsonify({
                'success': True,
                'message': 'Class created successfully',
                'class_id': class_id,
                'redirect': url_for('teacher.dashboard')
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to create class'}), 500
            
    except Exception as e:
        print(f"Error creating class: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@teacher_bp.route('/classes/<class_id>/students')
def class_students(class_id):
    """View all students in a class"""
    if 'teacher_id' not in session:
        return redirect(url_for('teacher.login_page'))
    
    # Get class info
    class_query = """
    MATCH (c:Class {class_id: $class_id})
    RETURN c.name as name, c.description as description
    """
    
    class_info = None
    try:
        result = graph_manager.neo4j.graph.query(class_query, {'class_id': class_id})
        if result:
            class_info = dict(result[0])
    except Exception as e:
        print(f"Error fetching class: {e}")
        return "Class not found", 404
    
    # Get students
    students_query = """
    MATCH (s:Student)-[:REGISTERED_IN]->(c:Class {class_id: $class_id})
    OPTIONAL MATCH (s)-[k:KNOWS]->(concept:Concept)
    WITH s, count(DISTINCT k) as concepts_known
    RETURN s.student_id as student_id,
           s.name as name,
           s.email as email,
           s.overall_score as overall_score,
           s.total_grades as total_grades,
           concepts_known,
           s.registration_date as registration_date
    ORDER BY s.name
    """
    
    students = []
    try:
        result = graph_manager.neo4j.graph.query(students_query, {'class_id': class_id})
        if result:
            students = [dict(r) for r in result]
    except Exception as e:
        print(f"Error fetching students: {e}")
    
    return render_template('teacher/class_students.html',
                         teacher_name=session['teacher_name'],
                         class_info=class_info,
                         class_id=class_id,
                         students=students)

