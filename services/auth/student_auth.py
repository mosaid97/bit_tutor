"""
Student Authentication Service
Handles student login, registration, and session management
"""

import hashlib
import uuid
from datetime import datetime
from services.auth.password_manager import PasswordManager


class StudentAuthService:
    """Service for student authentication and session management"""

    def __init__(self, graph_manager=None):
        """Initialize authentication service with graph manager"""
        self.graph_manager = graph_manager

    def hash_password(self, password):
        """
        Hash password using bcrypt (SECURITY FIX: upgraded from SHA-256)

        Note: This method is kept for backward compatibility but now uses bcrypt
        """
        return PasswordManager.hash_password(password)
    
    def register_student(self, name, email, password, class_id, hobbies=None, interests=None):
        """
        Register a new student
        
        Args:
            name: Student's full name
            email: Student's email (used as username)
            password: Student's password
            class_id: ID of class to register in
            hobbies: List of student hobbies
            interests: List of student interests
            
        Returns:
            dict: {'success': bool, 'student_id': str, 'message': str}
        """
        if not self.graph_manager or not self.graph_manager.neo4j:
            return {'success': False, 'message': 'Database not available'}
        
        try:
            # Check if email already exists
            check_query = """
            MATCH (s:Student {email: $email})
            RETURN s.student_id as student_id
            """
            result = self.graph_manager.neo4j.graph.query(check_query, {'email': email})
            
            if result:
                return {'success': False, 'message': 'Email already registered'}
            
            # Generate student ID
            student_id = f"student_{uuid.uuid4().hex[:8]}"
            
            # Hash password
            password_hash = self.hash_password(password)
            
            # Create student node
            create_query = """
            CREATE (s:Student {
                student_id: $student_id,
                name: $name,
                email: $email,
                password_hash: $password_hash,
                registration_date: datetime(),
                hobbies: $hobbies,
                interests: $interests,
                initial_assessment_completed: false,
                total_score: 0.0,
                average_score: 0.0,
                status: 'active'
            })
            RETURN s.student_id as student_id
            """
            
            params = {
                'student_id': student_id,
                'name': name,
                'email': email,
                'password_hash': password_hash,
                'hobbies': hobbies or [],
                'interests': interests or []
            }
            
            self.graph_manager.neo4j.graph.query(create_query, params)
            
            # Register in class
            register_query = """
            MATCH (s:Student {student_id: $student_id})
            MATCH (c:Class {class_id: $class_id})
            CREATE (s)-[:REGISTERED_IN {enrolled_date: datetime()}]->(c)
            """
            
            self.graph_manager.neo4j.graph.query(
                register_query, 
                {'student_id': student_id, 'class_id': class_id}
            )
            
            return {
                'success': True,
                'student_id': student_id,
                'message': 'Registration successful'
            }
            
        except Exception as e:
            print(f"Registration error: {e}")
            return {'success': False, 'message': f'Registration failed: {str(e)}'}
    
    def login_student(self, email, password):
        """
        Authenticate student login with secure password verification

        Args:
            email: Student's email
            password: Student's password

        Returns:
            dict: {'success': bool, 'student_id': str, 'name': str, 'message': str}
        """
        if not self.graph_manager or not self.graph_manager.neo4j:
            return {'success': False, 'message': 'Database not available'}

        try:
            # SECURITY FIX: Get student by email only (don't include password in query)
            query = """
            MATCH (s:Student {email: $email})
            WHERE s.status = 'active'
            RETURN s.student_id as student_id,
                   s.name as name,
                   s.email as email,
                   s.password_hash as password_hash
            """

            result = self.graph_manager.neo4j.graph.query(query, {'email': email})

            if result and len(result) > 0:
                student = result[0]

                # SECURITY FIX: Verify password using bcrypt
                if PasswordManager.verify_password(password, student.get('password_hash', '')):
                    return {
                        'success': True,
                        'student_id': student['student_id'],
                        'name': student['name'],
                        'email': student['email'],
                        'message': 'Login successful'
                }
            else:
                return {'success': False, 'message': 'Invalid email or password'}
                
        except Exception as e:
            print(f"Login error: {e}")
            return {'success': False, 'message': f'Login failed: {str(e)}'}
    
    def get_student_info(self, student_id):
        """
        Get student information
        
        Args:
            student_id: Student's ID
            
        Returns:
            dict: Student information or None
        """
        if not self.graph_manager or not self.graph_manager.neo4j:
            return None
        
        try:
            query = """
            MATCH (s:Student {student_id: $student_id})
            OPTIONAL MATCH (s)-[:REGISTERED_IN]->(c:Class)
            RETURN s.student_id as student_id,
                   s.name as name,
                   s.email as email,
                   s.hobbies as hobbies,
                   s.interests as interests,
                   s.registration_date as registration_date,
                   s.initial_assessment_completed as assessment_completed,
                   c.class_id as class_id,
                   c.name as class_name
            """
            
            result = self.graph_manager.neo4j.graph.query(query, {'student_id': student_id})
            
            if result and len(result) > 0:
                return dict(result[0])
            return None
            
        except Exception as e:
            print(f"Error getting student info: {e}")
            return None
    
    def get_available_classes(self):
        """
        Get list of available classes for registration

        Returns:
            list: List of active classes
        """
        if not self.graph_manager or not self.graph_manager.neo4j:
            return []

        try:
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

            result = self.graph_manager.neo4j.graph.query(query)

            if result:
                return [dict(record) for record in result]
            return []

        except Exception as e:
            print(f"Error getting classes: {e}")
            return []

