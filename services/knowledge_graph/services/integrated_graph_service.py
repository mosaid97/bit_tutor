# services/knowledge_graph/services/integrated_graph_service.py

"""
Integrated Knowledge Graph Service

This service provides a complete integration of all knowledge graph components:
- Students, Teachers, Topics, Concepts, Theories
- Assessments, Quizzes, Labs, Reading Materials
- All relationships and progress tracking
- Complete pipeline from database to front-end
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from pathlib import Path
import sys
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Load environment variables
load_dotenv()


class IntegratedGraphService:
    """
    Complete knowledge graph integration service.

    Manages the entire knowledge graph ecosystem:
    - Static content (Topics, Theories, Concepts from lab_tutor)
    - Dynamic content (Students, Teachers, Progress)
    - Generated content (Assessments, Quizzes, Labs)
    - Learning materials (Blogs, Resources)
    """

    def __init__(self, uri: str = None, username: str = None, password: str = None):
        """Initialize the integrated graph service."""
        # Get connection parameters from environment or use defaults
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = username or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "ktcd_password123")
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")

        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            # Test connection
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
            print(f"✅ Connected to Neo4j at {self.uri}")
        except Exception as e:
            print(f"❌ Could not connect to Neo4j: {e}")
            self.driver = None

        self.initialized = False

    def _execute_query(self, query: str, parameters: dict = None):
        """Execute a Cypher query and return results."""
        if self.driver is None:
            raise Exception("Neo4j driver not initialized")

        with self.driver.session(database=self.database) as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def _execute_write(self, query: str, parameters: dict = None):
        """Execute a write query."""
        if self.driver is None:
            raise Exception("Neo4j driver not initialized")

        with self.driver.session(database=self.database) as session:
            result = session.run(query, parameters or {})
            return result.consume()

    def initialize_complete_schema(self):
        """
        Create the complete knowledge graph schema with all node types and relationships.
        """
        if self.driver is None:
            print("Neo4j not available, skipping schema initialization")
            return False

        print("🚀 Initializing Complete Knowledge Graph Schema...")

        try:
            # ===== NODE CONSTRAINTS =====
            constraints = [
                # Core Academic Entities
                "CREATE CONSTRAINT class_id_unique IF NOT EXISTS FOR (c:Class) REQUIRE c.class_id IS UNIQUE",
                "CREATE CONSTRAINT topic_id_unique IF NOT EXISTS FOR (t:Topic) REQUIRE t.topic_id IS UNIQUE",
                "CREATE CONSTRAINT theory_id_unique IF NOT EXISTS FOR (th:Theory) REQUIRE th.theory_id IS UNIQUE",
                "CREATE CONSTRAINT concept_id_unique IF NOT EXISTS FOR (c:Concept) REQUIRE c.concept_id IS UNIQUE",

                # People
                "CREATE CONSTRAINT student_id_unique IF NOT EXISTS FOR (s:Student) REQUIRE s.student_id IS UNIQUE",
                "CREATE CONSTRAINT teacher_id_unique IF NOT EXISTS FOR (t:Teacher) REQUIRE t.teacher_id IS UNIQUE",
                "CREATE CONSTRAINT hobby_id_unique IF NOT EXISTS FOR (h:Hobby) REQUIRE h.hobby_id IS UNIQUE",

                # Assessment & Learning
                "CREATE CONSTRAINT assessment_id_unique IF NOT EXISTS FOR (a:Assessment) REQUIRE a.assessment_id IS UNIQUE",
                "CREATE CONSTRAINT quiz_id_unique IF NOT EXISTS FOR (q:Quiz) REQUIRE q.quiz_id IS UNIQUE",
                "CREATE CONSTRAINT question_id_unique IF NOT EXISTS FOR (q:Question) REQUIRE q.question_id IS UNIQUE",
                "CREATE CONSTRAINT lab_id_unique IF NOT EXISTS FOR (l:Lab) REQUIRE l.lab_id IS UNIQUE",

                # Resources
                "CREATE CONSTRAINT reading_id_unique IF NOT EXISTS FOR (r:ReadingMaterial) REQUIRE r.reading_id IS UNIQUE",
                "CREATE CONSTRAINT video_id_unique IF NOT EXISTS FOR (v:Video) REQUIRE v.video_id IS UNIQUE",
                "CREATE CONSTRAINT score_id_unique IF NOT EXISTS FOR (ss:StudentScore) REQUIRE ss.score_id IS UNIQUE",
                "CREATE CONSTRAINT final_exam_id_unique IF NOT EXISTS FOR (fe:FinalExam) REQUIRE fe.exam_id IS UNIQUE",
            ]

            # ===== PERFORMANCE INDEXES =====
            indexes = [
                # Class indexes
                "CREATE INDEX class_name_idx IF NOT EXISTS FOR (c:Class) ON (c.name)",
                "CREATE INDEX class_status_idx IF NOT EXISTS FOR (c:Class) ON (c.status)",

                # Student indexes
                "CREATE INDEX student_name_idx IF NOT EXISTS FOR (s:Student) ON (s.name)",
                "CREATE INDEX student_email_idx IF NOT EXISTS FOR (s:Student) ON (s.email)",

                # Teacher indexes
                "CREATE INDEX teacher_name_idx IF NOT EXISTS FOR (t:Teacher) ON (t.name)",

                # Topic indexes
                "CREATE INDEX topic_name_idx IF NOT EXISTS FOR (t:Topic) ON (t.name)",
                "CREATE INDEX topic_order_idx IF NOT EXISTS FOR (t:Topic) ON (t.order)",

                # Concept indexes
                "CREATE INDEX concept_name_idx IF NOT EXISTS FOR (c:Concept) ON (c.name)",

                # Assessment indexes
                "CREATE INDEX assessment_type_idx IF NOT EXISTS FOR (a:Assessment) ON (a.assessment_type)",

                # Quiz indexes
                "CREATE INDEX quiz_topic_idx IF NOT EXISTS FOR (q:Quiz) ON (q.topic_id)",

                # Question indexes
                "CREATE INDEX question_personalized_idx IF NOT EXISTS FOR (q:Question) ON (q.personalized)",

                # Lab indexes
                "CREATE INDEX lab_difficulty_idx IF NOT EXISTS FOR (l:Lab) ON (l.difficulty)",

                # Video indexes
                "CREATE INDEX video_source_idx IF NOT EXISTS FOR (v:Video) ON (v.source)",

                # Score indexes
                "CREATE INDEX score_student_idx IF NOT EXISTS FOR (ss:StudentScore) ON (ss.student_id)",
                "CREATE INDEX score_type_idx IF NOT EXISTS FOR (ss:StudentScore) ON (ss.assessment_type)",
            ]

            # Apply constraints
            for constraint in constraints:
                try:
                    self._execute_write(constraint)
                    entity = constraint.split('FOR')[1].split('REQUIRE')[0].strip()
                    print(f"  ✅ Constraint: {entity}")
                except Exception as e:
                    if "already exists" not in str(e).lower() and "equivalent" not in str(e).lower():
                        print(f"  ⚠️  Constraint error: {e}")

            # Apply indexes
            for index in indexes:
                try:
                    self._execute_write(index)
                    idx_name = index.split()[2]
                    print(f"  ✅ Index: {idx_name}")
                except Exception as e:
                    if "already exists" not in str(e).lower() and "equivalent" not in str(e).lower():
                        print(f"  ⚠️  Index error: {e}")

            self.initialized = True
            print("✅ Complete Knowledge Graph Schema Initialized!")
            return True

        except Exception as e:
            print(f"❌ Error initializing schema: {e}")
            return False
    
    def load_lab_tutor_content(self, json_file_path: str):
        """
        Load static content from lab_tutor JSON file.
        
        Creates:
        - Topic nodes
        - Theory nodes
        - Concept nodes (extracted from keywords)
        - Relationships: (Topic)-[:HAS_THEORY]->(Theory)
        - Relationships: (Theory)-[:COVERS]->(Concept)
        """
        if self.driver is None:
            print("Neo4j not available")
            return False
        
        print(f"📚 Loading lab_tutor content from {json_file_path}...")
        
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
            
            topics = data.get('topics', [])
            theories = data.get('theories', [])
            
            # Create Topics
            for topic in topics:
                query = """
                MERGE (t:Topic {name: $name})
                SET t.description = $description,
                    t.created_at = $created_at,
                    t.source = 'lab_tutor'
                RETURN t
                """
                self._execute_write(query, {
                    'name': topic['name'],
                    'description': topic.get('description', ''),
                    'created_at': datetime.now().isoformat()
                })
            
            print(f"  ✅ Created {len(topics)} Topic nodes")
            
            # Create Theories and link to Topics
            for theory in theories:
                # Create Theory node
                query = """
                MERGE (th:Theory {id: $id})
                SET th.original_text = $original_text,
                    th.compressed_text = $compressed_text,
                    th.source = $source,
                    th.keywords = $keywords,
                    th.created_at = $created_at
                RETURN th
                """
                self._execute_write(query, {
                    'id': theory['id'],
                    'original_text': theory.get('original_text', ''),
                    'compressed_text': theory.get('compressed_text', ''),
                    'source': theory.get('source', ''),
                    'keywords': json.dumps(theory.get('keywords', [])),
                    'created_at': datetime.now().isoformat()
                })
                
                # Link Theory to Topic
                for topic_name in theory.get('topics', []):
                    query = """
                    MATCH (t:Topic {name: $topic_name})
                    MATCH (th:Theory {id: $theory_id})
                    MERGE (t)-[:HAS_THEORY]->(th)
                    """
                    self._execute_write(query, {
                        'topic_name': topic_name,
                        'theory_id': theory['id']
                    })
                
                # Create Concept nodes from keywords
                for keyword in theory.get('keywords', []):
                    query = """
                    MERGE (c:Concept {name: $name})
                    SET c.definition = $definition,
                        c.source = 'lab_tutor',
                        c.created_at = $created_at
                    WITH c
                    MATCH (th:Theory {id: $theory_id})
                    MERGE (th)-[:COVERS]->(c)
                    """
                    self._execute_write(query, {
                        'name': keyword,
                        'definition': f"Concept from {theory.get('source', 'theory')}",
                        'theory_id': theory['id'],
                        'created_at': datetime.now().isoformat()
                    })
            
            print(f"  ✅ Created {len(theories)} Theory nodes with Concepts")
            
            # Link Concepts to Topics
            query = """
            MATCH (t:Topic)-[:HAS_THEORY]->(th:Theory)-[:COVERS]->(c:Concept)
            MERGE (t)-[:INCLUDES_CONCEPT]->(c)
            """
            self._execute_write(query)
            
            print("  ✅ Linked Concepts to Topics")
            print("✅ Lab tutor content loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading lab_tutor content: {e}")
            return False
    
    def create_student(self, student_data: Dict[str, Any]) -> bool:
        """
        Create a Student node in the knowledge graph.
        
        Args:
            student_data: Dictionary with student information
                - student_id (required)
                - name (required)
                - email
                - grade_level
                - enrollment_date
        """
        if self.driver is None:
            return False
        
        try:
            query = """
            MERGE (s:Student {student_id: $student_id})
            SET s.name = $name,
                s.email = $email,
                s.grade_level = $grade_level,
                s.enrollment_date = $enrollment_date,
                s.created_at = $created_at,
                s.updated_at = $updated_at
            RETURN s
            """
            
            self._execute_write(query, {
                'student_id': student_data['student_id'],
                'name': student_data['name'],
                'email': student_data.get('email', ''),
                'grade_level': student_data.get('grade_level', ''),
                'enrollment_date': student_data.get('enrollment_date', datetime.now().isoformat()),
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            })
            
            print(f"  ✅ Created Student: {student_data['name']}")
            return True

        except Exception as e:
            print(f"  ❌ Error creating student: {e}")
            return False

    def create_teacher(self, teacher_data: Dict[str, Any]) -> bool:
        """Create a Teacher node in the knowledge graph."""
        if self.driver is None:
            return False

        try:
            query = """
            MERGE (t:Teacher {teacher_id: $teacher_id})
            SET t.name = $name,
                t.email = $email,
                t.department = $department,
                t.created_at = $created_at,
                t.updated_at = $updated_at
            RETURN t
            """

            self._execute_write(query, {
                'teacher_id': teacher_data['teacher_id'],
                'name': teacher_data['name'],
                'email': teacher_data.get('email', ''),
                'department': teacher_data.get('department', ''),
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            })

            print(f"  ✅ Created Teacher: {teacher_data['name']}")
            return True

        except Exception as e:
            print(f"  ❌ Error creating teacher: {e}")
            return False

    def enroll_student_in_topic(self, student_id: str, topic_name: str) -> bool:
        """Create ENROLLED_IN relationship between Student and Topic."""
        if self.driver is None:
            return False

        try:
            query = """
            MATCH (s:Student {student_id: $student_id})
            MATCH (t:Topic {name: $topic_name})
            MERGE (s)-[r:ENROLLED_IN]->(t)
            SET r.enrolled_at = $enrolled_at,
                r.status = 'active',
                r.progress = 0.0
            RETURN r
            """

            self._execute_write(query, {
                'student_id': student_id,
                'topic_name': topic_name,
                'enrolled_at': datetime.now().isoformat()
            })

            return True

        except Exception as e:
            print(f"  ❌ Error enrolling student: {e}")
            return False

    def create_assessment(self, assessment_data: Dict[str, Any]) -> bool:
        """
        Create an Assessment template node and link to Topic.

        Assessment is the pre-topic diagnostic test template.
        Can be used by any student (not tied to specific student).
        """
        if self.driver is None:
            return False

        try:
            # Check if this is a template (no student_id) or student-specific
            if 'student_id' in assessment_data and assessment_data['student_id']:
                # Student-specific assessment (completed)
                query = """
                MERGE (a:Assessment {assessment_id: $assessment_id})
                SET a.topic_name = $topic_name,
                    a.student_id = $student_id,
                    a.questions = $questions,
                    a.answers = $answers,
                    a.score = $score,
                    a.mastery_levels = $mastery_levels,
                    a.completed_at = $completed_at,
                    a.is_diagnostic = true
                WITH a
                MATCH (t:Topic {name: $topic_name})
                MERGE (a)-[:ASSESSES]->(t)
                WITH a
                MATCH (s:Student {student_id: $student_id})
                MERGE (s)-[:TOOK_ASSESSMENT]->(a)
                RETURN a
                """

                self._execute_write(query, {
                    'assessment_id': assessment_data['assessment_id'],
                    'topic_name': assessment_data['topic_name'],
                    'student_id': assessment_data['student_id'],
                    'questions': json.dumps(assessment_data.get('questions', [])),
                    'answers': json.dumps(assessment_data.get('answers', [])),
                    'score': assessment_data.get('score', 0.0),
                    'mastery_levels': json.dumps(assessment_data.get('mastery_levels', {})),
                    'completed_at': datetime.now().isoformat()
                })
            else:
                # Assessment template (not tied to student)
                query = """
                MERGE (a:Assessment {assessment_id: $assessment_id})
                SET a.topic_name = $topic_name,
                    a.title = $title,
                    a.questions = $questions,
                    a.total_questions = $total_questions,
                    a.is_graded = $is_graded,
                    a.time_limit = $time_limit,
                    a.is_diagnostic = true,
                    a.created_at = $created_at
                WITH a
                MATCH (t:Topic {name: $topic_name})
                MERGE (a)-[:ASSESSES]->(t)
                RETURN a
                """

                self._execute_write(query, {
                    'assessment_id': assessment_data['assessment_id'],
                    'topic_name': assessment_data['topic_name'],
                    'title': assessment_data.get('title', f"Assessment: {assessment_data['topic_name']}"),
                    'questions': json.dumps(assessment_data.get('questions', [])),
                    'total_questions': assessment_data.get('total_questions', 0),
                    'is_graded': assessment_data.get('is_graded', False),
                    'time_limit': assessment_data.get('time_limit', 900),
                    'created_at': datetime.now().isoformat()
                })

            return True

        except Exception as e:
            print(f"  ❌ Error creating assessment: {e}")
            return False

    def create_quiz(self, quiz_data: Dict[str, Any]) -> bool:
        """
        Create a Quiz node and link to Topic, Theory, and Concepts.

        Quiz tests concepts under the theory part (NOT labs).
        Relationships created:
        - (Quiz)-[:TESTS]->(Topic) - Quiz evaluates topic
        - (Quiz)-[:TESTS]->(Theory) - Quiz tests theory knowledge
        - (Quiz)-[:TESTS]->(Concept) - Quiz tests specific concepts
        """
        if self.driver is None:
            return False

        try:
            # Create quiz node and link to topic
            query = """
            MERGE (q:Quiz {quiz_id: $quiz_id})
            SET q.topic_name = $topic_name,
                q.title = $title,
                q.questions = $questions,
                q.total_questions = $total_questions,
                q.passing_score = $passing_score,
                q.time_limit = $time_limit,
                q.is_graded = true,
                q.created_at = $created_at
            WITH q
            MATCH (t:Topic {name: $topic_name})
            MERGE (q)-[:TESTS]->(t)
            RETURN q
            """

            self._execute_write(query, {
                'quiz_id': quiz_data['quiz_id'],
                'topic_name': quiz_data['topic_name'],
                'title': quiz_data.get('title', f"Quiz: {quiz_data['topic_name']}"),
                'questions': json.dumps(quiz_data.get('questions', [])),
                'total_questions': quiz_data.get('total_questions', 0),
                'passing_score': quiz_data.get('passing_score', 0.7),
                'time_limit': quiz_data.get('time_limit', 1800),
                'created_at': datetime.now().isoformat()
            })

            # Link quiz to theory (quiz tests theory knowledge)
            theory_query = """
            MATCH (q:Quiz {quiz_id: $quiz_id})
            MATCH (t:Topic {name: $topic_name})-[:HAS_THEORY]->(th:Theory)
            MERGE (q)-[:TESTS]->(th)
            """
            self._execute_write(theory_query, {
                'quiz_id': quiz_data['quiz_id'],
                'topic_name': quiz_data['topic_name']
            })

            # Link quiz to concepts (quiz tests specific concepts from theory)
            concepts_query = """
            MATCH (q:Quiz {quiz_id: $quiz_id})
            MATCH (t:Topic {name: $topic_name})-[:HAS_THEORY]->(th:Theory)-[:COVERS]->(c:Concept)
            MERGE (q)-[:TESTS]->(c)
            """
            self._execute_write(concepts_query, {
                'quiz_id': quiz_data['quiz_id'],
                'topic_name': quiz_data['topic_name']
            })

            return True

        except Exception as e:
            print(f"  ❌ Error creating quiz: {e}")
            return False

    def create_lab(self, lab_data: Dict[str, Any]) -> bool:
        """
        Create a Lab node and link to Topic, Theory, and Concepts.

        Lab covers theory and relevant concepts (hands-on practice).
        Relationships created:
        - (Lab)-[:PRACTICES]->(Topic) - Lab practices topic
        - (Lab)-[:COVERS]->(Theory) - Lab covers theory
        - (Lab)-[:APPLIES]->(Concept) - Lab applies specific concepts
        """
        if self.driver is None:
            return False

        try:
            # Create lab node and link to topic
            query = """
            MERGE (l:Lab {lab_id: $lab_id})
            SET l.topic_name = $topic_name,
                l.title = $title,
                l.objective = $objective,
                l.difficulty = $difficulty,
                l.estimated_time = $estimated_time,
                l.instructions = $instructions,
                l.concepts_covered = $concepts_covered,
                l.created_at = $created_at
            WITH l
            MATCH (t:Topic {name: $topic_name})
            MERGE (l)-[:PRACTICES]->(t)
            RETURN l
            """

            self._execute_write(query, {
                'lab_id': lab_data['lab_id'],
                'topic_name': lab_data['topic_name'],
                'title': lab_data['title'],
                'objective': lab_data.get('objective', ''),
                'difficulty': lab_data.get('difficulty', 'intermediate'),
                'estimated_time': lab_data.get('estimated_time', 45),
                'instructions': json.dumps(lab_data.get('instructions', [])),
                'concepts_covered': json.dumps(lab_data.get('concepts_covered', [])),
                'created_at': datetime.now().isoformat()
            })

            # Link lab to theory (lab covers theory)
            theory_query = """
            MATCH (l:Lab {lab_id: $lab_id})
            MATCH (t:Topic {name: $topic_name})-[:HAS_THEORY]->(th:Theory)
            MERGE (l)-[:COVERS]->(th)
            """
            self._execute_write(theory_query, {
                'lab_id': lab_data['lab_id'],
                'topic_name': lab_data['topic_name']
            })

            # Link lab to concepts (lab applies concepts from theory)
            # If specific concepts are provided, use those; otherwise link to all concepts from theory
            if lab_data.get('concepts_covered'):
                # Link to specific concepts
                for concept_name in lab_data['concepts_covered']:
                    concept_query = """
                    MATCH (l:Lab {lab_id: $lab_id})
                    MATCH (c:Concept {name: $concept_name})
                    MERGE (l)-[:APPLIES]->(c)
                    """
                    self._execute_write(concept_query, {
                        'lab_id': lab_data['lab_id'],
                        'concept_name': concept_name
                    })
            else:
                # Link to all concepts from the topic's theory
                all_concepts_query = """
                MATCH (l:Lab {lab_id: $lab_id})
                MATCH (t:Topic {name: $topic_name})-[:HAS_THEORY]->(th:Theory)-[:COVERS]->(c:Concept)
                MERGE (l)-[:APPLIES]->(c)
                """
                self._execute_write(all_concepts_query, {
                    'lab_id': lab_data['lab_id'],
                    'topic_name': lab_data['topic_name']
                })

            return True

        except Exception as e:
            print(f"  ❌ Error creating lab: {e}")
            return False

    def create_reading_material(self, reading_data: Dict[str, Any]) -> bool:
        """Create a ReadingMaterial/Blog node and link to Concept."""
        if self.driver is None:
            return False

        try:
            query = """
            MERGE (r:ReadingMaterial {reading_id: $reading_id})
            SET r.concept_name = $concept_name,
                r.title = $title,
                r.content = $content,
                r.source = $source,
                r.generated_by = $generated_by,
                r.created_at = $created_at
            WITH r
            MATCH (c:Concept {name: $concept_name})
            MERGE (r)-[:EXPLAINS]->(c)
            RETURN r
            """

            self._execute_write(query, {
                'reading_id': reading_data['reading_id'],
                'concept_name': reading_data['concept_name'],
                'title': reading_data.get('title', ''),
                'content': reading_data.get('content', ''),
                'source': reading_data.get('source', 'generated'),
                'generated_by': reading_data.get('generated_by', 'llm'),
                'created_at': datetime.now().isoformat()
            })

            return True

        except Exception as e:
            print(f"  ❌ Error creating reading material: {e}")
            return False

    def create_personalized_learning_path(self, student_id: str, topic_name: str,
                                         concept_priorities: Dict[str, Dict]) -> bool:
        """
        Create personalized learning path relationships.

        Links Student to Concepts with priority and mastery information.
        """
        if self.driver is None:
            return False

        try:
            for concept_name, data in concept_priorities.items():
                query = """
                MATCH (s:Student {student_id: $student_id})
                MATCH (c:Concept {name: $concept_name})
                MERGE (s)-[r:PERSONALIZED_PATH]->(c)
                SET r.topic_name = $topic_name,
                    r.mastery = $mastery,
                    r.priority = $priority,
                    r.status = $status,
                    r.recommendation = $recommendation,
                    r.created_at = $created_at,
                    r.updated_at = $updated_at
                RETURN r
                """

                self._execute_write(query, {
                    'student_id': student_id,
                    'concept_name': concept_name,
                    'topic_name': topic_name,
                    'mastery': data.get('mastery', 0.0),
                    'priority': data.get('priority', 'medium'),
                    'status': data.get('status', 'not_started'),
                    'recommendation': data.get('recommendation', ''),
                    'created_at': datetime.now().isoformat(),
                    'updated_at': datetime.now().isoformat()
                })

            return True

        except Exception as e:
            print(f"  ❌ Error creating personalized path: {e}")
            return False

    def record_quiz_attempt(self, student_id: str, quiz_id: str,
                           score: float, answers: List[Dict]) -> bool:
        """Record a student's quiz attempt."""
        if self.driver is None:
            return False

        try:
            query = """
            MATCH (s:Student {student_id: $student_id})
            MATCH (q:Quiz {quiz_id: $quiz_id})
            CREATE (s)-[r:TOOK_QUIZ]->(q)
            SET r.score = $score,
                r.answers = $answers,
                r.passed = $passed,
                r.completed_at = $completed_at
            RETURN r
            """

            passed = score >= 0.7  # 70% passing score

            self._execute_write(query, {
                'student_id': student_id,
                'quiz_id': quiz_id,
                'score': score,
                'answers': json.dumps(answers),
                'passed': passed,
                'completed_at': datetime.now().isoformat()
            })

            return True

        except Exception as e:
            print(f"  ❌ Error recording quiz attempt: {e}")
            return False

    def record_lab_completion(self, student_id: str, lab_id: str,
                             completion_data: Dict[str, Any]) -> bool:
        """Record a student's lab completion."""
        if self.driver is None:
            return False

        try:
            query = """
            MATCH (s:Student {student_id: $student_id})
            MATCH (l:Lab {lab_id: $lab_id})
            CREATE (s)-[r:COMPLETED_LAB]->(l)
            SET r.completed_at = $completed_at,
                r.time_spent = $time_spent,
                r.notes = $notes,
                r.verified = $verified
            RETURN r
            """

            self._execute_write(query, {
                'student_id': student_id,
                'lab_id': lab_id,
                'completed_at': datetime.now().isoformat(),
                'time_spent': completion_data.get('time_spent', 0),
                'notes': completion_data.get('notes', ''),
                'verified': completion_data.get('verified', False)
            })

            return True

        except Exception as e:
            print(f"  ❌ Error recording lab completion: {e}")
            return False

    def get_complete_graph_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the knowledge graph."""
        if self.driver is None:
            return {}

        try:
            # Count all node types
            node_query = """
            MATCH (n)
            RETURN labels(n)[0] as label, count(n) as count
            ORDER BY count DESC
            """
            node_results = self._execute_query(node_query)

            # Count all relationship types
            rel_query = """
            MATCH ()-[r]->()
            RETURN type(r) as type, count(r) as count
            ORDER BY count DESC
            """
            rel_results = self._execute_query(rel_query)

            return {
                'nodes': {row['label']: row['count'] for row in node_results},
                'relationships': {row['type']: row['count'] for row in rel_results},
                'total_nodes': sum(row['count'] for row in node_results),
                'total_relationships': sum(row['count'] for row in rel_results)
            }

        except Exception as e:
            print(f"  ❌ Error getting graph stats: {e}")
            return {}

    def visualize_student_learning_path(self, student_id: str) -> Dict[str, Any]:
        """
        Get visualization data for a student's complete learning path.

        Returns all nodes and relationships connected to the student.
        """
        if self.driver is None:
            return {}

        try:
            query = """
            MATCH path = (s:Student {student_id: $student_id})-[*1..3]-(n)
            RETURN path
            LIMIT 500
            """

            results = self._execute_query(query, {'student_id': student_id})

            return {
                'student_id': student_id,
                'paths': results,
                'path_count': len(results)
            }

        except Exception as e:
            print(f"  ❌ Error visualizing learning path: {e}")
            return {}


# Global instance
_integrated_graph_service = None


def get_integrated_graph_service() -> IntegratedGraphService:
    """Get the global IntegratedGraphService instance (singleton pattern)."""
    global _integrated_graph_service
    if _integrated_graph_service is None:
        _integrated_graph_service = IntegratedGraphService()
    return _integrated_graph_service

